"""Redis-queue worker that executes local ML tasks and supports safe cancellation."""
from __future__ import annotations

import json
import os
import queue
import signal
import subprocess
import threading
import time
import uuid
from dataclasses import replace
from pathlib import Path
from typing import Any

from agent4ml.backend.agents.ml_research.failure_eval import FailureClassifier
from agent4ml.backend.agents.ml_research.task_models import (
    TERMINAL_TASK_STATUSES,
    InvalidTaskTransition,
    NodeStatus,
    QueueMessage,
    TaskRecord,
    TaskStatus,
    utc_now_iso,
)
from agent4ml.backend.agents.ml_research.task_store import TaskStore
from agent4ml.backend.agents.ml_research.tracking import LocalExperimentTracker, METRIC_PREFIX
from agent4ml.backend.agents.ml_research.workflow import NodeHandler, NodeResult, get_workflow
from agent4ml.backend.agents.ml_research.workflow_engine import WorkflowEngine
from agent4ml.backend.agents.ml_research.stages import (
    AnalyzeCodeHandler,
    BuildEnvironmentHandler,
    RunTrainingHandler,
    ValidateResultHandler,
)


_OUTPUT_EOF = object()


def _process_group_options() -> dict[str, Any]:
    if os.name == "nt":
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    return {"start_new_session": True}


def _terminate_process_group(process: subprocess.Popen[str], grace_seconds: float) -> None:
    if process.poll() is not None:
        return
    if os.name == "nt":
        try:
            process.send_signal(signal.CTRL_BREAK_EVENT)
        except (AttributeError, OSError):
            process.terminate()
    else:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
    try:
        process.wait(timeout=grace_seconds)
        return
    except subprocess.TimeoutExpired:
        pass
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            capture_output=True,
            check=False,
        )
    else:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            return
    process.wait(timeout=max(1.0, grace_seconds))


def _read_output(stream: Any, output_queue: queue.Queue[object]) -> None:
    try:
        for line in stream:
            output_queue.put(line)
    finally:
        output_queue.put(_OUTPUT_EOF)


def _metric_payload(line: str) -> dict[str, Any] | None:
    stripped = line.strip()
    if not stripped.startswith(METRIC_PREFIX):
        return None
    try:
        value = json.loads(stripped[len(METRIC_PREFIX):])
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


class MLTaskWorker:
    def __init__(
        self,
        store: TaskStore,
        *,
        worker_id: str | None = None,
        heartbeat_interval: float = 5.0,
        cancel_grace_seconds: float = 10.0,
        handlers: dict[str, NodeHandler] | None = None,
    ) -> None:
        self.store = store
        self.worker_id = worker_id or f"worker-{uuid.uuid4().hex[:12]}"
        self.heartbeat_interval = heartbeat_interval
        self.cancel_grace_seconds = cancel_grace_seconds
        self._stop = threading.Event()
        self._current_process: subprocess.Popen[str] | None = None
        self.engine = WorkflowEngine(store)
        self.handlers: dict[str, NodeHandler] = handlers or {
            "analyze_code": AnalyzeCodeHandler(),
            "build_environment": BuildEnvironmentHandler(
                cancel_requested=store.cancel_requested
            ),
            "run_training": RunTrainingHandler(
                cancel_requested=store.cancel_requested
            ),
            "validate_result": ValidateResultHandler(),
        }

    def stop(self) -> None:
        self._stop.set()

    def run_forever(self, *, poll_ms: int = 1000) -> None:
        while not self._stop.is_set():
            self.run_once(block_ms=poll_ms)

    def run_once(self, *, block_ms: int = 0) -> bool:
        claimed = self.store.claim(self.worker_id, block_ms=block_ms)
        if claimed is None:
            return False
        if isinstance(claimed, QueueMessage):
            message = claimed
        else:  # Compatibility with custom stores implementing the original tuple API.
            message_id, task_id, recovered = claimed
            message = QueueMessage(str(message_id), str(task_id), bool(recovered))
        message_id = message.message_id
        task_id = message.task_id
        recovered = message.recovered
        try:
            task = self.store.get(task_id)
            if task is None or task.status in TERMINAL_TASK_STATUSES:
                return True
            if task.status is TaskStatus.CANCELLING:
                self.store.transition(task_id, TaskStatus.CANCELLED)
                self.store.publish(task_id, "task.cancelled", {"status": "cancelled"})
                return True
            if task.is_workflow:
                self._run_workflow_message(task, message)
                return True
            if recovered and task.status is TaskStatus.RUNNING:
                self.store.transition(
                    task_id,
                    TaskStatus.FAILED,
                    failure_class="worker_lost",
                    error="worker claim expired before task completion",
                )
                self.store.publish(
                    task_id,
                    "task.failed",
                    {
                        "status": "failed",
                        "failure_class": "worker_lost",
                        "error": "worker claim expired before task completion",
                    },
                )
                return True
            if task.status is not TaskStatus.QUEUED:
                self.store.publish(
                    task_id,
                    "task.skipped",
                    {"status": task.status.value, "reason": "task is not queueable"},
                )
                return True
            try:
                self._execute(task, message_id)
            except Exception as exc:
                current = self.store.get(task_id)
                if current is not None and current.status not in TERMINAL_TASK_STATUSES:
                    self.store.transition(
                        task_id,
                        TaskStatus.FAILED,
                        failure_class="worker_error",
                        error=str(exc),
                    )
                    self.store.publish(
                        task_id,
                        "task.failed",
                        {
                            "status": "failed",
                            "failure_class": "worker_error",
                            "error": str(exc),
                        },
                    )
            return True
        finally:
            self.store.ack(message_id)

    def _run_workflow_message(
        self, task: TaskRecord, message: QueueMessage
    ) -> None:
        if message.recovered and task.status is TaskStatus.RUNNING:
            self.engine.recover_expired_claim(task)
            return
        decision = self.engine.validate_message(task, message)
        if decision.is_stale:
            self.store.publish(
                task.task_id,
                "node.skipped",
                {
                    "node_id": message.node_id,
                    "reason": decision.reason,
                    "expected_version": message.expected_version,
                },
            )
            return
        assert message.node_id is not None
        assert message.expected_version is not None
        assert message.attempt is not None
        running = self.store.mark_node_running(
            task.task_id,
            message.node_id,
            expected_version=message.expected_version,
            attempt=message.attempt,
            worker_id=self.worker_id,
        )
        definition = get_workflow(
            running.workflow_type or "", running.workflow_version or 0
        ).node(message.node_id)
        heartbeat_stop = threading.Event()

        def heartbeat() -> None:
            while not heartbeat_stop.wait(self.heartbeat_interval):
                try:
                    self.store.heartbeat_node(
                        task.task_id, message.message_id, self.worker_id
                    )
                except Exception:
                    return

        heartbeat_thread = threading.Thread(
            target=heartbeat,
            name=f"{self.worker_id}-workflow-heartbeat",
            daemon=True,
        )
        heartbeat_thread.start()
        try:
            handler = self.handlers[definition.handler]
            result = handler.execute(running, running.nodes[message.node_id])
        except Exception as exc:
            result = NodeResult.failed("worker_error", str(exc))
        finally:
            heartbeat_stop.set()
            heartbeat_thread.join(timeout=max(0.1, self.heartbeat_interval * 2))
        current = self.store.get(task.task_id)
        if current is not None and current.status is TaskStatus.CANCELLING:
            nodes = dict(current.nodes)
            current_node = nodes[message.node_id]
            nodes[message.node_id] = replace(
                current_node,
                status=NodeStatus.SKIPPED,
                finished_at=utc_now_iso(),
                error="task cancelled",
            )
            self.store.transition(
                task.task_id,
                TaskStatus.CANCELLED,
                nodes=nodes,
                worker_id=None,
                worker_heartbeat_at=None,
                pid=None,
            )
            self.store.publish(
                task.task_id,
                "task.cancelled",
                {"status": "cancelled", "node_id": message.node_id},
            )
            return
        self.engine.apply_result(running, result)

    def _execute(self, task: TaskRecord, message_id: str) -> None:
        tracker = LocalExperimentTracker(task.run_dir)
        tracker.initialize(
            name=task.name,
            command=task.command,
            cwd=task.cwd,
            metadata={**task.metadata, "task_id": task.task_id},
            run_id=task.task_id,
        )
        running = self.store.transition(
            task.task_id,
            TaskStatus.RUNNING,
            worker_id=self.worker_id,
            worker_heartbeat_at=utc_now_iso(),
        )
        tracker.update_manifest(
            status="running", started_at=running.started_at, worker_id=self.worker_id
        )
        tracker.log_event("experiment.started", data={"command": list(task.command)})
        self.store.publish(
            task.task_id,
            "task.started",
            {"status": "running", "worker_id": self.worker_id},
        )

        classifier = FailureClassifier()
        output_queue: queue.Queue[object] = queue.Queue()
        cancel_sent = False
        reader_done = False
        last_heartbeat = 0.0
        try:
            process = subprocess.Popen(
                list(task.command),
                cwd=task.cwd,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                **_process_group_options(),
            )
            self._current_process = process
            self.store.update(task.task_id, pid=process.pid)
            tracker.update_manifest(pid=process.pid)
            assert process.stdout is not None
            reader = threading.Thread(
                target=_read_output,
                args=(process.stdout, output_queue),
                name=f"{self.worker_id}-stdout",
                daemon=True,
            )
            reader.start()

            while not reader_done or process.poll() is None:
                try:
                    item = output_queue.get(timeout=0.2)
                    if item is _OUTPUT_EOF:
                        reader_done = True
                    else:
                        line = str(item)
                        tracker.record_output_line(line)
                        self.store.publish(
                            task.task_id, "task.log", {"line": line.rstrip()[:4000]}
                        )
                        metric = _metric_payload(line)
                        if metric is not None:
                            self.store.publish(task.task_id, "metric.reported", metric)
                        detection = classifier.feed(line)
                        if detection is not None:
                            self.store.publish(
                                task.task_id,
                                "failure.detected",
                                {
                                    "failure_class": detection.failure_class.value,
                                    "evidence": detection.evidence,
                                },
                            )
                            tracker.log_event(
                                "failure.detected",
                                message=detection.evidence,
                                data={"failure_class": detection.failure_class.value},
                            )
                except queue.Empty:
                    pass

                now = time.monotonic()
                if now - last_heartbeat >= self.heartbeat_interval:
                    self.store.update(
                        task.task_id, worker_heartbeat_at=utc_now_iso(), pid=process.pid
                    )
                    self.store.touch_claim(message_id, self.worker_id)
                    last_heartbeat = now
                if (
                    not cancel_sent
                    and (self._stop.is_set() or self.store.cancel_requested(task.task_id))
                ):
                    cancel_sent = True
                    current = self.store.get(task.task_id)
                    if current is not None and current.status is TaskStatus.RUNNING:
                        try:
                            self.store.transition(task.task_id, TaskStatus.CANCELLING)
                        except InvalidTaskTransition:
                            pass
                    self.store.publish(
                        task.task_id,
                        "task.termination_started",
                        {"pid": process.pid, "grace_seconds": self.cancel_grace_seconds},
                    )
                    _terminate_process_group(process, self.cancel_grace_seconds)

            exit_code = process.wait()
            self._finish(task, tracker, exit_code, classifier, cancel_sent)
        except Exception as exc:
            if self._current_process is not None:
                _terminate_process_group(self._current_process, self.cancel_grace_seconds)
            tracker.record_output_line(f"[agent4ml worker] launch failed: {exc}")
            tracker.log_event("experiment.launch_failed", message=str(exc))
            current = self.store.get(task.task_id)
            if current and current.status in {TaskStatus.RUNNING, TaskStatus.CANCELLING}:
                try:
                    self.store.transition(
                        task.task_id,
                        TaskStatus.FAILED,
                        error=str(exc),
                        failure_class="worker_error",
                    )
                except InvalidTaskTransition:
                    pass
            self.store.publish(
                task.task_id,
                "task.failed",
                {"status": "failed", "failure_class": "worker_error", "error": str(exc)},
            )
            tracker.update_manifest(
                status="failed", failure_class="worker_error", finished_at=utc_now_iso()
            )
        finally:
            self._current_process = None

    def _finish(
        self,
        task: TaskRecord,
        tracker: LocalExperimentTracker,
        exit_code: int,
        classifier: FailureClassifier,
        cancel_sent: bool,
    ) -> None:
        current = self.store.get(task.task_id)
        cancelled = cancel_sent or (
            current is not None and current.status is TaskStatus.CANCELLING
        )
        if cancelled:
            updated = self.store.transition(
                task.task_id, TaskStatus.CANCELLED, exit_code=exit_code
            )
            event_type = "task.cancelled"
        elif exit_code == 0:
            updated = self.store.transition(
                task.task_id, TaskStatus.SUCCEEDED, exit_code=exit_code
            )
            event_type = "task.succeeded"
        else:
            failure_class = (
                classifier.primary.failure_class.value if classifier.primary else "unknown"
            )
            updated = self.store.transition(
                task.task_id,
                TaskStatus.FAILED,
                exit_code=exit_code,
                failure_class=failure_class,
                error=f"process exited with code {exit_code}",
            )
            event_type = "task.failed"
        tracker.update_manifest(
            status=updated.status.value,
            exit_code=exit_code,
            failure_class=updated.failure_class,
            finished_at=updated.finished_at,
        )
        tracker.log_event(
            "experiment.finished",
            data={
                "status": updated.status.value,
                "exit_code": exit_code,
                "failure_class": updated.failure_class,
            },
        )
        self.store.publish(
            task.task_id,
            event_type,
            {
                "status": updated.status.value,
                "exit_code": exit_code,
                "failure_class": updated.failure_class,
            },
        )
