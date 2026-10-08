"""Reusable local command execution for workflow stage handlers."""
from __future__ import annotations

import os
import queue
import signal
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from agent4ml.backend.agents.ml_research.failure_eval import FailureClassifier
from agent4ml.backend.agents.ml_research.tracking import LocalExperimentTracker


@dataclass(frozen=True)
class CommandResult:
    exit_code: int
    failure_class: str | None = None
    error: str | None = None


class CommandExecutor:
    def execute(
        self,
        command: Sequence[str],
        *,
        cwd: str | Path,
        tracker: LocalExperimentTracker,
        timeout: float | None = None,
        should_cancel: Callable[[], bool] | None = None,
        cancel_grace_seconds: float = 10.0,
    ) -> CommandResult:
        classifier = FailureClassifier()
        output: queue.Queue[str | None] = queue.Queue()
        try:
            options = (
                {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
                if os.name == "nt"
                else {"start_new_session": True}
            )
            process = subprocess.Popen(
                list(command),
                cwd=str(cwd),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                **options,
            )
            assert process.stdout is not None

            def read_output() -> None:
                try:
                    for line in process.stdout:
                        output.put(line)
                finally:
                    output.put(None)

            reader = threading.Thread(target=read_output, daemon=True)
            reader.start()
            deadline = time.monotonic() + timeout if timeout is not None else None
            reader_done = False
            while not reader_done or process.poll() is None:
                try:
                    line = output.get(timeout=0.2)
                    if line is None:
                        reader_done = True
                    else:
                        tracker.record_output_line(line)
                        classifier.feed(line)
                except queue.Empty:
                    pass
                if should_cancel is not None and should_cancel():
                    self._terminate(process, cancel_grace_seconds)
                    return CommandResult(
                        process.returncode or -15, "cancelled", "task cancelled"
                    )
                if deadline is not None and time.monotonic() >= deadline:
                    self._terminate(process, cancel_grace_seconds)
                    return CommandResult(124, "network_timeout", "command timed out")
            exit_code = process.wait()
        except OSError as exc:
            tracker.record_output_line(f"[agent4ml] command launch failed: {exc}")
            return CommandResult(127, "worker_error", str(exc))
        if exit_code == 0:
            return CommandResult(0)
        failure_class = (
            classifier.primary.failure_class.value if classifier.primary else "unknown"
        )
        return CommandResult(
            exit_code,
            failure_class,
            f"process exited with code {exit_code}",
        )

    @staticmethod
    def _terminate(process: subprocess.Popen[str], grace_seconds: float) -> None:
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
            process.kill()
        else:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                return
        process.wait(timeout=max(1.0, grace_seconds))
