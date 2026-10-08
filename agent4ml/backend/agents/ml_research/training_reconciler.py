"""Reconcile persisted external training jobs without occupying an agent loop."""
from __future__ import annotations

from dataclasses import replace

from agent4ml.backend.agents.ml_research.task_models import (
    NodeStatus,
    TaskStatus,
    utc_now_iso,
)
from agent4ml.backend.agents.ml_research.task_store import TaskStore
from agent4ml.backend.agents.ml_research.workflow import NodeResult, TrainingBackend
from agent4ml.backend.agents.ml_research.workflow_engine import WorkflowEngine


class TrainingReconciler:
    def __init__(self, store: TaskStore, backend: TrainingBackend) -> None:
        self.store = store
        self.backend = backend
        self.engine = WorkflowEngine(store)

    def run_once(self) -> int:
        reconciled = 0
        for task in self.store.waiting_external():
            if task.current_node is None:
                continue
            node = task.nodes[task.current_node]
            if not node.external_job_id:
                continue
            if task.status is TaskStatus.CANCELLING:
                cancel = getattr(self.backend, "cancel", None)
                if not callable(cancel):
                    continue
                cancel(node.external_job_id)
                nodes = dict(task.nodes)
                nodes[node.node_id] = replace(
                    node,
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
                )
                self.store.publish(
                    task.task_id,
                    "task.cancelled",
                    {
                        "status": "cancelled",
                        "node_id": node.node_id,
                        "external_job_id": node.external_job_id,
                    },
                )
                reconciled += 1
                continue
            status = self.backend.get_status(node.external_job_id).upper()
            self.store.publish(
                task.task_id,
                "training.status_changed",
                {
                    "node_id": node.node_id,
                    "external_job_id": node.external_job_id,
                    "status": status,
                },
            )
            if status == "SUCCEEDED":
                self.engine.apply_result(
                    task,
                    NodeResult.succeeded(
                        {"training_run": f"external://{node.external_job_id}"}
                    ),
                )
                self.store.publish(
                    task.task_id,
                    "training.completed",
                    {"external_job_id": node.external_job_id, "status": status},
                )
            elif status == "FAILED":
                self.engine.apply_result(
                    task,
                    NodeResult.failed(
                        "unknown", f"external training job {node.external_job_id} failed"
                    ),
                )
            elif status not in {"RUNNING", "QUEUED"}:
                continue
            reconciled += 1
        return reconciled
