"""Training submission or local execution stage."""
from __future__ import annotations

from typing import Callable

from agent4ml.backend.agents.ml_research.command_executor import CommandExecutor
from agent4ml.backend.agents.ml_research.stages.base import node_artifact_dir
from agent4ml.backend.agents.ml_research.task_models import NodeRecord, TaskRecord
from agent4ml.backend.agents.ml_research.tracking import LocalExperimentTracker
from agent4ml.backend.agents.ml_research.workflow import (
    NodeResult,
    TrainingBackend,
)


class RunTrainingHandler:
    def __init__(
        self,
        executor: CommandExecutor | None = None,
        training_backend: TrainingBackend | None = None,
        cancel_requested: Callable[[str], bool] | None = None,
    ) -> None:
        self.executor = executor or CommandExecutor()
        self.training_backend = training_backend
        self.cancel_requested = cancel_requested

    def execute(self, task: TaskRecord, node: NodeRecord) -> NodeResult:
        idempotency_key = f"{task.task_id}:{node.node_id}:{node.attempt}"
        if self.training_backend is not None:
            config_path = task.artifacts.get("train_config", "")
            job_id = self.training_backend.submit(
                config_path=config_path,
                idempotency_key=idempotency_key,
                inputs=task.inputs,
            )
            return NodeResult.waiting_external(job_id)
        command = task.inputs.get("training_command") or []
        if not command:
            return NodeResult.failed(
                "config_missing", "training_command or external training backend is required"
            )
        run_dir = node_artifact_dir(task, node)
        tracker = LocalExperimentTracker(run_dir)
        tracker.initialize(
            name=f"{task.name}:{node.node_id}",
            command=command,
            cwd=task.cwd,
            metadata={
                "task_id": task.task_id,
                "node_id": node.node_id,
                "attempt": node.attempt,
                "idempotency_key": idempotency_key,
            },
            workflow_type=task.workflow_type,
            workflow_version=task.workflow_version,
            node_id=node.node_id,
            attempt=node.attempt,
        )
        executed = self.executor.execute(
            command,
            cwd=task.cwd,
            tracker=tracker,
            should_cancel=(
                lambda: self.cancel_requested(task.task_id)
                if self.cancel_requested is not None
                else False
            ),
        )
        if executed.exit_code != 0:
            return NodeResult.failed(
                executed.failure_class or "unknown",
                executed.error or "training failed",
            )
        return NodeResult.succeeded({"training_run": str(run_dir / "manifest.json")})
