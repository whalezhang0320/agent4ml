"""Environment preparation stage."""
from __future__ import annotations

import platform
import sys
from pathlib import Path
from typing import Callable

from agent4ml.backend.agents.ml_research.command_executor import CommandExecutor
from agent4ml.backend.agents.ml_research.stages.base import (
    node_artifact_dir,
    write_json_artifact,
)
from agent4ml.backend.agents.ml_research.task_models import NodeRecord, TaskRecord
from agent4ml.backend.agents.ml_research.tracking import LocalExperimentTracker
from agent4ml.backend.agents.ml_research.workflow import NodeResult


class BuildEnvironmentHandler:
    def __init__(
        self,
        executor: CommandExecutor | None = None,
        cancel_requested: Callable[[str], bool] | None = None,
    ) -> None:
        self.executor = executor or CommandExecutor()
        self.cancel_requested = cancel_requested

    def execute(self, task: TaskRecord, node: NodeRecord) -> NodeResult:
        output_dir = node_artifact_dir(task, node)
        command = task.inputs.get("build_command") or []
        if command:
            tracker = LocalExperimentTracker(output_dir)
            tracker.initialize(
                name=f"{task.name}:{node.node_id}",
                command=command,
                cwd=task.cwd,
                metadata={
                    "task_id": task.task_id,
                    "node_id": node.node_id,
                    "attempt": node.attempt,
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
                    executed.error or "environment build failed",
                )
        ref = write_json_artifact(
            output_dir / "environment.json",
            {
                "python": sys.version,
                "platform": platform.platform(),
                "build_command": list(command),
                "repository_path": str(Path(task.cwd)),
            },
        )
        return NodeResult.succeeded({"environment_manifest": ref})
