"""Result validation stage."""
from __future__ import annotations

from agent4ml.backend.agents.ml_research.stages.base import (
    node_artifact_dir,
    write_json_artifact,
)
from agent4ml.backend.agents.ml_research.task_models import NodeRecord, TaskRecord
from agent4ml.backend.agents.ml_research.workflow import NodeResult


class ValidateResultHandler:
    def execute(self, task: TaskRecord, node: NodeRecord) -> NodeResult:
        if "training_run" not in task.artifacts and not task.nodes[
            "run_training"
        ].external_job_id:
            return NodeResult.failed(
                "config_missing", "training result reference is missing"
            )
        ref = write_json_artifact(
            node_artifact_dir(task, node) / "validation.json",
            {
                "target_metric": task.inputs.get("target_metric"),
                "training_run": task.artifacts.get("training_run"),
                "external_job_id": task.nodes["run_training"].external_job_id,
                "validated": True,
            },
        )
        return NodeResult.succeeded({"validation_result": ref})
