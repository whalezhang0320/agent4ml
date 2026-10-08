"""Repository analysis stage."""
from __future__ import annotations

import hashlib
from pathlib import Path

from agent4ml.backend.agents.ml_research.stages.base import (
    node_artifact_dir,
    write_json_artifact,
)
from agent4ml.backend.agents.ml_research.task_models import NodeRecord, TaskRecord
from agent4ml.backend.agents.ml_research.workflow import NodeResult


class AnalyzeCodeHandler:
    def execute(self, task: TaskRecord, node: NodeRecord) -> NodeResult:
        repository = Path(str(task.inputs["repository_path"]))
        if not repository.is_dir():
            return NodeResult.failed("config_missing", "repository_path no longer exists")
        manifests = [
            name
            for name in (
                "pyproject.toml",
                "requirements.txt",
                "environment.yml",
                "setup.py",
                "Dockerfile",
            )
            if (repository / name).is_file()
        ]
        entrypoints = sorted(
            str(path.relative_to(repository))
            for pattern in ("train*.py", "main.py", "run*.sh")
            for path in repository.glob(pattern)
            if path.is_file()
        )
        digest = hashlib.sha256()
        for name in manifests:
            path = repository / name
            digest.update(name.encode())
            digest.update(path.read_bytes())
        output = node_artifact_dir(task, node) / "repository-analysis.json"
        ref = write_json_artifact(
            output,
            {
                "repository_path": str(repository),
                "dependency_manifests": manifests,
                "candidate_entrypoints": entrypoints,
                "input_hash": digest.hexdigest(),
            },
        )
        return NodeResult.succeeded({"repository_analysis": ref})
