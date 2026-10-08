"""Shared helpers for filesystem-backed workflow stages."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agent4ml.backend.agents.ml_research.task_models import NodeRecord, TaskRecord


def node_artifact_dir(task: TaskRecord, node: NodeRecord) -> Path:
    path = Path(task.run_dir) / "nodes" / node.node_id / f"attempt-{node.attempt}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json_artifact(path: Path, payload: dict[str, Any]) -> str:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)
    return str(path)
