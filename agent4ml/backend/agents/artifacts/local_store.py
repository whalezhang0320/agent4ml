from __future__ import annotations

from pathlib import Path
from typing import Any

from agent4ml.backend.agents.journal.events import utc_now_iso
from agent4ml.backend.agents.state.types import Artifact


class LocalArtifactStore:
    def save_artifact(
        self,
        content: str,
        output_dir: str | Path,
        title: str,
        filename: str,
        metadata: dict[str, Any] | None = None,
    ) -> Artifact:
        artifacts_dir = Path(output_dir) / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        path = artifacts_dir / filename
        path.write_text(content, encoding="utf-8")
        artifact_id = path.stem
        return Artifact(
            artifact_id=artifact_id,
            artifact_type="report_markdown",
            title=title,
            path=str(path),
            summary=(metadata or {}).get("summary", ""),
            created_at=utc_now_iso(),
        )

    def get_artifact(self, path: str | Path) -> str:
        return Path(path).read_text(encoding="utf-8")
