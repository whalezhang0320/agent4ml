import shutil
from pathlib import Path
from uuid import uuid4

from agent4ml.backend.agents.artifacts.local_store import LocalArtifactStore


def test_local_artifact_store_saves_final_report() -> None:
    temp_dir = _workspace_temp_dir()
    store = LocalArtifactStore()

    artifact = store.save_artifact(
        content="# Report",
        output_dir=temp_dir,
        title="Final Report",
        filename="final_report.md",
        metadata={"mode": "general"},
    )

    assert artifact.artifact_id == "final_report"
    assert artifact.artifact_type == "report_markdown"
    assert artifact.path == str(temp_dir / "artifacts" / "final_report.md")
    assert (temp_dir / "artifacts" / "final_report.md").read_text(encoding="utf-8") == "# Report"

    shutil.rmtree(temp_dir)


def _workspace_temp_dir() -> Path:
    path = Path(".agent4ml/test-logs") / uuid4().hex
    path.mkdir(parents=True, exist_ok=False)
    return path
