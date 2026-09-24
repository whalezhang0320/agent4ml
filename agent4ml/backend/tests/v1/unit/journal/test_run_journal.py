import json
import shutil
from pathlib import Path
from uuid import uuid4

from agent4ml.backend.agents.journal.run_journal import RunJournal


def test_run_journal_writes_event_jsonl() -> None:
    temp_dir = _workspace_temp_dir()
    journal = RunJournal(run_id="run-1", events_path=temp_dir / "events.jsonl")

    journal.append("run.started", {"mode": "general"})

    content = (temp_dir / "events.jsonl").read_text(encoding="utf-8")
    # Events are now pretty-printed JSON blocks separated by blank lines.
    blocks = [b for b in content.split("\n\n") if b.strip()]
    assert json.loads(blocks[0])["event_type"] == "run.started"

    shutil.rmtree(temp_dir)


def _workspace_temp_dir() -> Path:
    path = Path(".agent4ml/test-logs") / uuid4().hex
    path.mkdir(parents=True, exist_ok=False)
    return path
