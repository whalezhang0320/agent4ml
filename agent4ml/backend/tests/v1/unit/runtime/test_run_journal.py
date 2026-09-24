import json
import shutil
from pathlib import Path
from uuid import uuid4

from agent4ml.backend.agents.journal.events import RunEvent
from agent4ml.backend.agents.journal.run_journal import RunJournal


def test_run_journal_appends_jsonl_events() -> None:
    temp_dir = _workspace_temp_dir()
    journal = RunJournal(run_id="run-1", events_path=temp_dir / "events.jsonl")

    event = journal.append("run.started", {"mode": "general"})

    assert isinstance(event, RunEvent)
    assert event.run_id == "run-1"
    content = (temp_dir / "events.jsonl").read_text(encoding="utf-8")
    blocks = [b for b in content.split("\n\n") if b.strip()]
    assert len(blocks) == 1
    payload = json.loads(blocks[0])
    assert payload["event_type"] == "run.started"
    assert payload["payload"] == {"mode": "general"}

    shutil.rmtree(temp_dir)


def _workspace_temp_dir() -> Path:
    path = Path(".agent4ml/test-logs") / uuid4().hex
    path.mkdir(parents=True, exist_ok=False)
    return path
