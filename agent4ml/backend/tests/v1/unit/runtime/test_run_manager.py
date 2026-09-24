import json
import shutil
from pathlib import Path
from uuid import uuid4

from agent4ml.backend.agents.config.loader import load_config
from agent4ml.backend.agents.runtime.run_manager import RunManager
from agent4ml.backend.agents.runtime.run_record import RunStatus


def test_run_manager_creates_record_and_context() -> None:
    temp_dir = _workspace_temp_dir()
    config = load_config(cli_overrides={"logs_root": str(temp_dir / "logs")},
    )
    manager = RunManager(config=config)

    context = manager.create_run(
        thread_id="thread-1",
        user_id="user-1",
        run_id="run-1",
    )

    record_path = temp_dir / "logs" / "run-1" / "record.json"
    assert context.run_id == "run-1"
    assert context.output_dir == temp_dir / "logs" / "run-1"
    assert record_path.exists()
    record = json.loads(record_path.read_text(encoding="utf-8"))
    assert record["status"] == RunStatus.PENDING.value
    assert record["thread_id"] == "thread-1"

    shutil.rmtree(temp_dir)


def test_run_manager_updates_lifecycle_and_journal() -> None:
    temp_dir = _workspace_temp_dir()
    config = load_config(cli_overrides={"logs_root": str(temp_dir / "logs")},
    )
    manager = RunManager(config=config)
    context = manager.create_run(
        thread_id="thread-1",
        user_id="user-1",
        run_id="run-1",
    )

    running = manager.mark_running("run-1")
    success = manager.mark_success("run-1", usage_summary={"total_tokens": 12})

    assert running.status == RunStatus.RUNNING
    assert success.status == RunStatus.SUCCESS
    assert success.total_tokens == 12
    events_content = context.events_path.read_text(encoding="utf-8")
    blocks = [b for b in events_content.split("\n\n") if b.strip()]
    assert [json.loads(b)["event_type"] for b in blocks] == [
        "run.started",
        "run.finished",
    ]

    shutil.rmtree(temp_dir)


def _workspace_temp_dir() -> Path:
    path = Path(".agent4ml/test-logs") / uuid4().hex
    path.mkdir(parents=True, exist_ok=False)
    return path
