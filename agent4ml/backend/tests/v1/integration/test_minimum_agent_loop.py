import json
import shutil
from pathlib import Path
from uuid import uuid4

from langchain_core.language_models.fake_chat_models import FakeListChatModel

from agent4ml.backend.agents.artifacts.local_store import LocalArtifactStore
from agent4ml.backend.agents.capabilities.registry import CapabilityRegistry
from agent4ml.backend.agents.config.loader import load_config
from agent4ml.backend.agents.journal.run_journal import RunJournal
from agent4ml.backend.agents.leader.factory import make_lead_agent
from agent4ml.backend.agents.reporting.markdown_reporter import MarkdownReporter
from agent4ml.backend.agents.runtime.run_manager import RunManager
from agent4ml.backend.app.bootstrap import AppRuntime
from agent4ml.backend.tests.v1._fake_model import FakeChatModelWithTools


def _fake_model():
    return FakeChatModelWithTools(responses=["Research Agent4ML architecture summary."])


def test_minimum_agent_loop_generates_final_report_and_artifact() -> None:
    temp_dir = _workspace_temp_dir()
    config = load_config(expert_mode=True, cli_overrides={"logs_root": str(temp_dir)})
    model = _fake_model()
    registry = CapabilityRegistry(
        models={"researcher": model, "reporter": model},
        tools={},
        reporter=MarkdownReporter(),
        artifact_store=LocalArtifactStore(),
    )
    thread_dir = temp_dir / "threads" / "thread-1"
    thread_dir.mkdir(parents=True, exist_ok=True)
    runtime = AppRuntime(
        config=config,
        capability_registry=registry,
        run_manager=RunManager(config),
        researcher_model_name="fake-test",
        thread_id="thread-1",
        thread_dir=thread_dir,
        thread_journal=RunJournal("thread-1", thread_dir / "thread-events.jsonl"),
        leader_agent=make_lead_agent(expert_mode=True, capability_registry=registry),
    )

    result = runtime.run_question(
        question="Research Agent4ML architecture",
        run_id="run-1",
    )

    assert result.final_report
    run_dir = thread_dir / "runs" / "run-1"
    assert (run_dir / "record.json").exists()
    assert (run_dir / "events.jsonl").exists()
    assert (run_dir / "artifacts" / "final_report.md").exists()

    events_content = (run_dir / "events.jsonl").read_text(encoding="utf-8")
    events = [
        json.loads(b)["event_type"]
        for b in events_content.split("\n\n")
        if b.strip()
    ]
    assert "run.started" in events
    assert "report.generated" in events
    assert "run.finished" in events

    shutil.rmtree(temp_dir)


def _workspace_temp_dir() -> Path:
    path = Path(".agent4ml/test-logs") / uuid4().hex
    path.mkdir(parents=True, exist_ok=False)
    return path
