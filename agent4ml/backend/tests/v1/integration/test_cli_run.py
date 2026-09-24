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


def test_cli_run_prints_report_and_paths() -> None:
    temp_dir = _workspace_temp_dir()
    config = load_config(expert_mode=True, cli_overrides={"logs_root": str(temp_dir)})
    model = FakeChatModelWithTools(responses=["CLI research question answer."])
    registry = CapabilityRegistry(
        models={"researcher": model, "reporter": model},
        tools={},
        reporter=MarkdownReporter(),
        artifact_store=LocalArtifactStore(),
    )
    thread_dir = temp_dir / "threads" / "thread-cli"
    thread_dir.mkdir(parents=True, exist_ok=True)
    runtime = AppRuntime(
        config=config,
        capability_registry=registry,
        run_manager=RunManager(config),
        researcher_model_name="fake-test-model",
        thread_id="thread-cli",
        thread_dir=thread_dir,
        thread_journal=RunJournal("thread-cli", thread_dir / "thread-events.jsonl"),
        leader_agent=make_lead_agent(expert_mode=True, capability_registry=registry),
    )

    result = runtime.run_question(question="CLI research question", run_id="run-cli")

    assert "CLI research question" in result.final_report
    run_dir = thread_dir / "runs" / "run-cli"
    assert (run_dir / "events.jsonl").exists()
    assert (run_dir / "artifacts" / "final_report.md").exists()

    shutil.rmtree(temp_dir)


def _workspace_temp_dir() -> Path:
    path = Path(".agent4ml/test-logs") / uuid4().hex
    path.mkdir(parents=True, exist_ok=False)
    return path
