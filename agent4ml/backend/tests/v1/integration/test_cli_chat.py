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


def test_chat_run_question_produces_report_and_logs(tmp_path) -> None:
    config = load_config(expert_mode=True, cli_overrides={"logs_root": str(tmp_path)})
    model = FakeChatModelWithTools(responses=["你好！我是 Agent4ML 研究助手。"])
    registry = CapabilityRegistry(
        models={"researcher": model, "reporter": model},
        tools={},
        reporter=MarkdownReporter(),
        artifact_store=LocalArtifactStore(),
    )
    thread_dir = tmp_path / "threads" / "thread-chat"
    thread_dir.mkdir(parents=True, exist_ok=True)
    runtime = AppRuntime(
        config=config,
        capability_registry=registry,
        run_manager=RunManager(config),
        researcher_model_name="fake-test-model",
        thread_id="thread-chat",
        thread_dir=thread_dir,
        thread_journal=RunJournal("thread-chat", thread_dir / "thread-events.jsonl"),
        leader_agent=make_lead_agent(expert_mode=True, capability_registry=registry),
    )

    result = runtime.run_question("你好")

    assert result.final_report
    assert result.run_id
    assert "events.jsonl" in result.events_path
