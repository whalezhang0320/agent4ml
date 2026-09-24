"""集成测试：default 模式行为 + 模式切换 + 意图识别 + /report 触发。

覆盖 doc 19 核心场景：
- default 模式不自动生成报告（输出 last AIMessage，无 artifact）
- switch_expert_mode 保留 thread_id
- intent "生成报告" 触发 generate_report_from_thread
- intent "如何写报告" 不误触发
- /report 命令经 pending flag 触发
"""

from pathlib import Path

from agent4ml.backend.agents.artifacts.local_store import LocalArtifactStore
from agent4ml.backend.agents.capabilities.registry import CapabilityRegistry
from agent4ml.backend.agents.config.loader import load_config
from agent4ml.backend.agents.intent import default_intent_tree
from agent4ml.backend.agents.journal.run_journal import RunJournal
from agent4ml.backend.agents.leader.factory import make_lead_agent
from agent4ml.backend.agents.reporting import generate_report_from_thread
from agent4ml.backend.agents.reporting.markdown_reporter import MarkdownReporter
from agent4ml.backend.agents.runtime.run_manager import RunManager
from agent4ml.backend.app.bootstrap import AppRuntime
from agent4ml.backend.tests.v1._fake_model import FakeChatModelWithTools


def _make_runtime(tmp_path: Path, expert_mode: bool = False) -> AppRuntime:
    config = load_config(expert_mode=expert_mode, cli_overrides={"logs_root": str(tmp_path)})
    model = FakeChatModelWithTools(responses=["对话回答"])
    registry = CapabilityRegistry(
        models={"researcher": model, "reporter": model},
        tools={},
        reporter=MarkdownReporter(),
        artifact_store=LocalArtifactStore(),
    )
    thread_dir = tmp_path / "threads" / "thread-int"
    thread_dir.mkdir(parents=True, exist_ok=True)
    return AppRuntime(
        config=config,
        capability_registry=registry,
        run_manager=RunManager(config),
        researcher_model_name="fake-test",
        thread_id="thread-int",
        thread_dir=thread_dir,
        thread_journal=RunJournal("thread-int", thread_dir / "thread-events.jsonl"),
        leader_agent=make_lead_agent(expert_mode=expert_mode, capability_registry=registry),
    )


def test_default_mode_no_auto_report_no_artifact(tmp_path) -> None:
    """default 模式：run_question 输出 last AIMessage，不自动报告，不保存 artifact。"""
    runtime = _make_runtime(tmp_path, expert_mode=False)
    result = runtime.run_question(question="你好", run_id="run-default")
    assert result.final_report  # last AIMessage
    assert result.artifact_path is None  # default 不保存 artifact


def test_expert_mode_auto_report_with_artifact(tmp_path) -> None:
    """expert 模式：自动报告 + 保存 artifact。"""
    runtime = _make_runtime(tmp_path, expert_mode=True)
    result = runtime.run_question(question="研究 X", run_id="run-expert")
    assert result.final_report
    assert result.artifact_path is not None  # expert 保存 artifact


def test_switch_expert_mode_preserves_thread_id(tmp_path) -> None:
    """switch_expert_mode 保留 thread_id（checkpointer state 连续）。"""
    runtime = _make_runtime(tmp_path, expert_mode=False)
    original_thread_id = runtime.thread_id
    new_runtime = runtime.switch_expert_mode(expert_mode=True)
    assert new_runtime.thread_id == original_thread_id
    assert new_runtime.config.runtime.expert_mode is True
    assert new_runtime.leader_agent is not runtime.leader_agent  # 重建


def test_intent_generate_report_triggers_handler(tmp_path) -> None:
    """意图 '生成报告' 命中 ReportIntent → handler 调用。"""
    runtime = _make_runtime(tmp_path, expert_mode=False)
    called: list = []

    def handler(intent, rt):
        called.append(intent)
        return True

    tree = default_intent_tree(report_handler=handler)
    assert tree.detect_and_dispatch("生成报告", runtime) is True
    assert len(called) == 1


def test_intent_how_to_write_report_not_triggered(tmp_path) -> None:
    """'如何写报告' 不误触发（整句匹配）。"""
    runtime = _make_runtime(tmp_path, expert_mode=False)
    tree = default_intent_tree()
    assert tree.detect_and_dispatch("如何写报告", runtime) is False


def test_intent_slash_report_with_topic(tmp_path) -> None:
    """'/report 天气' 命中 + topic 提取。"""
    runtime = _make_runtime(tmp_path, expert_mode=False)
    captured: list = []

    def handler(intent, rt):
        captured.append(intent.payload.get("topic"))
        return True

    tree = default_intent_tree(report_handler=handler)
    assert tree.detect_and_dispatch("/report 天气调研", runtime) is True
    assert captured == ["天气调研"]


def test_generate_report_from_thread_default_mode(tmp_path) -> None:
    """default 模式 generate_report_from_thread：从 state 合成报告 + 保存 artifact。"""
    runtime = _make_runtime(tmp_path, expert_mode=False)
    # 模拟 graph.get_state 返回有 observations 的 state
    from types import SimpleNamespace
    fake_state = {"observations": [{"content": "obs1"}], "sources": [], "research_question": "测试"}
    runtime.leader_agent.graph.get_state = lambda config: SimpleNamespace(values=fake_state)
    result = generate_report_from_thread(runtime=runtime, topic="自定义主题")
    assert result.final_report
    assert result.artifact_path is not None


def test_generate_report_from_thread_expert_mode_via_handler(tmp_path) -> None:
    """expert 模式：handler 应提示已自动生成（presentation 层逻辑，此处测服务仍可调）。"""
    runtime = _make_runtime(tmp_path, expert_mode=True)
    from types import SimpleNamespace
    fake_state = {"final_report": "expert 已合成报告", "observations": []}
    runtime.leader_agent.graph.get_state = lambda config: SimpleNamespace(values=fake_state)
    result = generate_report_from_thread(runtime=runtime)
    assert "expert 已合成报告" in result.final_report


def test_commands_report_sets_pending_flag() -> None:
    """/report 命令设 pending_report flag。"""
    from io import StringIO
    from rich.console import Console
    from agent4ml.backend.app.cli.commands import handle_command

    state = {"pending_expert_mode": None, "pending_report": None}
    con = Console(file=StringIO(), width=120)
    handle_command("/report 天气", con, None, state, runtime=None)
    assert state["pending_report"] == "天气"


def test_commands_expert_default_set_pending() -> None:
    from io import StringIO
    from rich.console import Console
    from agent4ml.backend.app.cli.commands import handle_command

    state = {"pending_expert_mode": None, "pending_report": None}
    con = Console(file=StringIO(), width=120)
    handle_command("/expert", con, None, state, runtime=None)
    assert state["pending_expert_mode"] is True
    handle_command("/default", con, None, state, runtime=None)
    assert state["pending_expert_mode"] is False
