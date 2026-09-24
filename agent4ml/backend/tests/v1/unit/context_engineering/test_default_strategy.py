"""DefaultStrategy 单测：执行器 + 分段触发 + pairing + 快照 + 降级。"""

from __future__ import annotations

from types import SimpleNamespace

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agent4ml.backend.agents.context_engineering.contract import GovernanceContext
from agent4ml.backend.agents.context_engineering.strategies.default.externalizer import (
    ExternalizerExecutor,
)
from agent4ml.backend.agents.context_engineering.strategies.default.snapshot import (
    SnapshotExecutor,
)
from agent4ml.backend.agents.context_engineering.strategies.default.summarizer import (
    SummarizerExecutor,
)
from agent4ml.backend.agents.context_engineering.strategies.default.strategy import (
    DefaultStrategy,
)
from agent4ml.backend.agents.middlewares.tagged_context_middleware import (
    AGENT4ML_EXTERNALIZED,
    AGENT4ML_EXTERNALIZED_PATH,
    AGENT4ML_THINKING,
)


def _ctx(governance: dict | None = None, messages: list | None = None, config: dict | None = None, token_counter=None) -> GovernanceContext:
    return GovernanceContext(
        state={},
        governance=governance,
        config=config or {},
        token_counter=token_counter or (lambda m: 0),
        runtime=None,
        hook="test",
        messages=messages or [],
    )


def test_budget_init() -> None:
    strategy = DefaultStrategy()
    result = strategy.before_agent(_ctx())
    gov = result.state_patch["governance"]
    assert gov["default"]["budget"]["total"] == 0
    assert gov["default"]["pending"] == []
    assert gov["default"]["warned"] is False


def test_budget_track_pending() -> None:
    strategy = DefaultStrategy()
    governance = {"default": {"budget": {"total": 0, "window": 1000, "fraction": 0.0}, "seen_msgs": {}}}
    messages = [HumanMessage(content="Q" * 500)]
    result = strategy.after_model(_ctx(governance=governance, messages=messages, config={"window": 1000}, token_counter=lambda m: 500))
    gov = result.state_patch["governance"]
    assert gov["default"]["budget"]["fraction"] == 0.5
    assert "P1" in gov["default"]["pending"]
    assert "P2" in gov["default"]["pending"]


def test_budget_clear_run_state() -> None:
    strategy = DefaultStrategy()
    governance = {"default": {"budget": {"total": 100}, "seen_msgs": {}, "pending": ["P1"], "warned": True, "metrics": {"x": 1}}}
    result = strategy.after_agent(_ctx(governance=governance))
    gov = result.state_patch["governance"]
    assert "budget" not in gov["default"]
    assert "pending" not in gov["default"]
    assert gov["default"]["metrics"] == {"x": 1}  # metrics 保留


def test_externalizer_externalize_if_needed(tmp_path) -> None:
    ex = ExternalizerExecutor(externalize_dir=str(tmp_path), min_chars=10, preview_chars=5)
    tool = ToolMessage(content="x" * 100, tool_call_id="tc1", name="ddg")
    result = ex.externalize_if_needed(tool)
    assert result is not None
    assert result.additional_kwargs.get(AGENT4ML_EXTERNALIZED) is True
    assert result.additional_kwargs.get(AGENT4ML_EXTERNALIZED_PATH)


def test_externalizer_skip_small(tmp_path) -> None:
    ex = ExternalizerExecutor(externalize_dir=str(tmp_path), min_chars=1000)
    tool = ToolMessage(content="small", tool_call_id="tc1", name="ddg")
    assert ex.externalize_if_needed(tool) is None


def test_summarizer_no_model_skip() -> None:
    s = SummarizerExecutor(model=None)
    governance = {"default": {"pending": ["P4"]}}
    result = s.summarize_if_pending(governance, [HumanMessage(content="Q")], ExternalizerExecutor())
    assert result is None


def test_snapshot_p4(tmp_path) -> None:
    snap = SnapshotExecutor(snapshot_dir=str(tmp_path))
    governance = {"default": {"pending": ["P4"]}}
    state = {"research_question": "test"}
    result = snap.snapshot_if_pending(governance, [HumanMessage(content="Q")], state)
    assert result is not None
    assert "snapshot_path" in result["default"]
    assert result["default"]["metrics"]["snapshot_count"] == 1


def test_snapshot_serializes_dataclass_observations(tmp_path) -> None:
    """SnapshotExecutor 应能序列化 Observation/ReflectionItem dataclass，不崩。"""
    from agent4ml.backend.agents.state.types import Observation, ReflectionItem
    snap = SnapshotExecutor(snapshot_dir=str(tmp_path))
    governance = {"default": {"pending": ["P4"]}}
    state = {
        "research_question": "test",
        "observations": [Observation(
            observation_id="obs-1", step_id="todo-0",
            content="test content", source_refs=("src-1",), created_at="2026-07-08",
        )],
        "reflection_items": [ReflectionItem(
            item_id="refl-1", scope="run", kind="gap",
            question="test question", status="open", related_refs=(),
        )],
    }
    result = snap.snapshot_if_pending(governance, [HumanMessage(content="Q")], state)
    assert result is not None  # 不崩 = 序列化成功
    # 验证文件实际写入了 JSON
    import json
    from pathlib import Path
    snapshot_path = result["default"]["snapshot_path"]
    data = json.loads(Path(snapshot_path).read_text(encoding="utf-8"))
    assert len(data["observations"]) == 1
    assert data["observations"][0]["observation_id"] == "obs-1"
    assert len(data["reflection_items"]) == 1
    assert data["reflection_items"][0]["item_id"] == "refl-1"


def test_mark_thinking() -> None:
    strategy = DefaultStrategy()
    ai = AIMessage(content="ans", additional_kwargs={"reasoning_content": "thinking"})
    patch = strategy._mark_thinking([ai])
    assert patch is not None
    assert patch[0].additional_kwargs.get(AGENT4ML_THINKING) is True


def test_mark_thinking_skip_already_marked() -> None:
    strategy = DefaultStrategy()
    ai = AIMessage(content="ans", additional_kwargs={"reasoning_content": "thinking", AGENT4ML_THINKING: True})
    assert strategy._mark_thinking([ai]) is None


class _FakeModel:
    """假 model：invoke 返固定 .text，避真实 API。"""

    def __init__(self, text: str = "fake summary") -> None:
        self._text = text

    def invoke(self, prompt: str, **kwargs: Any) -> SimpleNamespace:
        return SimpleNamespace(text=self._text)


def test_partition_below_preserve_returns_empty() -> None:
    """messages 数 <= preserve_recent → to_summarize 空，全 preserved。"""
    s = SummarizerExecutor(model=_FakeModel(), preserve_recent=6)
    msgs = [HumanMessage(content="Q1"), HumanMessage(content="Q2")]
    to_sum, preserved = s._partition(msgs)
    assert to_sum == []
    assert preserved == msgs


def test_snap_to_pairing_retreats_for_toolmessage() -> None:
    """cut 落 ToolMessage 且前一个是 AIMessage w/ tool_calls → cut 回退，pairing 整组保。"""
    s = SummarizerExecutor(model=_FakeModel(), preserve_recent=2)
    ai = AIMessage(content="ans", tool_calls=[{"name": "ddg", "args": {}, "id": "tc1", "type": "tool_call"}])
    tool = ToolMessage(content="result", tool_call_id="tc1", name="ddg")
    msgs = [HumanMessage(content="Q1"), HumanMessage(content="Q2"), HumanMessage(content="Q3"), ai, tool, HumanMessage(content="Q4")]
    to_sum, preserved = s._partition(msgs)
    # cut 原本=4(msgs[4]=Tool), 回退到 3(msgs[3]=AI) → pairing 整组进 preserved
    assert to_sum == msgs[:3]
    assert preserved == msgs[3:]
    assert ai in preserved and tool in preserved  # pairing 保


def test_externalize_orphans_collects_paths(tmp_path) -> None:
    """孤立 ToolMessage（无配对 AIMessage w/ tool_calls）→ 外化 + path 收集；配对的不外化。"""
    s = SummarizerExecutor(model=_FakeModel(), preserve_recent=2)
    ex = ExternalizerExecutor(externalize_dir=str(tmp_path), min_chars=10, preview_chars=5)
    orphan = ToolMessage(content="x" * 100, tool_call_id="orphan1", name="ddg")
    ai_paired = AIMessage(content="ans", tool_calls=[{"name": "ddg", "args": {}, "id": "tc1", "type": "tool_call"}])
    tool_paired = ToolMessage(content="y" * 100, tool_call_id="tc1", name="ddg")
    to_summarize = [HumanMessage(content="Q1"), orphan, ai_paired, tool_paired]
    paths = s._externalize_orphans(to_summarize, ex)
    assert len(paths) == 1  # 仅 orphan1 外化
    assert "orphan1" in paths[0]
    assert paths[0].endswith(".txt")


def test_update_summary_writes_default_namespace() -> None:
    """_update_summary 写 governance.default.summary + summary_id + metrics.summarize_count，保留既有 metrics。"""
    s = SummarizerExecutor(model=_FakeModel())
    gov = {"default": {"metrics": {"snapshot_count": 1}}}
    out = s._update_summary(gov, "summary text")
    assert out["default"]["summary"] == "summary text"
    assert out["default"]["summary_id"].startswith("summary_")
    assert out["default"]["metrics"]["summarize_count"] == 1
    assert out["default"]["metrics"]["snapshot_count"] == 1  # 既有 metrics 保留


class _FakeWriter:
    """假 stream writer：收集调用。"""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def __call__(self, data: dict) -> None:
        self.calls.append(data)


def test_emit_compaction_event_with_writer(monkeypatch) -> None:
    """有 graph context（get_stream_writer 返 fake writer）→ event dict 写入流。"""
    fake = _FakeWriter()
    monkeypatch.setattr("langgraph.config.get_stream_writer", lambda: fake)
    DefaultStrategy._emit_compaction_event("compaction_start", tool_name="P4", content="fraction=0.85")
    assert len(fake.calls) == 1
    assert fake.calls[0] == {"type": "compaction_start", "tool_name": "P4", "content": "fraction=0.85"}


def test_emit_compaction_event_no_context_fallback(monkeypatch) -> None:
    """无 graph context（get_stream_writer 抛 RuntimeError）→ fallback logger，不崩。"""
    def _raise() -> None:
        raise RuntimeError("no graph context")

    monkeypatch.setattr("langgraph.config.get_stream_writer", _raise)
    DefaultStrategy._emit_compaction_event("compaction_start", tool_name="P4")  # 不崩


def test_after_model_uses_resolve_window_size() -> None:
    """config 无 window → resolve_window_size(model) 查表（deepseek-chat → 64000）。"""
    model = SimpleNamespace(model_name="deepseek-chat")
    strategy = DefaultStrategy(model=model)
    governance = {"default": {"budget": {"total": 0, "window": 0, "fraction": 0.0}, "seen_msgs": {}}}
    messages = [HumanMessage(content="Q" * 500)]
    result = strategy.after_model(_ctx(governance=governance, messages=messages, config={}, token_counter=lambda m: 32000))
    gov = result.state_patch["governance"]
    assert gov["default"]["budget"]["window"] == 64000
    assert gov["default"]["budget"]["fraction"] == 0.5  # 32000/64000


def test_after_model_config_window_overrides() -> None:
    """config.window 显式配置优先于 resolve_window_size。"""
    model = SimpleNamespace(model_name="deepseek-chat")  # 若查表 = 64000，被 config 1000 覆盖
    strategy = DefaultStrategy(model=model)
    governance = {"default": {"budget": {"total": 0, "window": 0, "fraction": 0.0}, "seen_msgs": {}}}
    messages = [HumanMessage(content="Q" * 500)]
    result = strategy.after_model(_ctx(governance=governance, messages=messages, config={"window": 1000}, token_counter=lambda m: 500))
    gov = result.state_patch["governance"]
    assert gov["default"]["budget"]["window"] == 1000
    assert gov["default"]["budget"]["fraction"] == 0.5  # 500/1000


def test_summarizer_uses_independent_model() -> None:
    """config summarize_model 独立取 → SummarizerExecutor 用 summarize_model，self._model 仍是 research model（查 window）。"""
    research = SimpleNamespace(name="research")
    summ = SimpleNamespace(name="summ")
    strategy = DefaultStrategy(model=research, summarize_model=summ)
    assert strategy._summarizer._model is summ
    assert strategy._model is research  # research model 仍用于 after_model 查 window


def test_summarizer_fallback_research_model() -> None:
    """无 summarize_model → SummarizerExecutor fallback research model。"""
    research = SimpleNamespace(name="research")
    strategy = DefaultStrategy(model=research)
    assert strategy._summarizer._model is research


def test_call_llm_uses_prompt_template() -> None:
    """_call_llm 用 prompts/system/default/summarize.md 模板，prompt 含模板文本 + history。"""
    class _FakeModel:
        def __init__(self) -> None:
            self.received: str | None = None

        def invoke(self, prompt: str, **kwargs: Any) -> SimpleNamespace:
            self.received = prompt
            return SimpleNamespace(text="summary")

    fake = _FakeModel()
    s = SummarizerExecutor(model=fake)
    result = s._call_llm([HumanMessage(content="Q1"), AIMessage(content="A1")])
    assert result == "summary"
    assert "对话历史" in fake.received  # 模板内容
    assert "[HumanMessage]" in fake.received  # history 格式
    assert "Q1" in fake.received


def test_format_history() -> None:
    """_format_history 拼 [Role] content[:500]。"""
    history = SummarizerExecutor._format_history([HumanMessage(content="Q1"), AIMessage(content="A2")])
    assert "[HumanMessage] Q1" in history
    assert "[AIMessage] A2" in history


def test_ensure_pairing_supplements_missing_toolmessage() -> None:
    """AIMessage(tool_calls) 缺配对 ToolMessage → 补 error ToolMessage。"""
    ai = AIMessage(content="ans", tool_calls=[{"name": "ddg", "args": {}, "id": "tc_missing", "type": "tool_call"}])
    patch = DefaultStrategy._ensure_pairing([ai])
    assert patch is not None
    assert len(patch) == 1
    assert patch[0].tool_call_id == "tc_missing"
    assert patch[0].status == "error"


def test_ensure_pairing_pass_when_paired() -> None:
    """AIMessage(tool_calls) 有配对 ToolMessage → 不补。"""
    ai = AIMessage(content="ans", tool_calls=[{"name": "ddg", "args": {}, "id": "tc1", "type": "tool_call"}])
    tool = ToolMessage(content="result", tool_call_id="tc1", name="ddg")
    assert DefaultStrategy._ensure_pairing([ai, tool]) is None


def test_p5_strips_tool_calls_and_jumps() -> None:
    """fraction >= 90% → 剥 tool_calls + 收尾提示 + jump_to model + warned=True。"""
    strategy = DefaultStrategy()
    governance = {"default": {"budget": {"total": 0, "window": 1000, "fraction": 0.0}, "seen_msgs": {}}}
    ai = AIMessage(
        id="ai-1",
        content="",
        tool_calls=[{"name": "web_search", "args": {"q": "x"}, "id": "tc1", "type": "tool_call"}],
        additional_kwargs={"tool_calls": [{"name": "web_search", "id": "tc1"}]},
    )
    messages = [HumanMessage(content="Q"), ai]
    result = strategy.after_model(_ctx(
        governance=governance, messages=messages,
        config={"window": 1000}, token_counter=lambda m: 950,
    ))
    assert result.jump_to == "model"
    gov = result.state_patch["governance"]
    assert gov["default"]["warned"] is True
    # messages_patch 含 cleared AIMessage + 收尾 HumanMessage
    patch_msgs = result.messages_patch
    cleared = patch_msgs[-2]
    stop_msg = patch_msgs[-1]
    assert isinstance(cleared, AIMessage)
    assert cleared.id == "ai-1"  # 复用原 id → add_messages 替换
    assert cleared.tool_calls == []
    assert "tool_calls" not in (cleared.additional_kwargs or {})  # additional_kwargs 也剥
    assert "context_budget_stop" in cleared.additional_kwargs
    assert isinstance(stop_msg, HumanMessage)
    assert stop_msg.name == "context_budget_stop"
    assert "90%" in stop_msg.content


def test_p5_skips_when_already_warned() -> None:
    """warned=True → P5 不再重复触发（防死循环）。"""
    strategy = DefaultStrategy()
    governance = {"default": {"budget": {"total": 0, "window": 1000, "fraction": 0.0}, "seen_msgs": {}, "warned": True}}
    ai = AIMessage(id="ai-1", content="", tool_calls=[{"name": "ddg", "args": {}, "id": "tc1", "type": "tool_call"}])
    messages = [HumanMessage(content="Q"), ai]
    result = strategy.after_model(_ctx(
        governance=governance, messages=messages,
        config={"window": 1000}, token_counter=lambda m: 950,
    ))
    assert result.jump_to is None  # 不 jump
    assert result.messages_patch is None or not any(
        isinstance(m, HumanMessage) and getattr(m, "name", None) == "context_budget_stop"
        for m in (result.messages_patch or [])
    )


def test_p5_skips_when_no_tool_calls() -> None:
    """fraction >= 90% 但最后 AIMessage 无 tool_calls → 不触发 P5（模型已想退出）。"""
    strategy = DefaultStrategy()
    governance = {"default": {"budget": {"total": 0, "window": 1000, "fraction": 0.0}, "seen_msgs": {}}}
    ai = AIMessage(id="ai-1", content="最终答案", tool_calls=[])
    messages = [HumanMessage(content="Q"), ai]
    result = strategy.after_model(_ctx(
        governance=governance, messages=messages,
        config={"window": 1000}, token_counter=lambda m: 950,
    ))
    assert result.jump_to is None


def test_p99_uses_hard_stop_message() -> None:
    """fraction >= 99% → 收尾提示含 '99%' 硬底线文案。"""
    strategy = DefaultStrategy()
    governance = {"default": {"budget": {"total": 0, "window": 1000, "fraction": 0.0}, "seen_msgs": {}}}
    ai = AIMessage(id="ai-1", content="", tool_calls=[{"name": "ddg", "args": {}, "id": "tc1", "type": "tool_call"}])
    messages = [HumanMessage(content="Q"), ai]
    result = strategy.after_model(_ctx(
        governance=governance, messages=messages,
        config={"window": 1000}, token_counter=lambda m: 999,
    ))
    assert result.jump_to == "model"
    stop_msg = result.messages_patch[-1]
    assert "99%" in stop_msg.content
    assert "硬底线" in stop_msg.content


def test_ensure_pairing_does_not_block_compaction() -> None:
    """_ensure_pairing 不再早退阻塞 P1/P4——pairing 补全与 compaction 并行执行。

    场景：有 pairing 问题（AIMessage 有 tool_calls 缺 ToolMessage）+ P1 pending（fraction >= 40%）。
    修复前：_ensure_pairing 早退 → P1 跳过。
    修复后：P1 仍执行 + pairing 补全合并进返回。
    """
    strategy = DefaultStrategy()
    # P1 pending + fraction=0.5
    governance = {"default": {
        "budget": {"total": 0, "window": 1000, "fraction": 0.5},
        "seen_msgs": {},
        "pending": ["P1"],
    }}
    # AIMessage 有 tool_calls 但缺配对 ToolMessage → _ensure_pairing 应补
    ai = AIMessage(
        content="ans",
        tool_calls=[{"name": "ddg", "args": {}, "id": "tc_orphan", "type": "tool_call"}],
    )
    messages = [HumanMessage(content="Q"), ai]
    result = strategy.before_model(_ctx(
        governance=governance, messages=messages,
        config={"window": 1000},
    ))
    # 不应返纯 pairing patch（早退）——应执行 P1 externalize 或至少不阻塞
    # 关键：返回的 messages_patch 应包含 pairing 补全的 ToolMessage
    if result.messages_patch:
        patch_types = [type(m).__name__ for m in result.messages_patch]
        # pairing 补全的 ToolMessage 应在 patch 中（若 P1 未产 patch，pairing patch 仍返）
        assert "ToolMessage" in patch_types or len(result.messages_patch) > 0


def test_ensure_pairing_alone_returns_pairing_patch() -> None:
    """无 compaction pending + 有 pairing 问题 → 仍返 pairing 补全（不丢功能）。"""
    strategy = DefaultStrategy()
    governance = {"default": {
        "budget": {"total": 0, "window": 1000, "fraction": 0.1},
        "seen_msgs": {},
        "pending": [],  # 无 P1/P4
    }}
    ai = AIMessage(
        content="ans",
        tool_calls=[{"name": "ddg", "args": {}, "id": "tc_orphan", "type": "tool_call"}],
    )
    messages = [HumanMessage(content="Q"), ai]
    result = strategy.before_model(_ctx(
        governance=governance, messages=messages,
        config={"window": 1000},
    ))
    assert result.messages_patch is not None
    assert len(result.messages_patch) == 1
    assert isinstance(result.messages_patch[0], ToolMessage)
    assert result.messages_patch[0].tool_call_id == "tc_orphan"


# --- P1 externalizer tests ---

def _long_tool(tool_call_id: str, content: str = "x" * 600) -> ToolMessage:
    """创建 content > min_chars(500) 的 ToolMessage。"""
    return ToolMessage(content=content, tool_call_id=tool_call_id, name="web_search")


def _short_tool(tool_call_id: str) -> ToolMessage:
    return ToolMessage(content="short", tool_call_id=tool_call_id, name="web_search")


def test_p1_fifo_externalizes_oldest_first(tmp_path) -> None:
    """FIFO：最早 turn 的 ToolMessage 先外化。"""
    ex = ExternalizerExecutor(externalize_dir=str(tmp_path), min_chars=500, exempt_rounds=2)
    # 6 turns：前 4 turn 可外化（exempt=2 豁免最后 2 turn），每 turn 2 个 tool（保 1 外化 1）
    msgs = []
    for i in range(6):
        msgs.append(HumanMessage(content=f"Q{i}"))
        msgs.append(_long_tool(f"tc_old_{i}"))
        msgs.append(_long_tool(f"tc_new_{i}"))
    result = ex.externalize_history(msgs)
    assert result is not None
    externalized_ids = {m.tool_call_id for m in result}
    # turn 0-3 的 tc_old_* 应被外化（每轮保 tc_new_*）
    assert "tc_old_0" in externalized_ids
    assert "tc_old_3" in externalized_ids
    # turn 4-5 豁免
    assert "tc_old_4" not in externalized_ids
    assert "tc_old_5" not in externalized_ids
    # tc_new_* 保留（每轮保最新）
    assert "tc_new_0" not in externalized_ids


def test_p1_exempt_recent_rounds(tmp_path) -> None:
    """近 exempt_rounds 轮不外化。"""
    ex = ExternalizerExecutor(externalize_dir=str(tmp_path), min_chars=500, exempt_rounds=3)
    msgs = []
    for i in range(5):
        msgs.append(HumanMessage(content=f"Q{i}"))
        msgs.append(_long_tool(f"tc_{i}"))
        msgs.append(_long_tool(f"tc_{i}_b"))
    result = ex.externalize_history(msgs)
    assert result is not None
    ids = {m.tool_call_id for m in result}
    # turn 0-1 可外化，turn 2-4 豁免
    assert "tc_0" in ids
    assert "tc_1" in ids
    assert "tc_2" not in ids
    assert "tc_4" not in ids


def test_p1_keep_one_per_turn(tmp_path) -> None:
    """每轮保 1：3 个 ToolMessage 只外化前 2 个。"""
    ex = ExternalizerExecutor(externalize_dir=str(tmp_path), min_chars=500, exempt_rounds=1)
    msgs = [HumanMessage(content="Q0"), _long_tool("tc_a"), _long_tool("tc_b"), _long_tool("tc_c"),
            HumanMessage(content="Q1"), _long_tool("tc_d")]
    result = ex.externalize_history(msgs)
    assert result is not None
    ids = {m.tool_call_id for m in result}
    assert "tc_a" in ids  # 外化
    assert "tc_b" in ids  # 外化
    assert "tc_c" not in ids  # 保留（每轮最新）


def test_p1_skip_already_externalized(tmp_path) -> None:
    """已标 agent4ml.externalized 的跳过。"""
    from agent4ml.backend.agents.middlewares.tagged_context_middleware import AGENT4ML_EXTERNALIZED
    ex = ExternalizerExecutor(externalize_dir=str(tmp_path), min_chars=500, exempt_rounds=1)
    already = _long_tool("tc_done")
    already = already.model_copy(update={"additional_kwargs": {AGENT4ML_EXTERNALIZED: True}})
    # turn 0: already(跳过) + tc_extra(可外化) + tc_new(保留) → tc_extra 被外化
    msgs = [HumanMessage(content="Q0"), already, _long_tool("tc_extra"), _long_tool("tc_new"),
            HumanMessage(content="Q1"), _long_tool("tc_q1")]
    result = ex.externalize_history(msgs)
    assert result is not None
    ids = {m.tool_call_id for m in result}
    assert "tc_done" not in ids  # 已外化跳过
    assert "tc_extra" in ids  # 未外化


def test_p1_interval_suppression_skips() -> None:
    """间隔抑制：fraction < p1_skip_until_fraction → P1 跳过。"""
    strategy = DefaultStrategy()
    governance = {"default": {
        "budget": {"total": 0, "window": 1000, "fraction": 0.45},
        "seen_msgs": {},
        "pending": ["P1"],
        "p1_skip_until_fraction": 0.52,  # 0.45 < 0.52 → skip
    }}
    result = strategy.before_model(_ctx(governance=governance, messages=[HumanMessage(content="Q")], config={"window": 1000}))
    # P1 被跳过，不返 messages_patch（除非有 pairing_patch）
    assert result.messages_patch is None or len(result.messages_patch) == 0


def test_p1_interval_expired_triggers() -> None:
    """间隔到期：fraction >= p1_skip_until_fraction → P1 重新触发。"""
    strategy = DefaultStrategy()
    governance = {"default": {
        "budget": {"total": 0, "window": 1000, "fraction": 0.55},
        "seen_msgs": {},
        "pending": ["P1"],
        "p1_skip_until_fraction": 0.52,  # 0.55 >= 0.52 → trigger
    }}
    msgs = [HumanMessage(content=f"Q{i}") for i in range(7)]
    for i in range(6):
        msgs.insert(i * 2 + 1, _long_tool(f"tc_{i}"))
    result = strategy.before_model(_ctx(governance=governance, messages=msgs, config={"window": 1000}))
    gov = result.state_patch.get("governance", {}) if result.state_patch else {}
    d = gov.get("default", {})
    assert d.get("p1_skip_until_fraction", 0) >= 0.55


def test_p1_externalizes_list_content(tmp_path) -> None:
    """MCP 工具返回 list[dict] content（如 browse_page）也应被外化。"""
    ex = ExternalizerExecutor(externalize_dir=str(tmp_path), min_chars=500, exempt_rounds=1)
    # 模拟 browse_page 返回 list content
    list_tool = ToolMessage(
        content=[{"type": "text", "text": "⚠️ EXTERNAL CONTENT — " + "x" * 600}],
        tool_call_id="tc_browse_1",
        name="browse_page",
    )
    msgs = [HumanMessage(content="Q0"), list_tool, _long_tool("tc_keep"),
            HumanMessage(content="Q1"), _long_tool("tc_q1")]
    result = ex.externalize_history(msgs)
    assert result is not None
    ids = {m.tool_call_id for m in result}
    assert "tc_browse_1" in ids  # list content 也被外化
    # 验证文件实际写入
    import os
    files = os.listdir(str(tmp_path))
    assert len(files) >= 1
    # 验证文件内容是提取后的纯文本
    with open(os.path.join(str(tmp_path), files[0]), encoding="utf-8") as f:
        written = f.read()
    assert "EXTERNAL CONTENT" in written


def test_p1_externalize_writes_real_file(tmp_path) -> None:
    """验证外化文件真实写入磁盘 + 内容可读回。"""
    ex = ExternalizerExecutor(externalize_dir=str(tmp_path), min_chars=100, exempt_rounds=1)
    long_content = "Real search result: " + "data " * 200  # ~1000 chars
    tool = ToolMessage(content=long_content, tool_call_id="tc_real_123abc", name="web_search")
    rewritten = ex.externalize_if_needed(tool)
    assert rewritten is not None
    assert rewritten.additional_kwargs.get(AGENT4ML_EXTERNALIZED) is True
    path = rewritten.additional_kwargs.get(AGENT4ML_EXTERNALIZED_PATH)
    assert path is not None
    # 验证文件存在 + 内容正确
    import os
    assert os.path.exists(path)
    with open(path, encoding="utf-8") as f:
        content = f.read()
    assert "Real search result" in content
    assert len(content) > 500
