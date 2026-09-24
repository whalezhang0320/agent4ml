"""标签化上下文基座单测：渲染正确性 + 标记幂等 + 时序 + trace + 不可压缩。"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from agent4ml.backend.agents.middlewares.tagged_context_middleware import (
    ContextAssembler,
    AGENT4ML_EXTERNALIZED,
    AGENT4ML_EXTERNALIZED_META,
    AGENT4ML_EXTERNALIZED_PATH,
    AGENT4ML_SUMMARY,
    AGENT4ML_THINKING,
    TaggedContextMiddleware,
)
from agent4ml.backend.agents.state.reducers import merge_tagged_context


def test_render_context_block_fields() -> None:
    """state 字段渲染 <goal><plan><date>。"""
    assembler = ContextAssembler()
    state = {"research_question": "研究X", "todos": [{"title": "步骤1", "status": "in_progress"}]}
    out = assembler.render_context_block(state, None)
    assert "<goal>研究X</goal>" in out
    assert "<plan>" in out
    assert "[>] 步骤1" in out
    assert "<date>" in out


def test_render_context_block_summary() -> None:
    """governance.default.summary 渲染 <summary>。"""
    assembler = ContextAssembler()
    out = assembler.render_context_block({}, {"default": {"summary": "上次摘要"}})
    assert "<summary>" in out
    assert "上次摘要" in out


def test_render_context_block_reflection() -> None:
    """reflection_items top-N 渲染 <reflection> 标签。"""
    assembler = ContextAssembler(max_reflections=2)
    state = {"reflection_items": [
        {"scope": "plan", "kind": "gap", "question": "Q1", "status": "open"},
        {"scope": "evidence", "kind": "sufficiency", "question": "Q2", "status": "resolved"},
        {"scope": "method", "kind": "bias", "question": "Q3", "status": "open"},
    ]}
    out = assembler.render_context_block(state, None)
    assert "<reflection>" in out
    assert "Q2" in out
    assert "Q3" in out
    assert "Q1" not in out  # top-2 最近


def test_render_messages_user_turn() -> None:
    """HumanMessage 开新 <turn> + <message role=user>。"""
    out = ContextAssembler().render_messages([HumanMessage(content="问题1")])
    assert '<turn id="1">' in out
    assert '<message role="user">问题1</message>' in out
    assert "</turn>" in out


def test_render_messages_thinking() -> None:
    """AIMessage agent4ml.thinking → <thinking>，content → <answer>。"""
    assembler = ContextAssembler()
    msg = AIMessage(content="回答", additional_kwargs={AGENT4ML_THINKING: True, "reasoning_content": "推理过程"})
    out = assembler.render_messages([HumanMessage(content="Q"), msg])
    assert "<thinking>推理过程</thinking>" in out
    assert "<answer>回答</answer>" in out


def test_render_messages_toolcall_toolresult_pairing() -> None:
    """toolcall + toolresult 同 turn 相邻（pairing 保）。"""
    assembler = ContextAssembler()
    ai = AIMessage(content="", tool_calls=[{"name": "ddg", "args": {"q": "x"}, "id": "tc1", "type": "tool_call"}])
    tool = ToolMessage(content="结果", tool_call_id="tc1", name="ddg")
    out = assembler.render_messages([HumanMessage(content="Q"), ai, tool])
    assert '<toolcall name="ddg"' in out
    assert '<toolresult name="ddg">结果</toolresult>' in out
    assert out.count("<turn ") == 1  # 同 turn 内


def test_render_messages_externalized() -> None:
    """ToolMessage agent4ml.externalized → <toolresult path=... tokens=...>。"""
    assembler = ContextAssembler()
    tool = ToolMessage(
        content="preview",
        tool_call_id="tc1",
        name="ddg",
        additional_kwargs={
            AGENT4ML_EXTERNALIZED: True,
            AGENT4ML_EXTERNALIZED_PATH: "/ext/tc1.txt",
            AGENT4ML_EXTERNALIZED_META: {"tokens_saved": 1200},
        },
    )
    out = assembler.render_messages([HumanMessage(content="Q"), AIMessage(content=""), tool])
    assert 'path="/ext/tc1.txt"' in out
    assert 'tokens="1200"' in out


def test_render_messages_summary_skip() -> None:
    """HumanMessage agent4ml.summary 跳过（context_block 已渲染，防重复）。"""
    assembler = ContextAssembler()
    summary_msg = HumanMessage(content="压缩摘要", additional_kwargs={AGENT4ML_SUMMARY: True})
    out = assembler.render_messages([summary_msg, HumanMessage(content="新问题")])
    assert "压缩摘要" not in out
    assert "新问题" in out


def test_render_messages_system_skip() -> None:
    """SystemMessage 跳过（wrap_model_call 从 request 提取为 <system>）。"""
    out = ContextAssembler().render_messages([SystemMessage(content="系统提示"), HumanMessage(content="Q")])
    assert "系统提示" not in out
    assert "Q" in out


def test_xml_escape() -> None:
    """content 含 < > & 转义，防破坏标签结构。"""
    out = ContextAssembler().render_messages([HumanMessage(content="a<b>&c")])
    assert "a&lt;b&gt;&amp;c" in out


def test_merge_tagged_context_last_write_wins() -> None:
    """merge_tagged_context：incoming None 保留 current，否则替换。"""
    assert merge_tagged_context(None, None) is None
    assert merge_tagged_context({"a": 1}, None) == {"a": 1}
    assert merge_tagged_context({"a": 1}, {"b": 2}) == {"b": 2}


def test_render_messages_for_llm_ai_thinking_answer() -> None:
    """AIMessage content 包 <thinking><answer>，角色保留。"""
    assembler = ContextAssembler()
    ai = AIMessage(content="回答", additional_kwargs={AGENT4ML_THINKING: True, "reasoning_content": "推理"})
    out = assembler.render_messages_for_llm([HumanMessage(content="Q"), ai])
    assert len(out) == 2
    assert out[0].content == "Q"
    assert "<thinking>推理</thinking>" in out[1].content
    assert "<answer>回答</answer>" in out[1].content


def test_render_messages_for_llm_toolmessage_classic() -> None:
    """ToolMessage 经典不变。"""
    assembler = ContextAssembler()
    tool = ToolMessage(content="结果", tool_call_id="tc1", name="ddg")
    out = assembler.render_messages_for_llm([tool])
    assert len(out) == 1
    assert out[0].content == "结果"


def test_render_messages_for_llm_skip() -> None:
    """SystemMessage + summary HumanMessage 跳过。"""
    assembler = ContextAssembler()
    summary_msg = HumanMessage(content="摘要", additional_kwargs={AGENT4ML_SUMMARY: True})
    out = assembler.render_messages_for_llm([SystemMessage(content="sys"), summary_msg, HumanMessage(content="Q")])
    assert len(out) == 1
    assert out[0].content == "Q"


def test_after_model_trace() -> None:
    """after_model 重组渲染写 state.tagged_context。"""
    mw = TaggedContextMiddleware()
    state = {"messages": [HumanMessage(content="Q")], "governance": None}
    result = mw.after_model(state, runtime=None)
    assert result is not None
    assert "tagged_context" in result
    assert "rendered" in result["tagged_context"]
    assert "Q" in result["tagged_context"]["rendered"]
    assert "created_at" in result["tagged_context"]
