"""Context summarizers 单测 — 3 个 summarizer：规则提取 + LLM 提取 mock + 不传全量 context。"""
from __future__ import annotations

from unittest.mock import MagicMock

from agent4ml.backend.agents.multiagent.summarizers.context.claude_code_context_summarizer import (
    ClaudeCodeContextSummarizer,
)
from agent4ml.backend.agents.multiagent.summarizers.context.codex_context_summarizer import (
    CodexContextSummarizer,
)
from agent4ml.backend.agents.multiagent.summarizers.context.self_copy_context_summarizer import (
    SelfCopyContextSummarizer,
)


def _msg(content: str) -> MagicMock:
    m = MagicMock()
    m.content = content
    return m


def _artifact(path: str) -> MagicMock:
    a = MagicMock()
    a.path = path
    return a


def _state(messages=None, artifacts=None, observations=None, user_input="") -> dict:
    return {
        "messages": messages or [],
        "artifacts": artifacts or [],
        "observations": observations or [],
        "user_input": user_input,
    }


# ---------------------------------------------------------------------------
# CodexContextSummarizer
# ---------------------------------------------------------------------------


def test_codex_extracts_code_blocks():
    cs = CodexContextSummarizer()
    state = _state(messages=[_msg("Here is code:\n```python\nprint('hello')\n```\nend")])
    result = cs.summarize(state, "write function", "passes tests")
    assert "```python" in result
    assert "print('hello')" in result


def test_codex_extracts_file_paths():
    cs = CodexContextSummarizer()
    state = _state(artifacts=[_artifact("/a.py"), _artifact("/b.py")])
    result = cs.summarize(state, "write function", "passes tests")
    assert "/a.py" in result
    assert "/b.py" in result


def test_codex_includes_goal_and_criteria():
    cs = CodexContextSummarizer()
    result = cs.summarize(_state(), "my goal", "my criteria")
    assert "my goal" in result
    assert "my criteria" in result


def test_codex_truncates_long_summary():
    cs = CodexContextSummarizer()
    long_code = "```python\n" + "x" * 5000 + "\n```"
    state = _state(messages=[_msg(long_code)])
    result = cs.summarize(state, "g", "sc")
    assert len(result) <= 3100
    assert "truncated" in result


def test_codex_no_code_returns_goal_only():
    cs = CodexContextSummarizer()
    state = _state(messages=[_msg("no code here")])
    result = cs.summarize(state, "goal", "criteria")
    assert "Relevant code" not in result


def test_codex_max_code_blocks():
    cs = CodexContextSummarizer()
    msgs = [_msg(f"```python\nblock {i}\n```") for i in range(5)]
    state = _state(messages=msgs)
    result = cs.summarize(state, "g", "sc")
    assert result.count("```python") == 3


# ---------------------------------------------------------------------------
# ClaudeCodeContextSummarizer
# ---------------------------------------------------------------------------


def test_claude_extracts_code_blocks():
    cs = ClaudeCodeContextSummarizer()
    state = _state(messages=[_msg("Review this:\n```python\nx = 1\n```")])
    result = cs.summarize(state, "review code", "review provided")
    assert "```python" in result
    assert "Code to review" in result


def test_claude_extracts_review_hints():
    cs = ClaudeCodeContextSummarizer()
    state = _state(messages=[_msg("Please check security and verify performance")])
    result = cs.summarize(state, "review", "review done")
    assert "Review focus" in result
    assert "security" in result
    assert "performance" in result


def test_claude_no_review_hints():
    cs = ClaudeCodeContextSummarizer()
    state = _state(messages=[_msg("just some text")])
    result = cs.summarize(state, "review", "review done")
    assert "Review focus" not in result


# ---------------------------------------------------------------------------
# SelfCopyContextSummarizer
# ---------------------------------------------------------------------------


def test_self_copy_rule_fallback_extracts_observations():
    cs = SelfCopyContextSummarizer(llm=None)
    obs = [MagicMock(content=f"finding {i}") for i in range(10)]
    state = _state(observations=obs, user_input="research X")
    result = cs.summarize(state, "summarize findings", "summary provided")
    assert "finding 5" in result
    assert "finding 9" in result
    assert "finding 4" not in result
    assert "research X" in result


def test_self_copy_llm_extraction():
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="LLM extracted context")
    cs = SelfCopyContextSummarizer(llm=mock_llm)
    result = cs.summarize(_state(user_input="research"), "goal", "criteria")
    assert result == "LLM extracted context"
    mock_llm.invoke.assert_called_once()


def test_self_copy_truncates_long_summary():
    cs = SelfCopyContextSummarizer(llm=None)
    obs = [MagicMock(content="x" * 1000) for _ in range(10)]
    state = _state(observations=obs)
    result = cs.summarize(state, "g", "sc")
    assert len(result) <= 3100
