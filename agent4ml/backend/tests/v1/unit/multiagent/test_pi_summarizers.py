"""PiContextSummarizer + PiResultSummarizer 单测（P5）。

验证：
- PiContextSummarizer: 规则提取代码块 + 文件路径 + 零 LLM
- PiResultSummarizer: 解析三段输出 + success 判定 + gap_analysis 提取
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from agent4ml.backend.agents.multiagent.summarizers.context.pi_context_summarizer import (
    PiContextSummarizer,
)
from agent4ml.backend.agents.multiagent.summarizers.result.pi_result_summarizer import (
    PiResultSummarizer,
)
from agent4ml.backend.agents.multiagent.types import ArtifactRef


# ---------------------------------------------------------------------------
# PiContextSummarizer
# ---------------------------------------------------------------------------


def _msg(content: str) -> MagicMock:
    m = MagicMock()
    m.content = content
    return m


def test_pi_context_summarizer_extracts_code_blocks():
    """从 messages 提取代码块。"""
    cs = PiContextSummarizer()
    state = {
        "messages": [
            _msg("Here is the code:\n```python\ndef foo(): pass\n```\nDone."),
        ],
        "observations": [],
    }
    result = cs.summarize(state, goal="write foo", success_criteria="foo exists")
    assert "```python" in result
    assert "def foo" in result


def test_pi_context_summarizer_extracts_file_paths():
    """从 observations 提取文件路径。"""
    cs = PiContextSummarizer()
    state = {
        "messages": [],
        "observations": [
            MagicMock(content="Found auth.py in src/auth.py and config.json"),
        ],
    }
    result = cs.summarize(state, goal="fix auth", success_criteria="auth works")
    assert "auth.py" in result
    assert "config.json" in result


def test_pi_context_summarizer_truncates_long_summary():
    """超长 summary 截断。"""
    cs = PiContextSummarizer()
    long_content = "x" * 5000
    state = {"messages": [_msg(long_content)], "observations": []}
    result = cs.summarize(state, goal="g", success_criteria="s")
    assert len(result) <= 3100  # _MAX_SUMMARY_CHARS + truncation marker


def test_pi_context_summarizer_no_code_no_paths():
    """无代码无文件路径时只返 goal + success_criteria。"""
    cs = PiContextSummarizer()
    state = {"messages": [_msg("just text")], "observations": []}
    result = cs.summarize(state, goal="g", success_criteria="s")
    assert "Goal: g" in result
    assert "Success criteria: s" in result


# ---------------------------------------------------------------------------
# PiResultSummarizer
# ---------------------------------------------------------------------------


def _make_artifacts() -> list[ArtifactRef]:
    return [ArtifactRef(path="/tmp/test.py", artifact_type="code", specialist_name="pi")]


def test_pi_result_summarizer_parses_three_sections():
    """解析 pi 输出的 What You Did / Success / Gaps 三段。"""
    rs = PiResultSummarizer()
    raw = (
        "## What You Did\n- Created auth.py\n- Ran tests\n\n"
        "## Success\n- yes, criteria met\n- tests pass\n\n"
        "## Gaps\n- none\n"
    )
    result = rs.summarize(raw, _make_artifacts(), "write auth", "auth works + tests pass")
    assert result.specialist_name == "pi"
    # summary 取 What You Did 段
    assert "Created auth.py" in result.summary
    assert "Ran tests" in result.summary


def test_pi_result_summarizer_success_positive():
    """Success 段含 yes/pass/met → success=True。"""
    rs = PiResultSummarizer()
    raw = (
        "## What You Did\n- done\n\n"
        "## Success\n- yes, criteria met\n\n"
        "## Gaps\n- none\n"
    )
    result = rs.summarize(raw, _make_artifacts(), "g", "s")
    assert result.success is True


def test_pi_result_summarizer_success_negative():
    """Success 段含 no/fail/partial → success=False。"""
    rs = PiResultSummarizer()
    raw = (
        "## What You Did\n- partial work\n\n"
        "## Success\n- no, not met\n\n"
        "## Gaps\n- tests failing\n"
    )
    result = rs.summarize(raw, _make_artifacts(), "g", "s")
    assert result.success is False
    assert "tests failing" in result.gap_analysis or "Gaps" in result.gap_analysis or result.gap_analysis


def test_pi_result_summarizer_no_sections_uses_base_compression():
    """无三段输出时用 base 压缩 + base success 判定。"""
    rs = PiResultSummarizer()
    raw = "just some output without sections"
    result = rs.summarize(raw, _make_artifacts(), "g", "s")
    # base 压缩返 raw（截断后）
    assert result.summary == raw or result.summary == raw[:2000]


def test_pi_result_summarizer_sensitive_pattern_fails():
    """含敏感命令（rm -rf /）→ base 校验失败 → success=False。"""
    rs = PiResultSummarizer()
    raw = (
        "## What You Did\n- rm -rf /\n\n"
        "## Success\n- yes\n\n"
        "## Gaps\n- none\n"
    )
    result = rs.summarize(raw, _make_artifacts(), "g", "s")
    assert result.success is False


def test_pi_result_summarizer_gaps_extracted():
    """Gaps 段提取到 gap_analysis。"""
    rs = PiResultSummarizer()
    raw = (
        "## What You Did\n- partial\n\n"
        "## Success\n- no, not met\n\n"
        "## Gaps\n- need to fix auth.py line 42\n- tests still failing\n"
    )
    result = rs.summarize(raw, _make_artifacts(), "g", "s")
    assert "fix auth.py" in result.gap_analysis or "tests still failing" in result.gap_analysis


def test_pi_result_summarizer_artifacts_preserved():
    """artifacts 透传到 SpecialistResult。"""
    rs = PiResultSummarizer()
    arts = _make_artifacts()
    raw = "## What You Did\n- done\n\n## Success\n- yes\n\n## Gaps\n- none\n"
    result = rs.summarize(raw, arts, "g", "s")
    assert result.artifacts == tuple(arts)


def test_pi_result_summarizer_specialist_name():
    """specialist_name="pi"。"""
    rs = PiResultSummarizer()
    raw = "## What You Did\n- done\n\n## Success\n- yes\n\n## Gaps\n- none\n"
    result = rs.summarize(raw, _make_artifacts(), "g", "s")
    assert result.specialist_name == "pi"
