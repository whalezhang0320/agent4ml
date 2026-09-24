"""Result summarizers 单测 — base 通用校验 + codex 测试通过率 + claude review 完整性 + gap_analysis。"""
from __future__ import annotations

from agent4ml.backend.agents.multiagent.summarizers.result.base import (
    BaseResultSummarizer,
)
from agent4ml.backend.agents.multiagent.summarizers.result.claude_code_result_summarizer import (
    ClaudeCodeResultSummarizer,
)
from agent4ml.backend.agents.multiagent.summarizers.result.codex_result_summarizer import (
    CodexResultSummarizer,
)
from agent4ml.backend.agents.multiagent.summarizers.result.self_copy_result_summarizer import (
    SelfCopyResultSummarizer,
)
from agent4ml.backend.agents.multiagent.types import ArtifactRef


def _artifact(path="/a.py") -> ArtifactRef:
    return ArtifactRef(path=path, artifact_type="code", specialist_name="test")


# ---------------------------------------------------------------------------
# BaseResultSummarizer
# ---------------------------------------------------------------------------


def test_base_success_with_output_and_artifacts():
    rs = BaseResultSummarizer("base")
    result = rs.summarize("done", [_artifact()], "goal", "criteria")
    assert result.success is True
    assert result.summary == "done"
    assert result.specialist_name == "base"


def test_base_fail_empty_output():
    rs = BaseResultSummarizer("base")
    result = rs.summarize("", [_artifact()], "goal", "criteria")
    assert result.success is False
    assert result.gap_analysis != ""


def test_base_fail_no_artifacts():
    rs = BaseResultSummarizer("base")
    result = rs.summarize("output", [], "goal", "criteria")
    assert result.success is False


def test_base_fail_sensitive_changes():
    rs = BaseResultSummarizer("base")
    result = rs.summarize("rm -rf /", [_artifact()], "goal", "criteria")
    assert result.success is False


def test_base_compresses_long_output():
    rs = BaseResultSummarizer("base")
    long_output = "x" * 5000
    result = rs.summarize(long_output, [_artifact()], "g", "sc")
    assert len(result.summary) <= 2100
    assert "truncated" in result.summary


def test_base_gap_analysis_on_failure():
    rs = BaseResultSummarizer("base")
    result = rs.summarize("line1\nline2\nline3", [], "goal", "must pass")
    assert "Criteria not met" in result.gap_analysis
    assert "must pass" in result.gap_analysis


# ---------------------------------------------------------------------------
# CodexResultSummarizer
# ---------------------------------------------------------------------------


def test_codex_success_with_passing_tests():
    rs = CodexResultSummarizer()
    result = rs.summarize("5 passed, 0 failed", [_artifact()], "g", "sc")
    assert result.success is True
    assert result.specialist_name == "codex"


def test_codex_fail_zero_tests_passed():
    rs = CodexResultSummarizer()
    result = rs.summarize("0 passed, 3 failed", [_artifact()], "g", "sc")
    assert result.success is False
    assert "Tests failed" in result.gap_analysis


def test_codex_success_no_test_results():
    """No test results in output → base check decides (artifacts present → success)."""
    rs = CodexResultSummarizer()
    result = rs.summarize("code written", [_artifact()], "g", "sc")
    assert result.success is True


def test_codex_gap_includes_diff_check():
    rs = CodexResultSummarizer()
    result = rs.summarize("no diff here", [], "g", "sc")
    assert "No diff markers" in result.gap_analysis


# ---------------------------------------------------------------------------
# ClaudeCodeResultSummarizer
# ---------------------------------------------------------------------------


def test_claude_success_with_suggestions():
    rs = ClaudeCodeResultSummarizer()
    result = rs.summarize(
        "I suggest fixing the bug. Consider using a set.",
        [_artifact()],
        "g", "sc",
    )
    assert result.success is True
    assert result.specialist_name == "claude"


def test_claude_fail_no_suggestions():
    rs = ClaudeCodeResultSummarizer()
    result = rs.summarize("The code looks fine.", [_artifact()], "g", "sc")
    assert result.success is False
    assert "No modification suggestions" in result.gap_analysis


def test_claude_gap_includes_issue_check():
    rs = ClaudeCodeResultSummarizer()
    result = rs.summarize("I suggest changes.", [], "g", "sc")
    assert "No review issues" in result.gap_analysis


# ---------------------------------------------------------------------------
# SelfCopyResultSummarizer
# ---------------------------------------------------------------------------


def test_self_copy_success():
    rs = SelfCopyResultSummarizer()
    result = rs.summarize("research summary", [_artifact()], "g", "sc")
    assert result.success is True
    assert result.specialist_name == "subagent"


def test_self_copy_fail_empty():
    rs = SelfCopyResultSummarizer()
    result = rs.summarize("", [_artifact()], "g", "sc")
    assert result.success is False
