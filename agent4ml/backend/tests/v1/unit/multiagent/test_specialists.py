"""Specialist 单测 — 3 个 specialist 组合 + invoke 流程 + 错误传播。"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from agent4ml.backend.agents.multiagent.exceptions import (
    SpecialistCrashError,
    SpecialistTimeoutError,
)
from agent4ml.backend.agents.multiagent.specialists.claude_code_specialist import (
    ClaudeCodeSpecialist,
)
from agent4ml.backend.agents.multiagent.specialists.codex_specialist import (
    CodexSpecialist,
)
from agent4ml.backend.agents.multiagent.specialists.subagent_specialist import (
    SubagentSpecialist,
)
from agent4ml.backend.agents.multiagent.types import (
    SpecialistCapability,
    SpecialistRawResult,
    SpecialistRequest,
)


def _make_request() -> SpecialistRequest:
    return SpecialistRequest(
        goal="g", success_criteria="sc", context_summary="cs",
        sandbox_id="sb1", artifacts_path="/p",
    )


# ---------------------------------------------------------------------------
# CodexSpecialist
# ---------------------------------------------------------------------------


def test_codex_specialist_name():
    s = CodexSpecialist()
    assert s.name == "codex"


def test_codex_specialist_capabilities():
    s = CodexSpecialist()
    assert s.capabilities.has(SpecialistCapability.CODING)
    assert not s.capabilities.has(SpecialistCapability.RESEARCH)


def test_codex_specialist_invoke_delegates_to_runtime():
    mock_rt = MagicMock()
    mock_rt.invoke.return_value = SpecialistRawResult(raw_output="done")
    s = CodexSpecialist(runtime=mock_rt)

    result = s.invoke(_make_request())

    mock_rt.invoke.assert_called_once()
    assert result.raw_output == "done"


def test_codex_specialist_invoke_propagates_error():
    mock_rt = MagicMock()
    mock_rt.invoke.side_effect = SpecialistTimeoutError(timeout_seconds=60)
    s = CodexSpecialist(runtime=mock_rt)

    with pytest.raises(SpecialistTimeoutError):
        s.invoke(_make_request())


def test_codex_specialist_default_runtime():
    s = CodexSpecialist()
    assert s._runtime is not None


# ---------------------------------------------------------------------------
# ClaudeCodeSpecialist
# ---------------------------------------------------------------------------


def test_claude_specialist_name():
    s = ClaudeCodeSpecialist()
    assert s.name == "claude"


def test_claude_specialist_capabilities():
    s = ClaudeCodeSpecialist()
    assert s.capabilities.has(SpecialistCapability.REVIEW)
    assert not s.capabilities.has(SpecialistCapability.CODING)


def test_claude_specialist_invoke_delegates_to_runtime():
    mock_rt = MagicMock()
    mock_rt.invoke.return_value = SpecialistRawResult(raw_output="review done")
    s = ClaudeCodeSpecialist(runtime=mock_rt)

    result = s.invoke(_make_request())

    assert result.raw_output == "review done"


def test_claude_specialist_invoke_propagates_error():
    mock_rt = MagicMock()
    mock_rt.invoke.side_effect = SpecialistCrashError(exit_code=1)
    s = ClaudeCodeSpecialist(runtime=mock_rt)

    with pytest.raises(SpecialistCrashError):
        s.invoke(_make_request())


# ---------------------------------------------------------------------------
# SubagentSpecialist
# ---------------------------------------------------------------------------


def test_subagent_specialist_name():
    s = SubagentSpecialist()
    assert s.name == "subagent"


def test_subagent_specialist_capabilities():
    s = SubagentSpecialist()
    assert s.capabilities.has(SpecialistCapability.RESEARCH)


def test_subagent_specialist_invoke_delegates_to_runtime():
    mock_rt = MagicMock()
    mock_rt.invoke.return_value = SpecialistRawResult(raw_output="research result")
    s = SubagentSpecialist(runtime=mock_rt)

    result = s.invoke(_make_request())

    assert result.raw_output == "research result"


def test_subagent_specialist_invoke_propagates_error():
    mock_rt = MagicMock()
    mock_rt.invoke.side_effect = SpecialistCrashError("agent crashed")
    s = SubagentSpecialist(runtime=mock_rt)

    with pytest.raises(SpecialistCrashError):
        s.invoke(_make_request())
