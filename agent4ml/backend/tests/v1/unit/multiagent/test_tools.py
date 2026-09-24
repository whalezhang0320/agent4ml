"""tools.py 单测 — make_specialist_tool 动态生成 + schema 精简 + handler 编排 + error 格式。"""
from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from agent4ml.backend.agents.multiagent.exceptions import (
    SpecialistCrashError,
    SpecialistTimeoutError,
    SubagentError,
    SubagentMaxStepsError,
)
from agent4ml.backend.agents.multiagent.tools import (
    get_current_state,
    make_specialist_tool,
    make_subagent_tool,
    set_current_state,
)
from agent4ml.backend.agents.multiagent.types import (
    ArtifactRef,
    SpecialistRawResult,
    SubagentResult,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_specialist(raw_output="done", artifacts=None):
    s = MagicMock()
    s.name = "codex"
    s.invoke.return_value = SpecialistRawResult(
        raw_output=raw_output,
        artifacts=tuple(artifacts or []),
    )
    return s


def _mock_context_summarizer(summary="context"):
    cs = MagicMock()
    cs.summarize.return_value = summary
    return cs


def _mock_result_summarizer(success=True, summary="result", gap=""):
    rs = MagicMock()
    rs.summarize.return_value = MagicMock(
        success=success,
        summary=summary,
        specialist_name="codex",
        gap_analysis=gap,
        artifacts=(),
    )
    return rs


def _mock_subagent_provider(success=True, summary="sub result", gap=""):
    provider = MagicMock()
    provider.spawn.return_value = SubagentResult(
        summary=summary,
        success=success,
        gap_analysis=gap,
    )
    return provider


# ---------------------------------------------------------------------------
# make_specialist_tool — 动态生成
# ---------------------------------------------------------------------------


def test_make_specialist_tool_generates_named_tool():
    t = make_specialist_tool(
        "codex", _mock_specialist(), _mock_context_summarizer(), _mock_result_summarizer(),
    )
    assert t.name == "delegate_to_codex"


def test_make_specialist_tool_returns_basetool():
    from langchain_core.tools import BaseTool
    t = make_specialist_tool(
        "claude", _mock_specialist(), _mock_context_summarizer(), _mock_result_summarizer(),
    )
    assert isinstance(t, BaseTool)


# ---------------------------------------------------------------------------
# tool schema 精简（3 参数）
# ---------------------------------------------------------------------------


def test_tool_schema_has_3_params():
    t = make_specialist_tool(
        "codex", _mock_specialist(), _mock_context_summarizer(), _mock_result_summarizer(),
    )
    schema = t.args_schema.model_json_schema()
    props = schema.get("properties", {})
    assert set(props.keys()) == {"goal", "success_criteria", "sandbox_id"}


def test_tool_schema_goal_required():
    t = make_specialist_tool(
        "codex", _mock_specialist(), _mock_context_summarizer(), _mock_result_summarizer(),
    )
    schema = t.args_schema.model_json_schema()
    required = schema.get("required", [])
    assert "goal" in required
    assert "success_criteria" in required
    assert "sandbox_id" not in required


# ---------------------------------------------------------------------------
# tool handler 编排顺序
# ---------------------------------------------------------------------------


def test_handler_calls_context_summarizer_first():
    specialist = _mock_specialist()
    cs = _mock_context_summarizer()
    rs = _mock_result_summarizer()
    t = make_specialist_tool("codex", specialist, cs, rs)

    t.invoke({"goal": "g", "success_criteria": "sc"})

    cs.summarize.assert_called_once()


def test_handler_calls_specialist_invoke():
    specialist = _mock_specialist()
    t = make_specialist_tool(
        "codex", specialist, _mock_context_summarizer(), _mock_result_summarizer(),
    )

    t.invoke({"goal": "g", "success_criteria": "sc"})

    specialist.invoke.assert_called_once()


def test_handler_calls_result_summarizer_after_invoke():
    specialist = _mock_specialist()
    rs = _mock_result_summarizer()
    t = make_specialist_tool("codex", specialist, _mock_context_summarizer(), rs)

    t.invoke({"goal": "g", "success_criteria": "sc"})

    rs.summarize.assert_called_once()


def test_handler_returns_json_with_success():
    t = make_specialist_tool(
        "codex", _mock_specialist(), _mock_context_summarizer(), _mock_result_summarizer(success=True),
    )

    result = t.invoke({"goal": "g", "success_criteria": "sc"})
    data = json.loads(result)

    assert data["success"] is True
    assert "summary" in data
    assert data["specialist"] == "codex"


# ---------------------------------------------------------------------------
# specialist 失败返 error JSON
# ---------------------------------------------------------------------------


def test_handler_specialist_error_returns_error_json():
    specialist = MagicMock()
    specialist.name = "codex"
    specialist.invoke.side_effect = SpecialistTimeoutError(timeout_seconds=60)
    t = make_specialist_tool(
        "codex", specialist, _mock_context_summarizer(), _mock_result_summarizer(),
    )

    result = t.invoke({"goal": "g", "success_criteria": "sc"})
    data = json.loads(result)

    assert data["success"] is False
    assert data["error"]["type"] == "SpecialistTimeoutError"
    assert "suggestion" in data


def test_handler_crash_error_returns_error_json():
    specialist = MagicMock()
    specialist.name = "codex"
    specialist.invoke.side_effect = SpecialistCrashError(exit_code=1)
    t = make_specialist_tool(
        "codex", specialist, _mock_context_summarizer(), _mock_result_summarizer(),
    )

    result = t.invoke({"goal": "g", "success_criteria": "sc"})
    data = json.loads(result)

    assert data["success"] is False
    assert data["error"]["type"] == "SpecialistCrashError"


# ---------------------------------------------------------------------------
# state via ContextVar
# ---------------------------------------------------------------------------


def test_set_and_get_current_state():
    state = {"user_input": "test"}
    set_current_state(state)
    assert get_current_state() == state


def test_handler_reads_sandbox_id_from_state():
    specialist = _mock_specialist()
    cs = _mock_context_summarizer()
    t = make_specialist_tool("codex", specialist, cs, _mock_result_summarizer())

    set_current_state({"sandbox": {"sandbox_id": "from-state"}})
    t.invoke({"goal": "g", "success_criteria": "sc"})

    request = specialist.invoke.call_args[0][0]
    assert request.sandbox_id == "from-state"


def test_handler_explicit_sandbox_id_overrides_state():
    specialist = _mock_specialist()
    t = make_specialist_tool(
        "codex", specialist, _mock_context_summarizer(), _mock_result_summarizer(),
    )

    set_current_state({"sandbox": {"sandbox_id": "from-state"}})
    t.invoke({"goal": "g", "success_criteria": "sc", "sandbox_id": "explicit"})

    request = specialist.invoke.call_args[0][0]
    assert request.sandbox_id == "explicit"


# ---------------------------------------------------------------------------
# make_subagent_tool
# ---------------------------------------------------------------------------


def test_make_subagent_tool_generates_named_tool():
    t = make_subagent_tool(
        _mock_subagent_provider(), _mock_context_summarizer(), _mock_result_summarizer(),
    )
    assert t.name == "delegate_to_subagent"


def test_subagent_tool_invoke_success():
    t = make_subagent_tool(
        _mock_subagent_provider(success=True, summary="sub done"),
        _mock_context_summarizer(),
        _mock_result_summarizer(success=True, summary="evaluated"),
    )

    result = t.invoke({"goal": "g", "success_criteria": "sc"})
    data = json.loads(result)

    assert data["success"] is True
    assert data["specialist"] == "subagent"


def test_subagent_tool_error_returns_error_json():
    provider = MagicMock()
    provider.spawn.side_effect = SubagentMaxStepsError(max_steps=20)
    t = make_subagent_tool(
        provider, _mock_context_summarizer(), _mock_result_summarizer(),
    )

    result = t.invoke({"goal": "g", "success_criteria": "sc"})
    data = json.loads(result)

    assert data["success"] is False
    assert data["error"]["type"] == "SubagentMaxStepsError"
