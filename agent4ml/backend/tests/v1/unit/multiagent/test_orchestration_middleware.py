"""OrchestrationMiddleware 单测 — 拦截 delegate_to_* + 打点 + 写 state + passthrough + error。"""
from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import ToolMessage
from langgraph.types import Command

from agent4ml.backend.agents.multiagent.middleware import OrchestrationMiddleware


def _make_request(tool_name="delegate_to_codex", state=None):
    req = MagicMock()
    req.tool_call = {"name": tool_name, "id": "call-1", "args": {}}
    req.state = state or {"messages": [], "sandbox": {"sandbox_id": "sb1"}}
    return req


def _success_result(specialist="codex"):
    return ToolMessage(
        content=json.dumps({
            "success": True,
            "summary": "done",
            "specialist": specialist,
            "artifacts": [{"path": "/a.py", "type": "code"}],
        }),
        tool_call_id="call-1",
    )


def _failure_result():
    return ToolMessage(
        content=json.dumps({
            "success": False,
            "error": {"type": "SpecialistTimeoutError", "message": "timeout"},
        }),
        tool_call_id="call-1",
    )


# ---------------------------------------------------------------------------
# 拦截 delegate_to_*
# ---------------------------------------------------------------------------


def test_intercepts_delegate_tool():
    mw = OrchestrationMiddleware()
    handler = MagicMock(return_value=_success_result())
    req = _make_request("delegate_to_codex")

    mw.wrap_tool_call(req, handler)

    handler.assert_called_once()


def test_passthrough_non_delegate_tool():
    mw = OrchestrationMiddleware()
    handler = MagicMock(return_value=_success_result())
    req = _make_request("web_search")

    mw.wrap_tool_call(req, handler)

    handler.assert_called_once()
    # Should return the raw result, not a Command
    result = mw.wrap_tool_call(req, handler)
    assert result == handler.return_value


def test_passthrough_bash_tool():
    mw = OrchestrationMiddleware()
    handler = MagicMock(return_value=_success_result())
    req = _make_request("bash")

    result = mw.wrap_tool_call(req, handler)

    assert result == handler.return_value


# ---------------------------------------------------------------------------
# 打点四计数器
# ---------------------------------------------------------------------------


def test_records_selection_and_invoked_before_handler():
    metrics = MagicMock()
    mw = OrchestrationMiddleware(metrics_store=metrics)
    handler = MagicMock(return_value=_success_result())

    mw.wrap_tool_call(_make_request("delegate_to_codex"), handler)

    metrics.record_selection.assert_called_once_with("codex")
    metrics.record_invoked.assert_called_once_with("codex")


def test_records_completion_on_success():
    metrics = MagicMock()
    mw = OrchestrationMiddleware(metrics_store=metrics)
    handler = MagicMock(return_value=_success_result())

    mw.wrap_tool_call(_make_request("delegate_to_codex"), handler)

    metrics.record_completion.assert_called_once_with("codex")
    metrics.record_fallback.assert_not_called()


def test_records_fallback_on_failure():
    metrics = MagicMock()
    mw = OrchestrationMiddleware(metrics_store=metrics)
    handler = MagicMock(return_value=_failure_result())

    mw.wrap_tool_call(_make_request("delegate_to_codex"), handler)

    metrics.record_fallback.assert_called_once_with("codex")
    metrics.record_completion.assert_not_called()


def test_records_fallback_on_exception():
    metrics = MagicMock()
    mw = OrchestrationMiddleware(metrics_store=metrics)
    handler = MagicMock(side_effect=RuntimeError("crash"))

    mw.wrap_tool_call(_make_request("delegate_to_codex"), handler)

    metrics.record_fallback.assert_called_once_with("codex")


def test_no_metrics_store_no_crash():
    mw = OrchestrationMiddleware(metrics_store=None)
    handler = MagicMock(return_value=_success_result())

    result = mw.wrap_tool_call(_make_request("delegate_to_codex"), handler)

    assert isinstance(result, Command)


# ---------------------------------------------------------------------------
# 写 ThreadState.orchestration
# ---------------------------------------------------------------------------


def test_writes_orchestration_state_on_success():
    mw = OrchestrationMiddleware()
    handler = MagicMock(return_value=_success_result())

    result = mw.wrap_tool_call(_make_request("delegate_to_codex"), handler)

    assert isinstance(result, Command)
    orch = result.update.get("orchestration")
    assert orch is not None
    assert "codex" in orch["active_specialists"]
    assert len(orch["specialist_artifacts"]) == 1
    assert orch["specialist_artifacts"][0].path == "/a.py"


def test_writes_orchestration_state_on_exception():
    mw = OrchestrationMiddleware()
    handler = MagicMock(side_effect=RuntimeError("crash"))

    result = mw.wrap_tool_call(_make_request("delegate_to_codex"), handler)

    assert isinstance(result, Command)
    orch = result.update.get("orchestration")
    assert "codex" in orch["active_specialists"]


def test_returns_command_with_messages():
    mw = OrchestrationMiddleware()
    handler = MagicMock(return_value=_success_result())

    result = mw.wrap_tool_call(_make_request("delegate_to_codex"), handler)

    assert isinstance(result, Command)
    assert "messages" in result.update
    assert "orchestration" in result.update


# ---------------------------------------------------------------------------
# specialist 失败转 error ToolMessage（pairing 完整性）
# ---------------------------------------------------------------------------


def test_exception_returns_error_tool_message():
    mw = OrchestrationMiddleware()
    handler = MagicMock(side_effect=RuntimeError("specialist crashed"))

    result = mw.wrap_tool_call(_make_request("delegate_to_codex"), handler)

    assert isinstance(result, Command)
    msg = result.update["messages"][0]
    assert isinstance(msg, ToolMessage)
    assert msg.status == "error"
    assert msg.tool_call_id == "call-1"
    data = json.loads(msg.content)
    assert data["success"] is False
    assert "error" in data


def test_exception_error_json_has_suggestion():
    mw = OrchestrationMiddleware()
    handler = MagicMock(side_effect=RuntimeError("crash"))

    result = mw.wrap_tool_call(_make_request("delegate_to_codex"), handler)

    msg = result.update["messages"][0]
    data = json.loads(msg.content)
    assert "suggestion" in data


# ---------------------------------------------------------------------------
# set_current_state before handler
# ---------------------------------------------------------------------------


def test_sets_current_state_before_handler():
    from agent4ml.backend.agents.multiagent.tools import get_current_state

    mw = OrchestrationMiddleware()
    captured_state = {}

    def capturing_handler(request):
        captured_state["state"] = get_current_state()
        return _success_result()

    state = {"messages": ["msg"], "sandbox": {"sandbox_id": "sb-test"}}
    mw.wrap_tool_call(_make_request("delegate_to_codex", state=state), capturing_handler)

    assert captured_state["state"] == state


# ---------------------------------------------------------------------------
# extract specialist name
# ---------------------------------------------------------------------------


def test_extract_specialist_name_codex():
    mw = OrchestrationMiddleware()
    assert mw._extract_specialist_name("delegate_to_codex") == "codex"


def test_extract_specialist_name_subagent():
    mw = OrchestrationMiddleware()
    assert mw._extract_specialist_name("delegate_to_subagent") == "subagent"


def test_extract_specialist_name_claude():
    mw = OrchestrationMiddleware()
    assert mw._extract_specialist_name("delegate_to_claude") == "claude"
