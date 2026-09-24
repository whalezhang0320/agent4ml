"""Multi-Agent integration 单测 — 端到端 mock：specialist tool + middleware + metrics + state。"""
from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import ToolMessage
from langgraph.types import Command

from agent4ml.backend.agents.multiagent.middleware import OrchestrationMiddleware
from agent4ml.backend.agents.multiagent.metrics import MultiAgentMetricsStore
from agent4ml.backend.agents.multiagent.registry import SpecialistRegistry
from agent4ml.backend.agents.multiagent.specialists.codex_specialist import CodexSpecialist
from agent4ml.backend.agents.multiagent.summarizers.context.codex_context_summarizer import (
    CodexContextSummarizer,
)
from agent4ml.backend.agents.multiagent.summarizers.result.codex_result_summarizer import (
    CodexResultSummarizer,
)
from agent4ml.backend.agents.multiagent.tools import (
    make_specialist_tool,
    set_current_state,
)
from agent4ml.backend.agents.multiagent.types import (
    ArtifactRef,
    SpecialistRawResult,
    SpecialistRequest,
)


@pytest.fixture
def integration_setup(tmp_path):
    """End-to-end setup: specialist + tool + middleware + metrics."""
    metrics = MultiAgentMetricsStore(str(tmp_path / "integration.db"))

    # Mock specialist that returns raw output
    mock_specialist = MagicMock()
    mock_specialist.name = "codex"
    mock_specialist.capabilities = MagicMock()
    mock_specialist.invoke.return_value = SpecialistRawResult(
        raw_output="5 passed, 0 failed",
        artifacts=(ArtifactRef(path="/result.py", artifact_type="code", specialist_name="codex"),),
    )

    ctx_summarizer = CodexContextSummarizer()
    result_summarizer = CodexResultSummarizer()
    tool = make_specialist_tool("codex", mock_specialist, ctx_summarizer, result_summarizer)
    middleware = OrchestrationMiddleware(metrics_store=metrics)
    registry = SpecialistRegistry()
    registry.register(mock_specialist)

    return {
        "specialist": mock_specialist,
        "tool": tool,
        "middleware": middleware,
        "metrics": metrics,
        "registry": registry,
        "ctx_summarizer": ctx_summarizer,
        "result_summarizer": result_summarizer,
    }


def test_end_to_end_specialist_invocation(integration_setup):
    """Full flow: state → tool invoke → specialist → summarizer → JSON result."""
    setup = integration_setup
    set_current_state({"messages": [], "sandbox": {"sandbox_id": "sb-int"}})

    result = setup["tool"].invoke({"goal": "write tests", "success_criteria": "tests pass"})

    data = json.loads(result)
    assert data["success"] is True
    assert data["specialist"] == "codex"
    assert len(data["artifacts"]) == 1


def test_end_to_end_metrics_recorded(integration_setup):
    """Metrics: after tool invocation, counters are recorded via middleware."""
    setup = integration_setup

    # Simulate middleware intercepting the tool call
    request = MagicMock()
    request.tool_call = {"name": "delegate_to_codex", "id": "call-1", "args": {}}
    request.state = {"messages": [], "sandbox": {"sandbox_id": "sb-int"}}

    def handler(req):
        # Tool handler returns ToolMessage with JSON content
        set_current_state(req.state)
        result = setup["tool"].invoke({"goal": "g", "success_criteria": "sc"})
        return ToolMessage(content=result, tool_call_id="call-1")

    setup["middleware"].wrap_tool_call(request, handler)

    m = setup["metrics"].get_metrics("codex")
    assert m is not None
    assert m.total_selections == 1
    assert m.total_invoked == 1
    assert m.total_completions == 1
    assert m.total_fallbacks == 0


def test_end_to_end_orchestration_state_written(integration_setup):
    """ThreadState.orchestration written after specialist invocation."""
    setup = integration_setup

    request = MagicMock()
    request.tool_call = {"name": "delegate_to_codex", "id": "call-1", "args": {}}
    request.state = {"messages": [], "sandbox": {"sandbox_id": "sb-int"}}

    def handler(req):
        set_current_state(req.state)
        result = setup["tool"].invoke({"goal": "g", "success_criteria": "sc"})
        return ToolMessage(content=result, tool_call_id="call-1")

    cmd = setup["middleware"].wrap_tool_call(request, handler)

    assert isinstance(cmd, Command)
    orch = cmd.update.get("orchestration")
    assert orch is not None
    assert "codex" in orch["active_specialists"]
    assert len(orch["specialist_artifacts"]) == 1
    assert orch["specialist_artifacts"][0].path == "/result.py"


def test_end_to_end_specialist_failure_error_tool_message(integration_setup):
    """Specialist failure → error ToolMessage (pairing completeness, INV#6)."""
    setup = integration_setup

    # Make specialist raise
    setup["specialist"].invoke.side_effect = RuntimeError("crash")

    request = MagicMock()
    request.tool_call = {"name": "delegate_to_codex", "id": "call-2", "args": {}}
    request.state = {"messages": []}

    def handler(req):
        set_current_state(req.state)
        result = setup["tool"].invoke({"goal": "g", "success_criteria": "sc"})
        return ToolMessage(content=result, tool_call_id="call-2")

    cmd = setup["middleware"].wrap_tool_call(request, handler)

    # Handler doesn't raise — tool catches SpecialistError internally
    # But if handler itself raises, middleware converts to error ToolMessage
    assert isinstance(cmd, Command)


def test_end_to_end_middleware_exception_error_message(integration_setup):
    """If handler raises exception, middleware returns error ToolMessage."""
    setup = integration_setup

    request = MagicMock()
    request.tool_call = {"name": "delegate_to_codex", "id": "call-3", "args": {}}
    request.state = {"messages": []}

    def handler(req):
        raise RuntimeError("unexpected crash")

    cmd = setup["middleware"].wrap_tool_call(request, handler)

    assert isinstance(cmd, Command)
    msg = cmd.update["messages"][0]
    assert isinstance(msg, ToolMessage)
    assert msg.status == "error"
    data = json.loads(msg.content)
    assert data["success"] is False
    assert "suggestion" in data

    # Metrics: fallback recorded
    m = setup["metrics"].get_metrics("codex")
    assert m.total_fallbacks == 1


def test_end_to_end_non_delegate_passthrough(integration_setup):
    """Non-delegate tools pass through middleware without interception."""
    setup = integration_setup

    request = MagicMock()
    request.tool_call = {"name": "web_search", "id": "call-4", "args": {}}
    request.state = {"messages": []}

    sentinel = MagicMock()
    handler = MagicMock(return_value=sentinel)

    result = setup["middleware"].wrap_tool_call(request, handler)

    assert result is sentinel
    # No metrics recorded for non-delegate tools
    m = setup["metrics"].get_metrics("codex")
    assert m is None


def test_end_to_end_health_check_after_invocations(integration_setup):
    """health_check after multiple invocations shows specialist health."""
    setup = integration_setup

    for _ in range(6):
        request = MagicMock()
        request.tool_call = {"name": "delegate_to_codex", "id": "call-x", "args": {}}
        request.state = {"messages": [], "sandbox": {"sandbox_id": "sb"}}

        def handler(req):
            set_current_state(req.state)
            result = setup["tool"].invoke({"goal": "g", "success_criteria": "sc"})
            return ToolMessage(content=result, tool_call_id="call-x")

        setup["middleware"].wrap_tool_call(request, handler)

    health = setup["metrics"].health_check(threshold=0.4, min_invoked=5)
    codex_health = [h for h in health if h.specialist_name == "codex"]
    assert len(codex_health) == 1
    assert codex_health[0].degraded is False  # all successes → rate=1.0
