"""SubagentRuntime 单测 — leaf factory + isolated context + shared sandbox + max_steps。"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from agent4ml.backend.agents.multiagent.exceptions import (
    SpecialistCrashError,
    SpecialistStartupError,
)
from agent4ml.backend.agents.multiagent.runtimes.subagent_runtime import (
    SubagentRuntime,
)
from agent4ml.backend.agents.multiagent.types import SpecialistRequest


def _make_request(**kwargs) -> SpecialistRequest:
    defaults = dict(
        goal="research topic X",
        success_criteria="summary of findings",
        context_summary="previous research context",
        sandbox_id="sb789ghi",
        artifacts_path="/workspace",
        max_steps=10,
        timeout_seconds=300,
    )
    defaults.update(kwargs)
    return SpecialistRequest(**defaults)


def test_invoke_success():
    mock_agent = MagicMock()
    mock_agent.invoke.return_value = {
        "messages": [MagicMock(content="subagent result text")],
    }
    factory = MagicMock(return_value=mock_agent)

    rt = SubagentRuntime(agent_factory=factory)
    result = rt.invoke(_make_request())

    assert result.raw_output == "subagent result text"
    assert result.duration_seconds >= 0
    factory.assert_called_once()
    mock_agent.invoke.assert_called_once()


def test_invoke_no_factory_raises_startup():
    rt = SubagentRuntime(agent_factory=None)
    with pytest.raises(SpecialistStartupError, match="agent_factory not configured"):
        rt.invoke(_make_request())


def test_isolated_state_no_inherited_messages():
    """isolated context（INV#4）：全新 ThreadState，不继承父 messages。"""
    mock_agent = MagicMock()
    mock_agent.invoke.return_value = {"messages": [MagicMock(content="ok")]}

    rt = SubagentRuntime(agent_factory=lambda: mock_agent)
    request = _make_request(context_summary="ctx summary here")
    rt.invoke(request)

    state = mock_agent.invoke.call_args[0][0]
    assert state["messages"] == []
    assert state["metadata"]["context_summary"] == "ctx summary here"
    assert "user_input" in state
    assert state["user_input"] == "research topic X"


def test_shared_sandbox_id():
    """shared thread sandbox（INV#3）：复用父 sandbox_id。"""
    mock_agent = MagicMock()
    mock_agent.invoke.return_value = {"messages": [MagicMock(content="ok")]}

    rt = SubagentRuntime(agent_factory=lambda: mock_agent)
    rt.invoke(_make_request(sandbox_id="shared-sb-001"))

    state = mock_agent.invoke.call_args[0][0]
    assert state["sandbox"]["sandbox_id"] == "shared-sb-001"


def test_max_steps_sets_recursion_limit():
    """max_steps 限制：recursion_limit = max_steps * 2。"""
    mock_agent = MagicMock()
    mock_agent.invoke.return_value = {"messages": [MagicMock(content="ok")]}

    rt = SubagentRuntime(agent_factory=lambda: mock_agent)
    rt.invoke(_make_request(max_steps=15))

    config = mock_agent.invoke.call_args[1]["config"]
    assert config["recursion_limit"] == 30


def test_max_steps_exceeded_raises():
    from agent4ml.backend.agents.multiagent.exceptions import (
        SubagentMaxStepsError,
    )

    mock_agent = MagicMock()
    mock_agent.invoke.side_effect = RecursionError("recursion limit exceeded")

    rt = SubagentRuntime(agent_factory=lambda: mock_agent)
    with pytest.raises(SubagentMaxStepsError) as exc_info:
        rt.invoke(_make_request(max_steps=10))

    assert exc_info.value.details.get("max_steps") == 10


def test_generic_crash_raises_specialist_crash():
    mock_agent = MagicMock()
    mock_agent.invoke.side_effect = RuntimeError("agent crashed")

    rt = SubagentRuntime(agent_factory=lambda: mock_agent)
    with pytest.raises(SpecialistCrashError):
        rt.invoke(_make_request())


def test_extract_output_from_dict_messages():
    rt = SubagentRuntime(agent_factory=lambda: MagicMock())
    result = rt._extract_output({"messages": [MagicMock(content="hello")]})
    assert result == "hello"


def test_extract_output_from_empty_messages():
    rt = SubagentRuntime(agent_factory=lambda: MagicMock())
    result = rt._extract_output({"messages": []})
    assert result == ""


def test_extract_output_from_non_dict():
    rt = SubagentRuntime(agent_factory=lambda: MagicMock())
    assert rt._extract_output("plain string") == "plain string"
