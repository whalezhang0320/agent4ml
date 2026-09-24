from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import ToolMessage

from agent4ml.backend.agents.sandbox.contracts import SandboxProvider
from agent4ml.backend.agents.sandbox.integration.context import set_sandbox_id
from agent4ml.backend.agents.middlewares.sandbox_middleware import SandboxMiddleware
from langgraph.types import Command


def _make_request(
    tool_name: str = "bash",
    sandbox_state: dict | None = None,
    thread_id: str = "thread-1",
) -> MagicMock:
    """构造 mock ToolCallRequest。"""
    request = MagicMock()
    request.tool_call = {"name": tool_name, "id": "call-1", "args": {}}
    request.state = {"sandbox": sandbox_state}
    request.runtime = MagicMock()
    request.runtime.config = {"configurable": {"thread_id": thread_id}}
    return request


def _make_provider(sandbox_id: str = "sb-1") -> MagicMock:
    """构造 mock provider。"""
    provider = MagicMock(spec=SandboxProvider)
    provider.acquire.return_value = sandbox_id
    return provider


def _async_handler(return_value: Any = None) -> AsyncMock:
    """构造 async handler。"""
    return AsyncMock(return_value=return_value)


@pytest.fixture(autouse=True)
def _reset_context():
    set_sandbox_id(None)
    yield
    set_sandbox_id(None)


class TestNonSandboxTool:
    @pytest.mark.anyio
    async def test_web_search_no_acquire(self) -> None:
        provider = _make_provider()
        middleware = SandboxMiddleware(provider)
        request = _make_request(tool_name="web_search")

        await middleware.awrap_tool_call(request, _async_handler())

        provider.acquire.assert_not_called()

    @pytest.mark.anyio
    async def test_non_sandbox_passthrough(self) -> None:
        provider = _make_provider()
        middleware = SandboxMiddleware(provider)
        request = _make_request(tool_name="browse_page")

        await middleware.awrap_tool_call(request, _async_handler())
        provider.acquire.assert_not_called()


class TestSandboxToolFirstAcquire:
    @pytest.mark.anyio
    async def test_acquire_on_first_sandbox_tool(self) -> None:
        provider = _make_provider("sb-1")
        middleware = SandboxMiddleware(provider)
        request = _make_request(tool_name="bash")

        await middleware.awrap_tool_call(request, _async_handler())

        provider.acquire.assert_called_once_with("thread-1")

    @pytest.mark.anyio
    async def test_set_sandbox_id_after_acquire(self) -> None:
        provider = _make_provider("sb-1")
        middleware = SandboxMiddleware(provider)
        request = _make_request(tool_name="bash")

        await middleware.awrap_tool_call(request, _async_handler())

        from agent4ml.backend.agents.sandbox.integration.context import get_sandbox_id

        assert get_sandbox_id() == "sb-1"

    @pytest.mark.anyio
    async def test_command_on_first_acquire_with_tool_message(self) -> None:
        provider = _make_provider("sb-1")
        middleware = SandboxMiddleware(provider)
        request = _make_request(tool_name="bash")
        tool_msg = ToolMessage(content="result", tool_call_id="call-1")

        result = await middleware.awrap_tool_call(request, _async_handler(tool_msg))

        assert isinstance(result, Command)
        assert result.update["sandbox"] == {"sandbox_id": "sb-1"}
        assert result.update["messages"] == [tool_msg]

    @pytest.mark.anyio
    async def test_no_command_on_non_tool_message(self) -> None:
        provider = _make_provider("sb-1")
        middleware = SandboxMiddleware(provider)
        request = _make_request(tool_name="bash")

        result = await middleware.awrap_tool_call(request, _async_handler("string result"))

        assert result == "string result"


class TestSandboxToolSubsequentCall:
    @pytest.mark.anyio
    async def test_no_acquire_on_subsequent(self) -> None:
        provider = _make_provider("sb-1")
        middleware = SandboxMiddleware(provider)
        request = _make_request(tool_name="bash")

        set_sandbox_id("sb-1")

        await middleware.awrap_tool_call(request, _async_handler())

        provider.acquire.assert_not_called()

    @pytest.mark.anyio
    async def test_no_command_on_subsequent(self) -> None:
        provider = _make_provider("sb-1")
        middleware = SandboxMiddleware(provider)
        request = _make_request(tool_name="bash")
        tool_msg = ToolMessage(content="result", tool_call_id="call-1")

        set_sandbox_id("sb-1")

        result = await middleware.awrap_tool_call(request, _async_handler(tool_msg))

        assert result is tool_msg


class TestAafterAgent:
    @pytest.mark.anyio
    async def test_release_on_sandbox_present(self) -> None:
        provider = _make_provider()
        middleware = SandboxMiddleware(provider)
        state = {"sandbox": {"sandbox_id": "sb-1"}}

        await middleware.aafter_agent(state, MagicMock())

        provider.release.assert_called_once_with("sb-1")

    @pytest.mark.anyio
    async def test_no_release_on_sandbox_none(self) -> None:
        provider = _make_provider()
        middleware = SandboxMiddleware(provider)
        state = {"sandbox": None}

        await middleware.aafter_agent(state, MagicMock())

        provider.release.assert_not_called()

    @pytest.mark.anyio
    async def test_no_release_on_no_sandbox_key(self) -> None:
        provider = _make_provider()
        middleware = SandboxMiddleware(provider)
        state = {}

        await middleware.aafter_agent(state, MagicMock())

        provider.release.assert_not_called()


class TestBeforeModelSandboxRestore:
    """abefore_model 从 state["sandbox"] 恢复 ContextVar（块 D1 subagent 共享）。"""

    @pytest.mark.anyio
    async def test_restore_from_state_when_contextvar_none(self) -> None:
        from agent4ml.backend.agents.sandbox.integration.context import get_sandbox_id

        provider = _make_provider()
        middleware = SandboxMiddleware(provider)
        state = {"sandbox": {"sandbox_id": "parent123"}}

        await middleware.abefore_model(state, MagicMock())

        assert get_sandbox_id() == "parent123"

    @pytest.mark.anyio
    async def test_no_restore_when_state_sandbox_none(self) -> None:
        from agent4ml.backend.agents.sandbox.integration.context import get_sandbox_id

        provider = _make_provider()
        middleware = SandboxMiddleware(provider)
        state = {"sandbox": None}

        await middleware.abefore_model(state, MagicMock())

        assert get_sandbox_id() is None

    @pytest.mark.anyio
    async def test_no_restore_when_state_missing_sandbox_key(self) -> None:
        from agent4ml.backend.agents.sandbox.integration.context import get_sandbox_id

        provider = _make_provider()
        middleware = SandboxMiddleware(provider)
        state = {}

        await middleware.abefore_model(state, MagicMock())

        assert get_sandbox_id() is None

    @pytest.mark.anyio
    async def test_no_overwrite_when_contextvar_already_set(self) -> None:
        from agent4ml.backend.agents.sandbox.integration.context import get_sandbox_id

        set_sandbox_id("existing_id")
        provider = _make_provider()
        middleware = SandboxMiddleware(provider)
        state = {"sandbox": {"sandbox_id": "parent123"}}

        await middleware.abefore_model(state, MagicMock())

        assert get_sandbox_id() == "existing_id"

    @pytest.mark.anyio
    async def test_no_restore_when_sandbox_state_not_dict(self) -> None:
        from agent4ml.backend.agents.sandbox.integration.context import get_sandbox_id

        provider = _make_provider()
        middleware = SandboxMiddleware(provider)
        state = {"sandbox": "not a dict"}

        await middleware.abefore_model(state, MagicMock())

        assert get_sandbox_id() is None

    @pytest.mark.anyio
    async def test_no_restore_when_sandbox_id_missing(self) -> None:
        from agent4ml.backend.agents.sandbox.integration.context import get_sandbox_id

        provider = _make_provider()
        middleware = SandboxMiddleware(provider)
        state = {"sandbox": {}}

        await middleware.abefore_model(state, MagicMock())

        assert get_sandbox_id() is None

    @pytest.mark.anyio
    async def test_returns_none(self) -> None:
        provider = _make_provider()
        middleware = SandboxMiddleware(provider)
        state = {"sandbox": {"sandbox_id": "parent123"}}

        result = await middleware.abefore_model(state, MagicMock())

        assert result is None
