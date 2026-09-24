"""StrategyMiddleware — 接入 adapter，路由 6 hook 到 GovernanceStrategy bundle。

單一 AgentMiddleware 持一個策略 bundle 實例。before/after hook 構 ctx 調 bundle，
apply GovernanceResult（state_patch/metrics/jump_to）。wrap hook adapter 包 handler：
- wrap_model_call = PRE：bundle 改 request（request_override）→ handler
- wrap_tool_call = POST：handler 取 result → bundle 改 result（request_override）
wrap hook 的 state_patch/metrics 不適用（wrap 返 ModelCallResult/ToolMessage），
狀態寫走 before/after hook。
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any, override

from langchain.agents.middleware.types import (
    AgentMiddleware,
    ModelCallResult,
    ModelRequest,
    ModelResponse,
    hook_config,
)
from langchain_core.messages import ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.runtime import Runtime
from langgraph.types import Command

from agent4ml.backend.agents.context_engineering.contract import (
    GovernanceContext,
    GovernanceResult,
    GovernanceStrategy,
    apply_governance_result,
)
from agent4ml.backend.agents.context_engineering.utilities import token_counter
from agent4ml.backend.agents.state.types import ThreadState


class StrategyMiddleware(AgentMiddleware):
    """接入 adapter：6 hook 路由到 GovernanceStrategy bundle。"""

    state_schema = ThreadState  # type: ignore[assignment]

    def __init__(self, bundle: GovernanceStrategy, config: Any = None) -> None:
        self._bundle = bundle
        self._config = config

    def _ctx(
        self,
        state: ThreadState,
        runtime: Runtime,
        hook: str,
        **hook_specific: Any,
    ) -> GovernanceContext:
        return GovernanceContext(
            state=state,
            governance=state.get("governance"),
            config=self._config,
            token_counter=token_counter,
            runtime=runtime,
            hook=hook,
            **hook_specific,
        )

    # ---- state-channel hooks（apply GovernanceResult）----

    @override
    def before_agent(self, state: ThreadState, runtime: Runtime) -> dict[str, Any] | None:
        return apply_governance_result(state, self._bundle.before_agent(self._ctx(state, runtime, "before_agent", messages=state.get("messages") or [])))

    @override
    async def abefore_agent(self, state: ThreadState, runtime: Runtime) -> dict[str, Any] | None:
        result = await self._bundle.abefore_agent(self._ctx(state, runtime, "before_agent", messages=state.get("messages") or []))
        return apply_governance_result(state, result)

    @override
    def after_agent(self, state: ThreadState, runtime: Runtime) -> dict[str, Any] | None:
        return apply_governance_result(state, self._bundle.after_agent(self._ctx(state, runtime, "after_agent", messages=state.get("messages") or [])))

    @override
    async def aafter_agent(self, state: ThreadState, runtime: Runtime) -> dict[str, Any] | None:
        result = await self._bundle.aafter_agent(self._ctx(state, runtime, "after_agent", messages=state.get("messages") or []))
        return apply_governance_result(state, result)

    @hook_config(can_jump_to=["model"])
    @override
    def before_model(self, state: ThreadState, runtime: Runtime) -> dict[str, Any] | None:
        return apply_governance_result(state, self._bundle.before_model(self._ctx(state, runtime, "before_model", messages=state.get("messages") or [])))

    @hook_config(can_jump_to=["model"])
    @override
    async def abefore_model(self, state: ThreadState, runtime: Runtime) -> dict[str, Any] | None:
        result = await self._bundle.abefore_model(self._ctx(state, runtime, "before_model", messages=state.get("messages") or []))
        return apply_governance_result(state, result)

    @hook_config(can_jump_to=["model"])
    @override
    def after_model(self, state: ThreadState, runtime: Runtime) -> dict[str, Any] | None:
        return apply_governance_result(state, self._bundle.after_model(self._ctx(state, runtime, "after_model", messages=state.get("messages") or [])))

    @hook_config(can_jump_to=["model"])
    @override
    async def aafter_model(self, state: ThreadState, runtime: Runtime) -> dict[str, Any] | None:
        result = await self._bundle.aafter_model(self._ctx(state, runtime, "after_model", messages=state.get("messages") or []))
        return apply_governance_result(state, result)

    # ---- wrap hooks（adapter 包 handler，僅 apply request_override）----

    @override
    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelCallResult:
        state = getattr(getattr(request, "runtime", None), "state", None) or {}
        ctx = self._ctx(state, getattr(request, "runtime", None), "wrap_model_call", model_request=request, messages=getattr(request, "messages", None) or [], tools=getattr(request, "tools", None))
        result = self._bundle.wrap_model_call(ctx)
        req = result.request_override if result.request_override is not None else request
        return handler(req)

    @override
    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        state = getattr(getattr(request, "runtime", None), "state", None) or {}
        ctx = self._ctx(state, getattr(request, "runtime", None), "wrap_model_call", model_request=request, messages=getattr(request, "messages", None) or [], tools=getattr(request, "tools", None))
        result = await self._bundle.awrap_model_call(ctx)
        req = result.request_override if result.request_override is not None else request
        return await handler(req)

    @override
    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        tool_result = handler(request)
        state = getattr(getattr(request, "runtime", None), "state", None) or {}
        ctx = self._ctx(state, getattr(request, "runtime", None), "wrap_tool_call", tool_call_request=request, tool_result=tool_result)
        result = self._bundle.wrap_tool_call(ctx)
        return result.request_override if result.request_override is not None else tool_result

    @override
    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
    ) -> ToolMessage | Command:
        tool_result = await handler(request)
        state = getattr(getattr(request, "runtime", None), "state", None) or {}
        ctx = self._ctx(state, getattr(request, "runtime", None), "wrap_tool_call", tool_call_request=request, tool_result=tool_result)
        result = await self._bundle.awrap_tool_call(ctx)
        return result.request_override if result.request_override is not None else tool_result
