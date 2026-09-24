"""StrategyMiddleware delegate 单测：stub bundle 验证 6 hook 路由 + apply。"""

from __future__ import annotations

from types import SimpleNamespace

from agent4ml.backend.agents.context_engineering.contract import GovernanceResult
from agent4ml.backend.agents.context_engineering.strategy_middleware import StrategyMiddleware


class _StubBundle:
    """满足 GovernanceStrategy 的 stub，各 hook 返回可配置结果。"""

    def __init__(
        self,
        before_agent_result: GovernanceResult | None = None,
        wrap_model_override: object | None = None,
        wrap_tool_override: object | None = None,
    ) -> None:
        self.before_agent_result = before_agent_result or GovernanceResult()
        self.wrap_model_override = wrap_model_override
        self.wrap_tool_override = wrap_tool_override
        self.wrap_model_ctx_seen = None
        self.wrap_tool_ctx_seen = None

    def before_agent(self, ctx):
        return self.before_agent_result

    async def abefore_agent(self, ctx):
        return self.before_agent_result

    def after_agent(self, ctx):
        return GovernanceResult()

    async def aafter_agent(self, ctx):
        return GovernanceResult()

    def before_model(self, ctx):
        return GovernanceResult()

    async def abefore_model(self, ctx):
        return GovernanceResult()

    def after_model(self, ctx):
        return GovernanceResult()

    async def aafter_model(self, ctx):
        return GovernanceResult()

    def wrap_model_call(self, ctx):
        self.wrap_model_ctx_seen = ctx
        return GovernanceResult(request_override=self.wrap_model_override)

    async def awrap_model_call(self, ctx):
        self.wrap_model_ctx_seen = ctx
        return GovernanceResult(request_override=self.wrap_model_override)

    def wrap_tool_call(self, ctx):
        self.wrap_tool_ctx_seen = ctx
        return GovernanceResult(request_override=self.wrap_tool_override)

    async def awrap_tool_call(self, ctx):
        self.wrap_tool_ctx_seen = ctx
        return GovernanceResult(request_override=self.wrap_tool_override)


def _state():
    return {"messages": [], "governance": None}


def test_before_agent_delegates_and_applies_state_patch() -> None:
    bundle = _StubBundle(
        before_agent_result=GovernanceResult(state_patch={"governance": {"vol.x": 1}})
    )
    m = StrategyMiddleware(bundle)
    patch = m.before_agent(_state(), None)
    assert patch == {"governance": {"vol.x": 1}}


def test_before_agent_empty_returns_none() -> None:
    m = StrategyMiddleware(_StubBundle())
    assert m.before_agent(_state(), None) is None


def test_wrap_model_call_pre_override_routes_to_handler() -> None:
    seen_by_handler = []

    def handler(req):
        seen_by_handler.append(req)
        return "resp"

    override_req = SimpleNamespace(name="override")
    bundle = _StubBundle(wrap_model_override=override_req)
    m = StrategyMiddleware(bundle)
    request = SimpleNamespace(runtime=SimpleNamespace(state=_state()), messages=[], tools=[])
    result = m.wrap_model_call(request, handler)
    assert seen_by_handler == [override_req]
    assert result == "resp"


def test_wrap_model_call_no_override_uses_original() -> None:
    def handler(req):
        return "resp"

    m = StrategyMiddleware(_StubBundle())  # wrap_model_override=None
    request = SimpleNamespace(runtime=SimpleNamespace(state=_state()), messages=[], tools=[])
    assert m.wrap_model_call(request, handler) == "resp"


def test_wrap_tool_call_post_override_replaces_result() -> None:
    def handler(req):
        return "original_result"

    bundle = _StubBundle(wrap_tool_override="patched_result")
    m = StrategyMiddleware(bundle)
    request = SimpleNamespace(runtime=SimpleNamespace(state=_state()))
    assert m.wrap_tool_call(request, handler) == "patched_result"


def test_wrap_tool_call_no_override_returns_original() -> None:
    def handler(req):
        return "original_result"

    m = StrategyMiddleware(_StubBundle())
    request = SimpleNamespace(runtime=SimpleNamespace(state=_state()))
    assert m.wrap_tool_call(request, handler) == "original_result"
