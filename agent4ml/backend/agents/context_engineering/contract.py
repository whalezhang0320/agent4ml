"""接入契約層：GovernanceStrategy Protocol + GovernanceContext/Result/Metric + MemorySink + helpers。

骨架族無關接入契約。策略 bundle 實現 GovernanceStrategy 6 hook，內部完全自由。
StrategyMiddleware adapter 構 GovernanceContext 調 bundle，apply GovernanceResult。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Protocol, runtime_checkable


@dataclass(frozen=True)
class GovernanceContext:
    """hook 級統一入參。策略 bundle 按 hook 取所需字段。"""

    state: Mapping[str, Any]
    governance: dict[str, Any] | None
    config: Any
    token_counter: Callable[[list], int]
    runtime: Any
    hook: str
    # hook-specific（不適用時 None）
    messages: list | None = None
    tools: list | None = None
    model_request: Any | None = None
    tool_call_request: Any | None = None
    tool_result: Any | None = None


@dataclass(frozen=True)
class GovernanceResult:
    """hook 級統一出參。

    - state_patch：寫 ThreadState（含 governance），持久
    - request_override：替換 request payload，request-scoped 不持久
    - messages_patch：消息級操作（RemoveMessage / 替換）
    - metrics：本次產出 metric
    - jump_to：跳轉目標節點
    """

    state_patch: dict[str, Any] | None = None
    request_override: Any | None = None
    messages_patch: list | None = None
    metrics: list[GovernanceMetric] = field(default_factory=list)
    jump_to: str | None = None


@dataclass(frozen=True)
class GovernanceMetric:
    """策略級 metric。strategy_name + metric_key 組合前綴寫入 governance.metrics。"""

    strategy_name: str
    metric_key: str
    value: int | float
    run_id: str = ""
    thread_id: str = ""


@runtime_checkable
class GovernanceStrategy(Protocol):
    """策略 bundle 接入契約。6 hook（sync + a 前綴 async 對）。

    方法名為 hook 名，不預設能力語義。策略內部協議/schema/執行器完全自由。
    wrap_model_call/wrap_tool_call 由 StrategyMiddleware adapter 包 handler，
    bundle 經 GovernanceContext 收 request/result，不直接持 handler。
    """

    def before_agent(self, ctx: GovernanceContext) -> GovernanceResult: ...
    async def abefore_agent(self, ctx: GovernanceContext) -> GovernanceResult: ...

    def after_agent(self, ctx: GovernanceContext) -> GovernanceResult: ...
    async def aafter_agent(self, ctx: GovernanceContext) -> GovernanceResult: ...

    def before_model(self, ctx: GovernanceContext) -> GovernanceResult: ...
    async def abefore_model(self, ctx: GovernanceContext) -> GovernanceResult: ...

    def after_model(self, ctx: GovernanceContext) -> GovernanceResult: ...
    async def aafter_model(self, ctx: GovernanceContext) -> GovernanceResult: ...

    def wrap_model_call(self, ctx: GovernanceContext) -> GovernanceResult: ...
    async def awrap_model_call(self, ctx: GovernanceContext) -> GovernanceResult: ...

    def wrap_tool_call(self, ctx: GovernanceContext) -> GovernanceResult: ...
    async def awrap_tool_call(self, ctx: GovernanceContext) -> GovernanceResult: ...


@runtime_checkable
class MemorySink(Protocol):
    """短期治理層與長期記憶層邊界接口。策略壓縮丟棄消息前可調 flush。"""

    def flush(self, messages_to_drop: list) -> None: ...
    async def aflush(self, messages_to_drop: list) -> None: ...


def merge_metrics_into_governance(
    governance: dict[str, Any] | None, metrics: list[GovernanceMetric]
) -> dict[str, Any]:
    """將 GovernanceMetric 列表合併進 governance，按 ``{strategy_name}.{metric_key}`` 前綴寫 metrics 子 dict。"""
    if not metrics:
        return dict(governance) if governance else {}
    g: dict[str, Any] = dict(governance) if governance else {}
    metrics_dict: dict[str, Any] = dict(g.get("metrics") or {})
    for m in metrics:
        metrics_dict[f"{m.strategy_name}.{m.metric_key}"] = m.value
    g["metrics"] = metrics_dict
    return g


def apply_governance_result(
    state: Mapping[str, Any], result: GovernanceResult
) -> dict[str, Any] | None:
    """將 GovernanceResult 轉為 state-channel hook 返回的 dict patch。

    適用於 before/after_agent、before/after_model（返回 dict 的 hook）。
    wrap_model_call/wrap_tool_call 的 request_override 由 StrategyMiddleware adapter inline 處理。
    """
    patch: dict[str, Any] = dict(result.state_patch or {})
    if result.messages_patch:
        patch["messages"] = result.messages_patch
    if result.metrics:
        base_gov = patch.get("governance", state.get("governance"))
        patch["governance"] = merge_metrics_into_governance(base_gov, result.metrics)
    if result.jump_to:
        patch["jump_to"] = result.jump_to
    return patch or None
