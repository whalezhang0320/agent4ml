"""L3 EvalBridge Protocol — L2 调 L3 的唯一入口契约（L3 自建，不共享 skill）.

设计（43 文档 §4.1 + §11.1 L3-2.5 + spec.md EvalBridge Requirement）:
- runtime_checkable Protocol（Agent4ML 既有 pattern，与 L2 MetricsView/EvolutionArtifact 一致）
- 不共享 skill EvalBridge（skill 用 SkillRecord 专属类型不能跨模块共享，L3-2.5 决策 c）
- OrchestrationBridge + MultiagentProgrammaticFacade 实现此 Protocol
- evaluate(ctx) → EvalResult 同步调用（与 L1 D10 sync only 一致，L3-2.3 决策 a）
- fail-closed（异常返 EvalResult(success=False)，不抛异常，L3-2.4 决策 a）
- list_available_methods() 返已注册 adapter 方法名 tuple
- health_check() 返 registry 非空时 True
- 复用 L2 自建 EvalResult（import from evolution/promotion_gate.py）
"""


from typing import Protocol, runtime_checkable

from agent4ml.backend.agents.multiagent.eval.registry import SpecialistEvalRegistry
from agent4ml.backend.agents.multiagent.eval.types import EvalContext
from agent4ml.backend.agents.multiagent.evolution.promotion_gate import EvalResult


@runtime_checkable
class EvalBridge(Protocol):
    """L3 EvalBridge Protocol（L3 自建，不共享 skill EvalBridge）.

    skill EvalBridge 用 SkillRecord 专属类型不能跨模块共享（L3-2.5 决策 c）.
    OrchestrationBridge（L3 启用时）+ MultiagentProgrammaticFacade（L3 未启用时）实现此 Protocol.
    L2 PromotionGate.bridge 参数类型即此 Protocol.
    """

    def evaluate(self, ctx: EvalContext) -> EvalResult:
        """同步阻塞调用（与 L1 D10 sync only 一致）.

        fail-closed: 异常时返 EvalResult(success=False, failure_reason=...)，不抛异常.
        """
        ...

    def list_available_methods(self) -> tuple[str, ...]:
        """返已注册 EvalAdapter 方法名 tuple（如 ('programmatic', 'llm_judge', 'longitudinal_pairs')）."""
        ...

    def health_check(self) -> bool:
        """registry 非空时 True（至少 1 个 adapter 注册）."""
        ...


class OrchestrationBridge:
    """L3 multiagent 评估转接层实现，实现 EvalBridge Protocol（43 文档 §4.1）.

    L2 调 L3 的唯一入口（PromotionGate.bridge 参数注入）.
    evaluate(ctx) → _select_method(ctx) 选 adapter → adapter.evaluate(ctx).
    fail-closed: adapter 异常返 EvalResult(success=False)，不抛异常（L3-2.4 决策 a）.
    _select_method: eval_method_hint > expected_outcome(longitudinal_pairs) > open_ended(llm_judge) > programmatic.
    """

    def __init__(self, adapter_registry: SpecialistEvalRegistry) -> None:
        self._adapters = adapter_registry

    def evaluate(self, ctx: EvalContext) -> EvalResult:
        method = self._select_method(ctx)
        adapter = self._adapters.get(method)
        if adapter is None:
            return EvalResult(
                candidate_score=0.0, baseline_score=0.0,
                ci_low=0.0, ci_high=0.0, sample_size=0,
                method_used=method, raw_data_ref=None,
                success=False, failure_reason=f"adapter '{method}' not registered",
            )
        try:
            return adapter.evaluate(ctx)
        except Exception as exc:
            return EvalResult(
                candidate_score=0.0, baseline_score=0.0,
                ci_low=0.0, ci_high=0.0, sample_size=0,
                method_used=method, raw_data_ref=None,
                success=False, failure_reason=str(exc),
            )

    def _select_method(self, ctx: EvalContext) -> str:
        """根据 ctx 特征自动选 adapter（L3-3.3 决策 c：配置默认 + Bridge 自动覆盖）."""
        if ctx.eval_method_hint:
            return ctx.eval_method_hint
        if ctx.task_sample and all(t.expected_outcome for t in ctx.task_sample):
            return "longitudinal_pairs"
        if ctx.metadata.get("task_type") == "open_ended":
            return "llm_judge"
        return "programmatic"

    def list_available_methods(self) -> tuple[str, ...]:
        return self._adapters.list_methods()

    def health_check(self) -> bool:
        return len(self._adapters.list_methods()) > 0
