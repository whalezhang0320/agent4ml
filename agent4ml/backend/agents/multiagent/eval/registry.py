"""L3 SpecialistEvalRegistry + EvalAdapter Protocol.

设计（43 文档 §4.2 + §11.2 L3-3.1/L3-3.2 + spec.md SpecialistEvalRegistry Requirement）:
- EvalAdapter: runtime_checkable Protocol（evaluate(ctx) → EvalResult + health_check() → bool）
- SpecialistEvalRegistry: 实例级 dict（不做 class-level 全局注册，遵循 Agent4ML 禁全局 singleton 原则）
- pattern 复用 skill EvalRegistry（L3-3.2 决策 a）
- 3 adapter 各自独立实现（programmatic/llm_judge/longitudinal_pairs，Bridge 自动选）
- 复用 L2 自建 EvalResult（import from evolution/promotion_gate.py）
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from agent4ml.backend.agents.multiagent.eval.types import EvalContext
from agent4ml.backend.agents.multiagent.evolution.promotion_gate import EvalResult


@runtime_checkable
class EvalAdapter(Protocol):
    """L3 EvalAdapter Protocol（3 adapter 各自实现，L3-3.1 决策 b）.

    ProgrammaticAdapter / LLMJudgeAdapter / LongitudinalPairsAdapter 实现此 Protocol.
    SpecialistEvalRegistry 统一管理（register/get/list_methods）.
    """

    def evaluate(self, ctx: EvalContext) -> EvalResult:
        """评估 candidate vs baseline，返 EvalResult（含 CI + success + method_used）."""
        ...

    def health_check(self) -> bool:
        """adapter 健康检查（如 LLM-judge 检查 model 可用 / programmatic 检查 ResultSummarizer 可用）."""
        ...


class SpecialistEvalRegistry:
    """实例级 registry，管理 3 个 EvalAdapter，bootstrap 装配（L3-3.2 决策 a）.

    不做 class-level 全局注册（遵循 Agent4ML 禁全局 singleton 原则）.
    pattern 复用 skill EvalRegistry（实例级 dict + register/get/list_methods）.
    OrchestrationBridge.__init__ 接收此 registry，evaluate 时按 method 名 get adapter.
    """

    def __init__(self) -> None:
        self._adapters: dict[str, EvalAdapter] = {}

    def register(self, method: str, adapter: EvalAdapter) -> None:
        """注册 adapter（method 名如 'programmatic'/'llm_judge'/'longitudinal_pairs'）.

        重复注册覆盖（后注册的生效，bootstrap 装配时按顺序）.
        """
        self._adapters[method] = adapter

    def get(self, method: str) -> EvalAdapter | None:
        """按 method 名取 adapter，未注册返 None."""
        return self._adapters.get(method)

    def list_methods(self) -> tuple[str, ...]:
        """已注册 method 名 tuple（注册顺序）."""
        return tuple(self._adapters.keys())
