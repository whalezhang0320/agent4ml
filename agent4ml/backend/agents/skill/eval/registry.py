"""EvalRegistry + RegistryEvalBridge — L2 唯一看见的 L3 实现。

D-L3-2: EvalBridge Protocol 保持 L2 已有。RegistryEvalBridge 替换 ProgrammaticEvalBridge。
D-L3-8: fail-closed — registry 空 / checker 异常 → reject EvalResult，不裸跑（D8）。

EvalRegistry 实例级注册（不做 class-level 全局注册），bootstrap 装配。
RegistryEvalBridge 实现 EvalBridge Protocol，内部调 ResponseContractChecker。
"""
from __future__ import annotations

from pathlib import Path

from agent4ml.backend.agents.skill.evolution.types import EvalContext, EvalEvidence, EvalResult
from agent4ml.backend.agents.skill.eval.analyzers import checks
from agent4ml.backend.agents.skill.eval.analyzers.contract_compiler import ContractCompiler
from agent4ml.backend.agents.skill.eval.analyzers.response_contract_checker import (
    ResponseContractChecker,
)


class EvalRegistry:
    """实例级 registry，bootstrap 装配。不做 class-level 全局注册。"""

    def __init__(self, contract_checker: ResponseContractChecker) -> None:
        self._contract_checker = contract_checker

    def get_contract_checker(self) -> ResponseContractChecker:
        return self._contract_checker


class RegistryEvalBridge:
    """L2 唯一看见的 L3 实现。实现 EvalBridge Protocol。fail-closed。"""

    def __init__(self, registry: EvalRegistry) -> None:
        self._registry = registry

    def evaluate(self, ctx: EvalContext) -> EvalResult:
        """调 ResponseContractChecker 产 EvalResult。异常 fail-closed reject。"""
        try:
            checker = self._registry.get_contract_checker()
            candidate_content = checks.read_content(ctx.candidate)
            baseline_content = (
                checks.read_content(ctx.baseline)
                if ctx.baseline
                else ""
            )
            return checker.check(candidate_content, baseline_content)
        except Exception as exc:
            return EvalResult(
                score=0.0,
                metric="hard",
                hard_failures=("eval_exception",),
                evidence=(
                    EvalEvidence(
                        kind="programmatic_rule",
                        rule_name="eval_bridge",
                        baseline_pass=False,
                        candidate_pass=False,
                        detail=str(exc),
                    ),
                ),
                confidence=0.0,
                recommendation="reject",
            )


def build_default_registry() -> EvalRegistry:
    """建默认 registry（含 ContractCompiler + ResponseContractChecker）。bootstrap 用。"""
    checker = ResponseContractChecker(ContractCompiler())
    return EvalRegistry(checker)
