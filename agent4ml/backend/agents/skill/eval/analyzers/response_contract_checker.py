"""ResponseContractChecker — 响应层 eval（pre-promotion，检查 candidate SKILL.md）。

D-L3-4: contract-aware 替代 L2a ProgrammaticEvalBridge 固定 7 mode。
用 ContractCompiler 从 skill 文本编译适用规则，跑在 candidate 上，产 EvalResult。
硬失败 short-circuit（nonempty/frontmatter_parseable 失败 → 直接 reject）。
复用 L2a ProgrammaticEvalBridge 的 _check_* 静态方法（单一真相源，不重复实现）。
"""
from __future__ import annotations

from agent4ml.backend.agents.skill.evolution.types import (
    EvalEvidence,
    EvalResult,
)
from agent4ml.backend.agents.skill.eval.analyzers import checks
from agent4ml.backend.agents.skill.eval.analyzers.contract_compiler import ContractCompiler


class ResponseContractChecker:
    """响应层 eval：编译规则 + 跑 candidate + 产 EvalResult。"""

    def __init__(self, compiler: ContractCompiler | None = None) -> None:
        self._compiler = compiler or ContractCompiler()

    def check(
        self,
        candidate_content: str,
        baseline_content: str,
    ) -> EvalResult:
        """跑编译规则，返 EvalResult。硬失败 short-circuit。"""
        rules = self._compiler.compile(candidate_content)

        evidence: list[EvalEvidence] = []
        hard_failures: list[str] = []
        passed = 0
        total = 0

        for rule in rules:
            total += 1
            cand_pass = self._run_rule(rule.rule_id, candidate_content, rule.params)
            base_pass = self._run_rule(rule.rule_id, baseline_content, rule.params) if baseline_content else True

            if cand_pass:
                passed += 1
            if rule.hard and not cand_pass:
                hard_failures.append(rule.rule_id)

            evidence.append(EvalEvidence(
                kind="programmatic_rule",
                rule_name=rule.rule_id,
                baseline_pass=base_pass,
                candidate_pass=cand_pass,
            ))

        score = passed / total if total else 0.0
        recommendation = "reject" if hard_failures else "accept"

        return EvalResult(
            score=score,
            metric="hard",
            hard_failures=tuple(hard_failures),
            evidence=tuple(evidence),
            confidence=0.7,
            recommendation=recommendation,  # type: ignore[arg-type]
        )

    @staticmethod
    def _run_rule(rule_id: str, content: str, params: dict) -> bool:
        """分发规则到 checks 模块的检查函数。"""
        if rule_id == "nonempty":
            return checks.check_nonempty(content)
        if rule_id == "json_parseable":
            return checks.check_json_parseable(content)
        if rule_id == "must_cite":
            return checks.check_must_cite(content)
        if rule_id == "lead_with_conclusion":
            return checks.check_lead_with_conclusion(content)
        if rule_id == "paragraph_limit":
            return checks.check_paragraph_limit(content)
        if rule_id == "no_unfounded_claims":
            return checks.check_no_unfounded_claims(content)
        if rule_id == "semantic_density":
            if not content:
                return False
            density = checks.semantic_density(content)
            return checks.SEMANTIC_DENSITY_MIN <= density <= checks.SEMANTIC_DENSITY_MAX
        return True  # 未知规则默认 pass
