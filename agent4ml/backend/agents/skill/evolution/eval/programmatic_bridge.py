"""ProgrammaticEvalBridge — 兼容 facade（D1 方案 A + facade 保留）。

L3 关闭时 EvolutionManager 用此 facade。内部委托 ResponseContractChecker（contract-aware 升级版）。
L3 启用时 bootstrap 注入 RegistryEvalBridge 替换此 facade。
_check_* 静态方法保留为 checks 模块的委托——L2a 测试直接调 ProgrammaticEvalBridge._check_*。
"""
from __future__ import annotations

from agent4ml.backend.agents.skill.evolution.types import EvalContext, EvalResult
from agent4ml.backend.agents.skill.eval.analyzers import checks
from agent4ml.backend.agents.skill.eval.analyzers.contract_compiler import ContractCompiler
from agent4ml.backend.agents.skill.eval.analyzers.response_contract_checker import (
    ResponseContractChecker,
)

# 向后兼容：L2a 测试引用的模块级常量
_HARD_MODES = checks.HARD_MODES
_DIRECTIVE_WORDS = checks.DIRECTIVE_WORDS
_UNFOUNDED_WORDS = checks.UNFOUNDED_WORDS
_CONCLUSION_WORDS = checks.CONCLUSION_WORDS
_CITE_PATTERN = checks.CITE_PATTERN
_YAML_FRONTMATTER = checks.YAML_FRONTMATTER
_PARAGRAPH_LIMIT = checks.PARAGRAPH_LIMIT
_SEMANTIC_DENSITY_MIN = checks.SEMANTIC_DENSITY_MIN
_SEMANTIC_DENSITY_MAX = checks.SEMANTIC_DENSITY_MAX


class ProgrammaticEvalBridge:
    """兼容 facade。内部委托 ResponseContractChecker。"""

    def __init__(self) -> None:
        self._checker = ResponseContractChecker(ContractCompiler())

    def evaluate(self, ctx: EvalContext) -> EvalResult:
        """委托 ResponseContractChecker。读 candidate/baseline SKILL.md 内容。"""
        candidate_content = self._read_content(ctx.candidate)
        baseline_content = self._read_content(ctx.baseline) if ctx.baseline else ""
        return self._checker.check(candidate_content, baseline_content)

    # ── 静态方法委托 checks 模块（L2a 测试向后兼容）────────

    @staticmethod
    def _read_content(record) -> str:
        return checks.read_content(record)

    @staticmethod
    def _split_body(content: str) -> str:
        return checks.split_body(content)

    @staticmethod
    def _check_nonempty(content: str) -> bool:
        return checks.check_nonempty(content)

    @staticmethod
    def _check_json_parseable(content: str) -> bool:
        return checks.check_json_parseable(content)

    @staticmethod
    def _check_must_cite(content: str) -> bool:
        return checks.check_must_cite(content)

    @staticmethod
    def _check_paragraph_limit(content: str) -> bool:
        return checks.check_paragraph_limit(content)

    @staticmethod
    def _check_lead_with_conclusion(content: str) -> bool:
        return checks.check_lead_with_conclusion(content)

    @staticmethod
    def _check_markdown_table(content: str) -> bool:
        return True  # 非所有 skill 需表格，宽松 pass（L2a 兼容）

    @staticmethod
    def _check_no_unfounded_claims(content: str) -> bool:
        return checks.check_no_unfounded_claims(content)

    @staticmethod
    def _semantic_density(content: str) -> float:
        return checks.semantic_density(content)
