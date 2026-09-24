"""ContractCompiler — 从 skill 文本自动编译 contract 规则（AutoSkill EvalCompiler 模型）。

D-L3-4: contract-aware 替代 L2a 固定 7 mode。外部 skill 零改造——不要求 frontmatter eval-contract 声明，
扫 skill 自身文本关键词决定编译哪些规则。

规则层级：
- 全局硬规则（始终跑，失败触发 hard_failure）：nonempty + frontmatter_parseable
- contract 规则（skill 文本提到才编译，失败不触发 hard_failure）：
  must_cite / lead_with_conclusion / paragraph_limit
- soft 规则（始终跑，不作 hard_failure）：no_unfounded_claims + semantic_density

借鉴：AutoSkill/SkillEvo/evals.py EvalCompiler._mentions_sources / _mentions_conclusion_first 等。
"""
from __future__ import annotations

import re

from agent4ml.backend.agents.skill.eval.types import ContractRule

_RE_PARAGRAPH_LIMIT = re.compile(
    r"(不超过|少于|最多|within|less than|at most)\s*(\d+)\s*(段|paragraph)",
    re.IGNORECASE,
)


class ContractCompiler:
    """从 skill 文本编译 contract 规则。"""

    def compile(self, skill_content: str) -> list[ContractRule]:
        """编译规则。全局硬规则始终含；contract 规则按 skill 文本关键词；soft 规则始终含。"""
        rules: list[ContractRule] = [
            ContractRule("nonempty", kind="programmatic", hard=True,
                         description="SKILL.md body 非空"),
            ContractRule("json_parseable", kind="programmatic", hard=True,
                         description="frontmatter YAML 可解析"),
        ]

        corpus = skill_content.lower()
        if self._mentions_sources(corpus):
            rules.append(ContractRule("must_cite", kind="programmatic", hard=False,
                                      description="skill 声明引用来源，SKILL.md 应含引用标记"))
        if self._mentions_conclusion_first(corpus):
            rules.append(ContractRule("lead_with_conclusion", kind="programmatic", hard=False,
                                      description="skill 声明先给结论，首段应含结论词"))
        if self._paragraph_limit(corpus) > 0:
            rules.append(ContractRule("paragraph_limit", kind="programmatic", hard=False,
                                      description="skill 声明段落数限制",
                                      params={"max": self._paragraph_limit(corpus)}))

        # soft 规则（始终跑，不作 hard_failure）
        rules.append(ContractRule("no_unfounded_claims", kind="programmatic", hard=False,
                                  description="无绝对化无据声明"))
        rules.append(ContractRule("semantic_density", kind="programmatic", hard=False,
                                  description="指令性词密度在合理区间"))
        return rules

    # ── 关键词检测（借鉴 AutoSkill evals.py）──────────────

    @staticmethod
    def _mentions_sources(corpus: str) -> bool:
        return any(k in corpus for k in (
            "引用来源", "标注来源", "注明来源", "cite sources",
            "with sources", "provide sources", "source-backed",
        ))

    @staticmethod
    def _mentions_conclusion_first(corpus: str) -> bool:
        return any(k in corpus for k in (
            "先给结论", "结论在前", "先说结论",
            "answer first", "lead with the conclusion", "bottom line first",
        ))

    @staticmethod
    def _paragraph_limit(corpus: str) -> int:
        for match in _RE_PARAGRAPH_LIMIT.finditer(corpus):
            try:
                return max(1, int(match.group(2)))
            except Exception:
                continue
        return 0
