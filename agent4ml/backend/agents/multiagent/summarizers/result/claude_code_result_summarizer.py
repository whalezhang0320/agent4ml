"""ClaudeCodeResultSummarizer — extends base with review completeness + suggestions.

设计（design.md §4）:
- 扩展 base：review 意见完整性 + 修改建议存在性
- programmatic eval floor（INV#10）
"""
from __future__ import annotations

import re

from agent4ml.backend.agents.multiagent.summarizers.result.base import (
    BaseResultSummarizer,
)
from agent4ml.backend.agents.multiagent.types import ArtifactRef

_SUGGESTION_RE = re.compile(
    r"(?:suggest|recommend|should|fix|change|improve|consider)", re.IGNORECASE
)
_ISSUE_RE = re.compile(r"(?:issue|problem|bug|concern|risk|warning)", re.IGNORECASE)


class ClaudeCodeResultSummarizer(BaseResultSummarizer):
    """Claude Code specialist 输出端转换器（review 完整性 + 修改建议）。"""

    def __init__(self) -> None:
        super().__init__(specialist_name="claude")

    def _evaluate_success(
        self,
        raw_output: str,
        artifacts: list[ArtifactRef],
        success_criteria: str,
    ) -> bool:
        if not super()._evaluate_success(raw_output, artifacts, success_criteria):
            return False
        if not self._has_suggestions(raw_output):
            return False
        return True

    def _has_suggestions(self, raw_output: str) -> bool:
        return bool(_SUGGESTION_RE.search(raw_output))

    def _has_issues(self, raw_output: str) -> bool:
        return bool(_ISSUE_RE.search(raw_output))

    def _extract_gap(self, raw_output: str, success_criteria: str) -> str:
        base_gap = super()._extract_gap(raw_output, success_criteria)
        if not self._has_suggestions(raw_output):
            return f"{base_gap}. No modification suggestions found."
        if not self._has_issues(raw_output):
            return f"{base_gap}. No review issues identified."
        return base_gap
