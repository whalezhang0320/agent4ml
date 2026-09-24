"""CodexResultSummarizer — extends base with test pass rate + diff legality.

设计（design.md §4）:
- 扩展 base：测试通过率解析 + diff 合法性校验
- programmatic eval floor（INV#10）
"""
from __future__ import annotations

import re

from agent4ml.backend.agents.multiagent.summarizers.result.base import (
    BaseResultSummarizer,
)
from agent4ml.backend.agents.multiagent.types import ArtifactRef

_TEST_RESULT_RE = re.compile(r"(\d+)\s+passed", re.IGNORECASE)
_TEST_FAIL_RE = re.compile(r"(\d+)\s+failed", re.IGNORECASE)
_DIFF_MARKER_RE = re.compile(r"^@@|^---|^\+\+\+", re.MULTILINE)


class CodexResultSummarizer(BaseResultSummarizer):
    """Codex specialist 输出端转换器（测试通过率 + diff 合法性）。"""

    def __init__(self) -> None:
        super().__init__(specialist_name="codex")

    def _evaluate_success(
        self,
        raw_output: str,
        artifacts: list[ArtifactRef],
        success_criteria: str,
    ) -> bool:
        if not super()._evaluate_success(raw_output, artifacts, success_criteria):
            return False
        passed = self._parse_test_pass_rate(raw_output)
        if passed is not None and passed == 0:
            return False
        return True

    def _parse_test_pass_rate(self, raw_output: str) -> int | None:
        match = _TEST_RESULT_RE.search(raw_output)
        if match:
            return int(match.group(1))
        return None

    def _extract_gap(self, raw_output: str, success_criteria: str) -> str:
        base_gap = super()._extract_gap(raw_output, success_criteria)
        failed = _TEST_FAIL_RE.search(raw_output)
        if failed:
            return f"{base_gap}. Tests failed: {failed.group(1)}"
        if not _DIFF_MARKER_RE.search(raw_output):
            return f"{base_gap}. No diff markers found in output."
        return base_gap
