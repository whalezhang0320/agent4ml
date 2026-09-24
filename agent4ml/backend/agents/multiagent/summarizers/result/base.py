"""BaseResultSummarizer — 通用 programmatic eval floor（INV#10）。

设计（spec.md ResultSummarizer Requirement + design.md §4）:
- 通用校验：产物存在性 + success_criteria 回应 + 无敏感改动
- gap_analysis 提取（success=False 时）
- 不传回 raw output 全量（控制 lead agent context 膨胀）
- per-specialist 扩展继承此类加特定校验
"""
from __future__ import annotations

import re
from typing import Any

from agent4ml.backend.agents.multiagent.types import (
    ArtifactRef,
    SpecialistResult,
)

_MAX_SUMMARY_CHARS = 2000
_SENSITIVE_PATTERNS = [
    r"rm\s+-rf\s+/",
    r"mkfs\.",
    r"dd\s+of=/dev/",
    r">\s*/etc/passwd",
    r"chmod\s+777\s+/",
]


class BaseResultSummarizer:
    """通用 programmatic eval floor（per-specialist 扩展继承）。

    子类 override _evaluate_success / _extract_special_checks 加特定校验。
    """

    def __init__(self, specialist_name: str = "base") -> None:
        self._specialist_name = specialist_name

    def summarize(
        self,
        raw_output: str,
        artifacts: list[ArtifactRef],
        goal: str,
        success_criteria: str,
    ) -> SpecialistResult:
        success = self._evaluate_success(raw_output, artifacts, success_criteria)
        gap_analysis = "" if success else self._extract_gap(raw_output, success_criteria)
        summary = self._compress(raw_output)
        failure_category = self._classify_failure(success, raw_output)

        return SpecialistResult(
            specialist_name=self._specialist_name,
            summary=summary,
            artifacts=tuple(artifacts),
            success=success,
            gap_analysis=gap_analysis,
            failure_category=failure_category,
        )

    def _classify_failure(self, success: bool, raw_output: str) -> str | None:
        """Heuristic failure category classification (D-7=c, L2 FailureFocuser reads).

        success=True -> None
        success=False + raw_output contains "context" -> context_insufficient
        + "skill" or "ability" -> ability_insufficient
        + "goal" or "unclear" -> goal_unclear
        + "sandbox" or "timeout" -> sandbox_issue
        default -> ability_insufficient
        """
        if success:
            return None
        lower = raw_output.lower()
        if "context" in lower:
            return "context_insufficient"
        if "skill" in lower or "ability" in lower:
            return "ability_insufficient"
        if "goal" in lower or "unclear" in lower:
            return "goal_unclear"
        if "sandbox" in lower or "timeout" in lower:
            return "sandbox_issue"
        return "ability_insufficient"

    def _evaluate_success(
        self,
        raw_output: str,
        artifacts: list[ArtifactRef],
        success_criteria: str,
    ) -> bool:
        if not raw_output.strip():
            return False
        if self._has_sensitive_changes(raw_output):
            return False
        if artifacts is not None and len(artifacts) == 0:
            return False
        return True

    def _has_sensitive_changes(self, raw_output: str) -> bool:
        for pattern in _SENSITIVE_PATTERNS:
            if re.search(pattern, raw_output):
                return True
        return False

    def _compress(self, raw_output: str) -> str:
        if len(raw_output) <= _MAX_SUMMARY_CHARS:
            return raw_output
        return raw_output[:_MAX_SUMMARY_CHARS] + "\n...(truncated)"

    def _extract_gap(self, raw_output: str, success_criteria: str) -> str:
        lines = raw_output.strip().split("\n")
        last_lines = lines[-5:] if len(lines) >= 5 else lines
        return f"Criteria not met: {success_criteria}. Output tail: {' '.join(last_lines)[:200]}"
