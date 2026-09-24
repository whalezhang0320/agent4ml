"""PiResultSummarizer — extends base with three-section parsing (What You Did / Success / Gaps).

设计（design_docs/46 §10.3.6）:
- 继承 BaseResultSummarizer 通用 programmatic eval floor
- Pi-specific：解析 pi 输出的三段（_build_prompt 要求的 What You Did / Success / Gaps）
- 不传回 raw output 全量（控制 lead agent context 膨胀）
"""
from __future__ import annotations

import re

from agent4ml.backend.agents.multiagent.summarizers.result.base import (
    BaseResultSummarizer,
)
from agent4ml.backend.agents.multiagent.types import ArtifactRef


# 三段输出格式正则（与 PiRuntime._build_prompt 的 Output Format 一致）
_WHAT_YOU_DID_RE = re.compile(
    r"##\s*What You Did\s*\n(.*?)(?=##\s*Success|##\s*Gaps|$)",
    re.DOTALL | re.IGNORECASE,
)
_SUCCESS_SECTION_RE = re.compile(
    r"##\s*Success\s*\n(.*?)(?=##\s*Gaps|$)",
    re.DOTALL | re.IGNORECASE,
)
_GAPS_SECTION_RE = re.compile(
    r"##\s*Gaps\s*\n(.*?)$",
    re.DOTALL | re.IGNORECASE,
)
# success 判定关键词（yes / pass / met）
_SUCCESS_POSITIVE_RE = re.compile(r"\b(yes|pass|met|complete)\b", re.IGNORECASE)
# success 判定关键词（no / fail / not met / partial）
_SUCCESS_NEGATIVE_RE = re.compile(r"\b(no|fail|not met|partial|incomplete)\b", re.IGNORECASE)


class PiResultSummarizer(BaseResultSummarizer):
    """Pi specialist 输出端转换器（三段解析 + programmatic eval floor）。

    继承 BaseResultSummarizer 通用校验（产物存在性 + success_criteria 回应 + 无敏感改动）。
    Pi-specific 扩展：解析 _build_prompt 要求的三段输出格式。
    """

    def __init__(self) -> None:
        super().__init__(specialist_name="pi")

    def summarize(
        self,
        raw_output: str,
        artifacts: list[ArtifactRef],
        goal: str,
        success_criteria: str,
    ) -> "SpecialistResult":  # type: ignore[name-defined]
        """压缩 raw output + 校验 success_criteria + 解析三段。"""
        # 调 base 通用校验（产物存在性 + success_criteria 回应 + 无敏感改动）
        base_success = self._evaluate_success(raw_output, artifacts, success_criteria)

        # 解析 pi 输出的三段
        sections = self._parse_pi_sections(raw_output)

        # pi-specific success 判定：base 校验通过 + Success 段无负面关键词
        pi_success = self._evaluate_pi_success(sections, base_success)

        # gap_analysis：优先用 Gaps 段，否则用 base 提取
        gap = sections.get("gaps", "") or (
            "" if pi_success else self._extract_gap(raw_output, success_criteria)
        )

        # summary：优先用 What You Did 段，否则用 base 压缩
        summary = sections.get("did", "") or self._compress(raw_output)

        from agent4ml.backend.agents.multiagent.types import SpecialistResult

        return SpecialistResult(
            specialist_name=self._specialist_name,
            summary=summary,
            artifacts=tuple(artifacts),
            success=pi_success,
            gap_analysis=gap,
        )

    def _parse_pi_sections(self, output: str) -> dict[str, str]:
        """解析 pi 输出的 What You Did / Success / Gaps 三段。

        pi 按 _build_prompt 要求的格式输出，此方法提取三段内容。
        缺失段返空串。
        """
        sections: dict[str, str] = {}

        did_match = _WHAT_YOU_DID_RE.search(output)
        if did_match:
            sections["did"] = did_match.group(1).strip()

        success_match = _SUCCESS_SECTION_RE.search(output)
        if success_match:
            sections["success"] = success_match.group(1).strip()

        gaps_match = _GAPS_SECTION_RE.search(output)
        if gaps_match:
            sections["gaps"] = gaps_match.group(1).strip()

        return sections

    def _evaluate_pi_success(
        self, sections: dict[str, str], base_success: bool
    ) -> bool:
        """pi-specific success 判定。

        判定逻辑：
        1. base 通用校验必须通过（产物存在 + success_criteria 回应 + 无敏感改动）
        2. Success 段无负面关键词（no / fail / not met / partial / incomplete）
        3. 如果 Success 段有正面关键词（yes / pass / met / complete），加强信心
        """
        if not base_success:
            return False

        success_section = sections.get("success", "").lower()
        if not success_section:
            # 无 Success 段，依赖 base 判定
            return base_success

        # 有负面关键词 → 失败
        if _SUCCESS_NEGATIVE_RE.search(success_section):
            return False

        # 无负面关键词 → 通过（base 已校验）
        return True
