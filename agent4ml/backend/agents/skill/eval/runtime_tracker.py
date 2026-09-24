"""RuntimeTracker — 趋势层 eval（OpenSpace quality metrics 模型）。

D-L3-16: after_agent 写数据 + 命令时算趋势 + GitRatchet 按需调 degraded_skills。
从 L1 4 计数器 + SkillJudgment 历史算趋势，产 SkillHealthReport + 退化检测。

趋势判定：将近 N 条 SkillJudgment 分前后两半，比较 applied_rate 差值。
- delta > degradation_delta → improving
- delta < -degradation_delta → degrading
- 否则 → stable
- 数据不足 → insufficient_data
"""
from __future__ import annotations

from typing import Any

from agent4ml.backend.agents.skill.eval.types import SkillHealthReport, SkillJudgment, Trend

_MIN_JUDGMENTS_FOR_TREND = 4


class RuntimeTracker:
    """趋势层 eval：从历史数据产出健康报告 + 退化检测。"""

    def __init__(self, store: Any, degradation_delta: float = 0.15) -> None:
        self._store = store
        self._degradation_delta = degradation_delta

    def health_report(self, skill_id: str, window: int = 20) -> SkillHealthReport:
        """产 skill 健康报告。读 L1 计数器 + 近 N 条 SkillJudgment。"""
        metrics = self._store.get_metrics(skill_id)
        judgments = self._store.get_judgments(skill_id, window)

        if metrics is None:
            return SkillHealthReport(
                skill_id=skill_id, skill_name="",
                window_selections=0,
                applied_rate=0.0, completion_rate=0.0,
                effective_rate=0.0, fallback_rate=0.0,
                trend="insufficient_data",
            )

        trend = self._compute_trend(judgments, self._degradation_delta)
        advice = self._build_advice(metrics, trend, judgments)

        return SkillHealthReport(
            skill_id=skill_id,
            skill_name=self._get_skill_name(skill_id),
            window_selections=metrics.selections,
            applied_rate=metrics.applied_rate,
            completion_rate=metrics.completion_rate,
            effective_rate=metrics.effective_rate,
            fallback_rate=metrics.fallback_rate,
            trend=trend,
            recent_judgments=tuple(judgments[:5]),
            advice=advice,
        )

    def degraded_skills(self, threshold: float = 0.15) -> list[str]:
        """返退化 skill_id 列表（trend=degrading）。给 GitRatchet 回滚信号。"""
        active = self._store.list_active()
        degraded: list[str] = []
        for skill in active:
            judgments = self._store.get_judgments(skill.skill_id, 20)
            trend = self._compute_trend(judgments, threshold)
            if trend == "degrading":
                degraded.append(skill.skill_id)
        return degraded

    @staticmethod
    def _compute_trend(
        judgments: list[SkillJudgment], delta: float,
    ) -> Trend:
        """从 SkillJudgment 历史算趋势。judgments 按 timestamp DESC（newest first）。"""
        if len(judgments) < _MIN_JUDGMENTS_FOR_TREND:
            return "insufficient_data"

        mid = len(judgments) // 2
        # judgments[0:mid] = recent, judgments[mid:] = older
        recent = judgments[:mid]
        older = judgments[mid:]

        recent_rate = sum(1 for j in recent if j.skill_applied) / len(recent)
        older_rate = sum(1 for j in older if j.skill_applied) / len(older)

        diff = recent_rate - older_rate
        if diff > delta:
            return "improving"
        if diff < -delta:
            return "degrading"
        return "stable"

    @staticmethod
    def _build_advice(metrics: Any, trend: Trend, judgments: list[SkillJudgment]) -> str:
        """产文字建议。metric-based advice 始终检查（即使 insufficient_data）。"""
        parts: list[str] = []
        if trend == "insufficient_data":
            parts.append("数据不足，需更多任务积累")
        if metrics.fallback_rate > 0.4:
            parts.append("fallback_rate 高，skill 触发条件可能有问题")
        if metrics.applied_rate < 0.3 and metrics.selections > 5:
            parts.append("applied_rate 低，skill 被选中但未被应用")
        if metrics.completion_rate < 0.35 and metrics.applied > 3:
            parts.append("completion_rate 低，skill 指导可能无效")
        if trend == "degrading":
            parts.append("趋势退化，建议 GitRatchet 回滚或进化修复")
        if trend == "improving":
            parts.append("趋势改善，进化效果良好")
        return "；".join(parts) if parts else "健康状态正常"

    def _get_skill_name(self, skill_id: str) -> str:
        """从 store 查 skill name。失败返空。"""
        try:
            rec = self._store.get(skill_id)
            return rec.name if rec else ""
        except Exception:
            return ""
