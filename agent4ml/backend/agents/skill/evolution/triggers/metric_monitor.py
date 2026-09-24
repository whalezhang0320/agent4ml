"""MetricMonitorTrigger — 周期 metric 扫描触发（借鉴 OpenSpace Trigger 3）。

两阶段筛选：
- Phase 1 规则（_diagnose_skill_health，阈值宽松）：
  - fallback_rate > 0.4 → FIX（常选不用，指令不清）
  - applied_rate > 0.4 且 completion_rate < 0.35 → FIX（用了没完成，指令错）
  - effective_rate < 0.55 且 applied_rate > 0.25 → DERIVED（中等效果，2b）
- Phase 2 LLM 确认（llm 非 None 时调，过滤误报；None 跳过）

anti-loop：
- min_selections：total_selections < min 不触发（新进化 skill selections=0）
- cooldown：自上次进化后需 cooldown_turns 次新 selections 才重评（数据驱动，无时间计数）
"""
from __future__ import annotations

from typing import Any

from agent4ml.backend.agents.skill.evolution.types import EvolutionContext
from agent4ml.backend.agents.skill.types import SkillRecord

# 阈值（借鉴 OpenSpace，宽松——LLM 确认过滤误报）
_FALLBACK_THRESHOLD = 0.4
_LOW_COMPLETION_THRESHOLD = 0.35
_HIGH_APPLIED_FOR_FIX = 0.4
_MODERATE_EFFECTIVE_THRESHOLD = 0.55
_MIN_APPLIED_FOR_DERIVED = 0.25


class MetricMonitorTrigger:
    """周期扫 metrics：effective_rate < threshold AND selections >= min AND cooldown 过。

    产出 FIX 类型 EvolutionContext（2a；DERIVED 留 2b）。
    """

    def __init__(
        self,
        threshold: float = 0.3,
        min_selections: int = 5,
        cooldown_turns: int = 10,
        llm: Any | None = None,
    ) -> None:
        self._threshold = threshold
        self._min_selections = min_selections
        self._cooldown_turns = cooldown_turns
        self._llm = llm
        # anti-loop：skill_name → 上次进化时的 total_selections
        self._last_evolve_selections: dict[str, int] = {}

    def should_trigger(self, store: Any) -> list[EvolutionContext]:
        """扫所有 active skill，产出需进化的 FIX context 列表。"""
        results: list[EvolutionContext] = []
        for rec in store.list_active():
            if not rec.enabled:
                continue
            # anti-loop: 新进化 skill selections=0 不触发
            if rec.total_selections < self._min_selections:
                continue
            # anti-loop: cooldown——自上次进化需 cooldown_turns 次新 selections
            last = self._last_evolve_selections.get(rec.name, 0)
            if rec.total_selections - last < self._cooldown_turns:
                continue

            evo_type, direction = self._diagnose_skill_health(rec)
            if evo_type is None:
                continue
            # 2a 只产 FIX；DERIVED 留 2b
            if evo_type != "FIX":
                continue

            # Phase 2 LLM 确认（llm None 跳过，纯规则）
            if self._llm is not None and not self._llm_confirm_evolution(rec, direction):
                continue

            results.append(EvolutionContext(
                trigger="METRIC",
                evolution_type="FIX",
                target_skill=rec,
                fix_direction=direction,
            ))
        return results

    def mark_evolved(self, skill_name: str, total_selections: int) -> None:
        """EvolutionManager 进化完成后调，记录 anti-loop 锚点。"""
        self._last_evolve_selections[skill_name] = total_selections

    @staticmethod
    def _diagnose_skill_health(
        record: SkillRecord,
    ) -> tuple[str | None, str]:
        """规则诊断。返 (evolution_type, direction)；健康返 (None, "")。

        阈值宽松——LLM 确认步过滤误报。
        """
        # 高 fallback → 常选不用 → FIX
        if record.fallback_rate > _FALLBACK_THRESHOLD:
            return "FIX", (
                f"高 fallback_rate({record.fallback_rate:.0%})：skill 常被选但未应用，"
                f"指令不清或过时。"
            )
        # 用了但没完成 → 指令错 → FIX
        if (record.applied_rate > _HIGH_APPLIED_FOR_FIX
                and record.completion_rate < _LOW_COMPLETION_THRESHOLD):
            return "FIX", (
                f"低 completion_rate({record.completion_rate:.0%}) 但高 applied_rate"
                f"({record.applied_rate:.0%})：skill 指令可能错或不全。"
            )
        # 中等效果 → DERIVED（2b）
        if (record.effective_rate < _MODERATE_EFFECTIVE_THRESHOLD
                and record.applied_rate > _MIN_APPLIED_FOR_DERIVED):
            return "DERIVED", (
                f"中等 effective_rate({record.effective_rate:.0%})：可派生增强版。"
            )
        return None, ""

    def _llm_confirm_evolution(self, record: SkillRecord, direction: str) -> bool:
        """Phase 2 LLM 确认。问 LLM 这 skill 真需进化吗。

        MVP：llm.invoke 返 "yes"/"no" 文本。实现时核对 OpenSpace _llm_confirm_evolution prompt。
        """
        try:
            prompt = (
                f"Skill: {record.name}\n描述: {record.description}\n"
                f"诊断: {direction}\n"
                f"metrics: selections={record.total_selections}, "
                f"effective_rate={record.effective_rate:.0%}, "
                f"fallback_rate={record.fallback_rate:.0%}\n"
                f"这 skill 真需要进化吗？只返 yes 或 no。"
            )
            from langchain_core.messages import HumanMessage
            resp = self._llm.invoke([HumanMessage(content=prompt)])
            content = resp.content if hasattr(resp, "content") else str(resp)
            return "yes" in content.lower()
        except Exception:
            return False  # LLM 失败保守不触发
