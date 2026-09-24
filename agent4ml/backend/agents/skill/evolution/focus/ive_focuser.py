"""IVEFocuser — IVE 5 问失败聚焦（借鉴 EvoSkills IVE）。

5 问诊断区分 fundamental（skill 方向错）vs implementation（执行偏差）：
1. 失败发生在 skill 哪一步？
2. 是 skill 指令本身错（fundamental）还是执行偏差（implementation）？
3. 若 implementation，是工具调用失败 / 上下文不足 / 模型理解偏差？
4. 失败是否可经微调 skill 文本修复？
5. 修复方向是什么（不直接改，给 Mutator 输入）？

implementation failure 累计 impl_fail_threshold 次（默认 3）升级 fundamental
（防保守分类导致无限重试，借鉴 EvoSkills）。

LLM=None 降级为全量摘要（不崩，弱但可用）。
"""
from __future__ import annotations

from typing import Any

from agent4ml.backend.agents.skill.evolution.types import (
    EvolutionContext,
    FailureClass,
    FailureEvidence,
)


class IVEFocuser:
    """IVE 5 问诊断。LLM 调用产 failure_class + fix_direction。"""

    def __init__(
        self,
        llm: Any | None = None,
        impl_fail_threshold: int = 3,
    ) -> None:
        self._llm = llm
        self._impl_fail_threshold = impl_fail_threshold
        # skill_name → implementation 累计次数
        self._impl_fail_counts: dict[str, int] = {}

    def focus(self, ctx: EvolutionContext, store: Any) -> EvolutionContext:
        """聚焦失败证据，产 failure_class + fix_direction。

        D-L3-20: 读 SkillJudgment 历史，deviation_note 作 failure_evidence 补充。
        ctx.failure_evidence 非空时 LLM 5 问诊断；空时（如 CAPTURED）保留原 ctx。
        """
        if not ctx.failure_evidence:
            # CAPTURED 无失败证据，直接返原 ctx（CAPTURED 是沉淀非修复）
            return ctx

        # D-L3-20: 读 SkillJudgment 历史补充失败证据
        judgment_notes = self._read_judgment_notes(ctx, store)

        if self._llm is None:
            # 降级：全量摘要，默认 IMPLEMENTATION（保守，不轻易判 fundamental）
            return self._degrade_focus(ctx, judgment_notes)

        # LLM 5 问诊断
        skill_name = ctx.target_skill.name if ctx.target_skill else ctx.suggested_name
        failure_class, fix_direction = self._llm_diagnose(ctx, skill_name, judgment_notes)

        # implementation 累计升级 fundamental
        if failure_class == "IMPLEMENTATION":
            self._impl_fail_counts[skill_name] = self._impl_fail_counts.get(skill_name, 0) + 1
            if self._impl_fail_counts[skill_name] >= self._impl_fail_threshold:
                failure_class = "FUNDAMENTAL"
                fix_direction = (
                    f"[升级 fundamental] implementation 累计 "
                    f"{self._impl_fail_counts[skill_name]} 次，skill 方向可能有误。"
                    f" {fix_direction}"
                )
        elif failure_class == "FUNDAMENTAL":
            # fundamental 重置 implementation 计数
            self._impl_fail_counts[skill_name] = 0

        # 更新 failure_evidence 的 failure_class
        updated_evidence = tuple(
            FailureEvidence(
                turn_index=e.turn_index,
                tool_name=e.tool_name,
                failure_class=failure_class,
                description=e.description,
                impl_fail_count=self._impl_fail_counts.get(skill_name, 0),
            )
            for e in ctx.failure_evidence
        )
        from dataclasses import replace
        return replace(ctx, failure_evidence=updated_evidence, fix_direction=fix_direction)

    def _degrade_focus(
        self, ctx: EvolutionContext, judgment_notes: list[str] | None = None,
    ) -> EvolutionContext:
        """LLM=None 降级：全量摘要，默认 IMPLEMENTATION。"""
        summaries = [e.description for e in ctx.failure_evidence]
        parts = ["失败证据：" + " | ".join(summaries)]
        if judgment_notes:
            parts.append("SkillJudgment 偏差：" + " | ".join(judgment_notes))
        fix_direction = "LLM 未启用，降级全量摘要。" + " ".join(parts)
        from dataclasses import replace
        return replace(ctx, fix_direction=fix_direction)

    @staticmethod
    def _read_judgment_notes(ctx: EvolutionContext, store: Any) -> list[str]:
        """D-L3-20: 从 store 读 SkillJudgment 历史，返 deviation_note 列表。"""
        if store is None or ctx.target_skill is None:
            return []
        try:
            judgments = store.get_judgments(ctx.target_skill.skill_id, limit=10)
            return [j.deviation_note for j in judgments if j.deviation_note]
        except Exception:
            return []

    def _llm_diagnose(
        self, ctx: EvolutionContext, skill_name: str,
        judgment_notes: list[str] | None = None,
    ) -> tuple[FailureClass, str]:
        """LLM 5 问诊断。返 (failure_class, fix_direction)。

        实现：调 LLM 读失败证据 + skill 描述，返 JSON {"class": "...", "direction": "..."}。
        失败时保守返 IMPLEMENTATION（不轻易判 fundamental 杀掉 skill）。
        """
        try:
            import json
            from langchain_core.messages import HumanMessage

            skill_desc = ctx.target_skill.description if ctx.target_skill else ctx.capture_pattern
            evidence_text = "\n".join(
                f"- turn={e.turn_index} tool={e.tool_name}: {e.description}"
                for e in ctx.failure_evidence
            )
            judgment_text = ""
            if judgment_notes:
                judgment_text = "\n\nSkillJudgment 偏差记录:\n" + "\n".join(
                    f"- {n}" for n in judgment_notes
                )
            prompt = (
                f"Skill: {skill_name}\n描述: {skill_desc}\n\n"
                f"失败证据:\n{evidence_text}{judgment_text}\n\n"
                f"IVE 5 问诊断：\n"
                f"1. 失败发生在 skill 哪一步？\n"
                f"2. 是 skill 指令本身错（FUNDAMENTAL）还是执行偏差（IMPLEMENTATION）？\n"
                f"3. 若 implementation，子类（工具失败/上下文不足/模型理解偏差）？\n"
                f"4. 可否微调 skill 文本修复？\n"
                f"5. 修复方向？\n\n"
                f'只返 JSON: {{"class": "FUNDAMENTAL"|"IMPLEMENTATION", "direction": "修复方向"}}'
            )
            resp = self._llm.invoke([HumanMessage(content=prompt)])
            content = resp.content if hasattr(resp, "content") else str(resp)
            # 提取 JSON
            s = content.find("{")
            e_idx = content.rfind("}")
            if s != -1 and e_idx != -1 and e_idx > s:
                data = json.loads(content[s : e_idx + 1])
                cls = data.get("class", "IMPLEMENTATION")
                if cls not in ("FUNDAMENTAL", "IMPLEMENTATION"):
                    cls = "IMPLEMENTATION"
                return cls, data.get("direction", "")
            return "IMPLEMENTATION", ""
        except Exception:
            # LLM 失败保守返 IMPLEMENTATION
            return "IMPLEMENTATION", ""
