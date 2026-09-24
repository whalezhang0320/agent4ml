"""SkillJudgmentAnalyzer — 执行层 eval（OpenSpace ExecutionAnalyzer 模型）。

D-L3-21: 一次 LLM 调用同时产 SkillJudgment + EvolutionSuggestion。
D-L3-3: 更新 L1 4 计数器（analyzer 内调 store.record_outcome）。
D-L3-19: 异步 fire-and-forget（caller 负责 asyncio.create_task）。
D-L3-14: 无 skill 注入时返空（caller 负责跳过）。

LLM=None 或异常 → 返空列表（graceful degradation，不崩）。
"""
from __future__ import annotations

import json
import uuid
from typing import Any

from agent4ml.backend.agents.journal.events import utc_now_iso
from agent4ml.backend.agents.skill.eval.types import (
    EvolutionSuggestion,
    SkillJudgment,
)

_MAX_MESSAGES_CHARS = 80000
_MAX_JOURNAL_EVENTS = 50


class SkillJudgmentAnalyzer:
    """执行层 eval：post-execution LLM 分析，产 per-skill SkillJudgment + EvolutionSuggestion。"""

    def __init__(self, llm: Any | None = None, store: Any = None) -> None:
        self._llm = llm
        self._store = store

    async def analyze_execution(
        self,
        task_id: str,
        journal_events: list[dict],
        messages_summary: str,
        injected_skills: list[dict],
        task_completed: bool = True,
    ) -> tuple[list[SkillJudgment], list[EvolutionSuggestion]]:
        """LLM 分析执行日志，产 SkillJudgment + EvolutionSuggestion。

        返 (judgments, suggestions)。LLM=None/异常 → 返空。
        持久化 judgment + 更新 L1 计数器 + journal 事件。
        """
        if not injected_skills or self._llm is None:
            return [], []

        try:
            judgments, suggestions = self._llm_analyze(
                task_id, journal_events, messages_summary, injected_skills,
            )
        except Exception:
            return [], []

        self._persist(task_id, judgments, task_completed)
        return judgments, suggestions

    def _llm_analyze(
        self,
        task_id: str,
        journal_events: list[dict],
        messages_summary: str,
        injected_skills: list[dict],
    ) -> tuple[list[SkillJudgment], list[EvolutionSuggestion]]:
        """调 LLM 分析，解析 JSON 返 judgments + suggestions。"""
        from langchain_core.messages import HumanMessage

        skills_text = "\n".join(
            f"{i+1}. {s['skill_id']}: {s['name']} — {s.get('description', '')}"
            for i, s in enumerate(injected_skills)
        )
        events_text = self._format_events(journal_events)
        summary = (messages_summary or "")[:_MAX_MESSAGES_CHARS]

        prompt = (
            "你在分析一个已完成的 agent 任务执行。\n\n"
            f"被注入的 skills:\n{skills_text}\n\n"
            f"执行摘要:\n{summary}\n\n"
            f"journal 事件:\n{events_text}\n\n"
            "对每个被注入的 skill 判断：\n"
            "1. agent 是否实际应用了该 skill 的指导？（true/false）\n"
            "2. 有什么偏差？（简短记录）\n\n"
            "如有进化建议：\n"
            "- type: FIX（修复现有 skill）/ DERIVED（派生增强）/ CAPTURED（捕获新模式）\n"
            "- target: skill_id 列表\n"
            "- direction: 该修什么/该捕获什么\n\n"
            '只返 JSON: {"judgments": [{"skill_id": "...", "skill_applied": true, '
            '"deviation_note": "..."}], "suggestions": [{"evolution_type": "FIX", '
            '"target_skill_ids": ["..."], "direction": "..."}]}'
        )
        resp = self._llm.invoke([HumanMessage(content=prompt)])
        content = resp.content if hasattr(resp, "content") else str(resp)

        data = self._extract_json(content)
        if data is None:
            return [], []

        skill_map = {s["skill_id"]: s for s in injected_skills}
        judgments: list[SkillJudgment] = []
        for j_data in data.get("judgments", []):
            sid = j_data.get("skill_id", "")
            if sid not in skill_map:
                continue
            judgments.append(SkillJudgment(
                judgment_id=f"judgment_{uuid.uuid4().hex[:12]}",
                skill_id=sid,
                skill_name=skill_map[sid].get("name", ""),
                task_id=task_id,
                skill_applied=bool(j_data.get("skill_applied", False)),
                deviation_note=j_data.get("deviation_note", ""),
                timestamp=utc_now_iso(),
            ))

        suggestions: list[EvolutionSuggestion] = []
        for s_data in data.get("suggestions", []):
            etype = s_data.get("evolution_type", "FIX")
            if etype not in ("FIX", "DERIVED", "CAPTURED"):
                etype = "FIX"
            suggestions.append(EvolutionSuggestion(
                evolution_type=etype,  # type: ignore[arg-type]
                target_skill_ids=tuple(s_data.get("target_skill_ids", [])),
                direction=s_data.get("direction", ""),
            ))

        return judgments, suggestions

    def _persist(
        self, task_id: str, judgments: list[SkillJudgment], task_completed: bool,
    ) -> None:
        """持久化 judgment + 更新 L1 计数器 + journal。"""
        if self._store is None:
            return
        for j in judgments:
            try:
                self._store.save_judgment(j)
                self._store.record_outcome(
                    j.skill_id, run_id=task_id,
                    applied=j.skill_applied,
                    task_completed=task_completed,
                    note=j.deviation_note,
                )
            except Exception:
                pass  # silent degradation

    @staticmethod
    def _format_events(events: list[dict]) -> str:
        lines = []
        for e in (events or [])[:_MAX_JOURNAL_EVENTS]:
            etype = e.get("event_type", e.get("type", ""))
            lines.append(f"- {etype}: {json.dumps(e, ensure_ascii=False)[:200]}")
        return "\n".join(lines) or "(无事件)"

    @staticmethod
    def _extract_json(content: str) -> dict | None:
        s = content.find("{")
        e_idx = content.rfind("}")
        if s != -1 and e_idx != -1 and e_idx > s:
            try:
                return json.loads(content[s : e_idx + 1])
            except Exception:
                return None
        return None
