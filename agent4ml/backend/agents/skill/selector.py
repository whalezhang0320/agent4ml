"""SkillSelector — quality filter + LLM select。混合注入的 agent 自动选半。

INVARIANT:
- override（用户 /skill 显式）强制包含，不受 quality filter 影响
- quality filter: 淘汰 effective_rate < threshold AND total_selections >= min
  （< min 的新 skill 不淘汰，给数据积累——anti-loop，INV#12）
- 候选 <= max_skills → 全返，跳过 LLM
- 候选 > max_skills 且 llm 提供 → LLM select ≤ max
- 候选 > max_skills 且无 llm → 按 effective_rate 降序取 top max（fallback）
- LLM select 失败 → fallback effective_rate 排序
- 无 store 时返空
"""
from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.messages import HumanMessage

from agent4ml.backend.agents.skill.types import SkillRecord

logger = logging.getLogger(__name__)


class SkillSelector:
    """skill 检索：override 强制 + quality filter + LLM select。"""

    def __init__(
        self,
        store: Any,
        llm: Any | None = None,
        max_skills: int = 3,
        quality_threshold: float = 0.3,
        min_selections: int = 5,
    ) -> None:
        self._store = store
        self._llm = llm
        self._max_skills = max_skills
        self._quality_threshold = quality_threshold
        self._min_selections = min_selections

    def select_for_task(
        self,
        task_description: str,
        overrides: list[str] | None = None,
    ) -> list[SkillRecord]:
        if self._store is None:
            return []
        # 1. override 强制包含（不受 filter 影响）
        forced: list[SkillRecord] = []
        for name in overrides or []:
            rec = self._store.get_active(name)
            if rec is not None and rec.enabled:
                forced.append(rec)
        # 2. active enabled + quality filter
        active = [r for r in self._store.list_active() if r.enabled]
        filtered = self._quality_filter(active)
        # 3. dedup combine（override 优先）
        seen: set[str] = {r.skill_id for r in forced}
        candidates = list(forced)
        for r in filtered:
            if r.skill_id not in seen:
                candidates.append(r)
                seen.add(r.skill_id)
        # 4. <= max → 全返，跳过 LLM
        if len(candidates) <= self._max_skills:
            return candidates
        # 5. > max + llm → LLM select
        if self._llm is not None:
            selected = self._llm_select(candidates, task_description, self._max_skills)
            if selected:
                return selected
        # 6. fallback: effective_rate 降序 top max
        ranked = sorted(candidates, key=lambda r: r.effective_rate, reverse=True)
        return ranked[: self._max_skills]

    def _quality_filter(self, skills: list[SkillRecord]) -> list[SkillRecord]:
        """淘汰 effective_rate < threshold AND selections >= min。新 skill 不淘汰。"""
        return [
            r for r in skills
            if not (r.total_selections >= self._min_selections
                    and r.effective_rate < self._quality_threshold)
        ]

    def _llm_select(
        self,
        candidates: list[SkillRecord],
        task: str,
        max_skills: int,
    ) -> list[SkillRecord]:
        """LLM 从候选选 ≤ max。失败返空（调用方 fallback）。"""
        catalog = self._build_catalog(candidates)
        prompt = (
            f"任务: {task}\n\n可用 skill 目录:\n{catalog}\n\n"
            f"选出最多 {max_skills} 个最相关的 skill。"
            '只返 JSON: {"skills": ["skill_id1", "skill_id2"]}'
        )
        try:
            resp = self._llm.invoke(
                [HumanMessage(content=prompt)],
                config={"tags": ["internal_llm"]},
            )
            content = resp.content if hasattr(resp, "content") else str(resp)
            data = json.loads(self._extract_json(content))
            ids = data.get("skills", [])
            id_to_rec = {r.skill_id: r for r in candidates}
            selected = [id_to_rec[i] for i in ids if i in id_to_rec]
            return selected[:max_skills]
        except Exception as exc:
            logger.warning("LLM skill select failed: %s", exc)
            return []

    def _build_catalog(self, candidates: list[SkillRecord]) -> str:
        lines = []
        for r in candidates:
            rate = f"{r.effective_rate:.0%}" if r.total_selections else "n/a"
            lines.append(f"- {r.skill_id}: {r.description} (成功率 {rate})")
        return "\n".join(lines)

    @staticmethod
    def _extract_json(text: str) -> str:
        """从可能含 markdown fence 的文本提取首个 JSON 对象。"""
        s = text.find("{")
        e = text.rfind("}")
        if s != -1 and e != -1 and e > s:
            return text[s : e + 1]
        return text
