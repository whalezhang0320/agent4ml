"""TaskQualityJudge — 任务层 eval（SkillClaw session_judge 模型）。

D-L3-5/D-L3-13: LLM 4 维加权评分（task_completion:0.50 / response_quality:0.35 /
efficiency:0.05 / tool_usage:0.10），产 TaskQualityScore。

post-execution 异步调用（D-L3-19）。LLM=None/异常 → 返 None（graceful degradation）。
"""
from __future__ import annotations

import json
import uuid
from typing import Any

from agent4ml.backend.agents.journal.events import utc_now_iso
from agent4ml.backend.agents.skill.eval.types import TaskQualityScore

_WEIGHTS = {
    "task_completion": 0.50,
    "response_quality": 0.35,
    "efficiency": 0.05,
    "tool_usage": 0.10,
}

_MAX_TRACE_CHARS = 80000
_MAX_OUTPUT_CHARS = 20000


class TaskQualityJudge:
    """任务层 eval：LLM 4 维加权评分。"""

    def __init__(self, llm: Any | None = None, store: Any = None) -> None:
        self._llm = llm
        self._store = store

    async def judge_task(
        self,
        task_id: str,
        execution_trace: str,
        final_output: str,
    ) -> TaskQualityScore | None:
        """LLM 4 维评分，返 TaskQualityScore。LLM=None/异常 → None。"""
        if self._llm is None:
            return None

        try:
            score = self._llm_judge(task_id, execution_trace, final_output)
        except Exception:
            return None

        if score is not None and self._store is not None:
            try:
                self._store.save_task_score(score)
            except Exception:
                pass
        return score

    def _llm_judge(
        self, task_id: str, execution_trace: str, final_output: str,
    ) -> TaskQualityScore | None:
        """调 LLM 4 维评分，解析 JSON 返 TaskQualityScore。"""
        from langchain_core.messages import HumanMessage

        trace = (execution_trace or "")[:_MAX_TRACE_CHARS]
        output = (final_output or "")[:_MAX_OUTPUT_CHARS]

        prompt = (
            "你在评估一个 agent 任务的执行质量。\n\n"
            f"执行 trace:\n{trace}\n\n"
            f"最终输出:\n{output}\n\n"
            "4 维评分（0.0-1.0）：\n"
            "- task_completion: 用户目标是否完成\n"
            "- response_quality: 最终输出的正确性/完整性/清晰度\n"
            "- efficiency: 是否避免不必要的重试/绕路\n"
            "- tool_usage: 工具使用是否恰当有效\n\n"
            '只返 JSON: {"task_completion": 0.9, "response_quality": 0.8, '
            '"efficiency": 0.7, "tool_usage": 0.8, "rationale": "简短解释"}'
        )
        resp = self._llm.invoke([HumanMessage(content=prompt)])
        content = resp.content if hasattr(resp, "content") else str(resp)

        data = self._extract_json(content)
        if data is None:
            return None

        dims = {
            k: max(0.0, min(1.0, float(data.get(k, 0.5))))
            for k in _WEIGHTS
        }
        overall = sum(dims[k] * w for k, w in _WEIGHTS.items())

        return TaskQualityScore(
            score_id=f"score_{uuid.uuid4().hex[:12]}",
            task_id=task_id,
            task_completion=dims["task_completion"],
            response_quality=dims["response_quality"],
            efficiency=dims["efficiency"],
            tool_usage=dims["tool_usage"],
            overall_score=round(overall, 3),
            rationale=data.get("rationale", ""),
            timestamp=utc_now_iso(),
        )

    @staticmethod
    def _extract_json(content: str) -> dict | None:
        s = content.find("{")
        e = content.rfind("}")
        if s != -1 and e != -1 and e > s:
            try:
                return json.loads(content[s : e + 1])
            except Exception:
                return None
        return None
