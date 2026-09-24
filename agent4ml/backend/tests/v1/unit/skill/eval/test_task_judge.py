"""L3-E6 单测：TaskQualityJudge — 4 维加权评分 + 持久化 + 降级。

用 asyncio.run() 跑 async 方法。
"""
from __future__ import annotations

import asyncio
import json

from agent4ml.backend.agents.skill.eval.analyzers.task_quality_judge import (
    TaskQualityJudge,
)
from agent4ml.backend.agents.skill.eval.types import TaskQualityScore


class _FakeLLM:
    def __init__(self, content: str):
        self._content = content

    def invoke(self, messages):
        return type("R", (), {"content": self._content})()


class _FakeStore:
    def __init__(self):
        self.scores: list[TaskQualityScore] = []

    def save_task_score(self, score):
        self.scores.append(score)
        return score.score_id


_GOOD_RESPONSE = json.dumps({
    "task_completion": 0.9,
    "response_quality": 0.8,
    "efficiency": 0.7,
    "tool_usage": 0.8,
    "rationale": "good analysis",
})


def _run(coro):
    return asyncio.run(coro)


# ── 基本产出 ────────────────────────────────────────────

def test_produces_4_dim_score():
    judge = TaskQualityJudge(_FakeLLM(_GOOD_RESPONSE), _FakeStore())
    score = _run(judge.judge_task("t1", "trace", "output"))
    assert score is not None
    assert score.task_completion == 0.9
    assert score.response_quality == 0.8
    assert score.efficiency == 0.7
    assert score.tool_usage == 0.8


def test_weighted_overall_score():
    """overall = 0.50*0.9 + 0.35*0.8 + 0.05*0.7 + 0.10*0.8 = 0.845。"""
    judge = TaskQualityJudge(_FakeLLM(_GOOD_RESPONSE), _FakeStore())
    score = _run(judge.judge_task("t1", "trace", "output"))
    assert score is not None
    assert abs(score.overall_score - 0.845) < 0.01


def test_rationale_preserved():
    judge = TaskQualityJudge(_FakeLLM(_GOOD_RESPONSE), _FakeStore())
    score = _run(judge.judge_task("t1", "trace", "output"))
    assert score is not None
    assert score.rationale == "good analysis"


def test_score_id_generated():
    judge = TaskQualityJudge(_FakeLLM(_GOOD_RESPONSE), _FakeStore())
    score = _run(judge.judge_task("t1", "trace", "output"))
    assert score is not None
    assert score.score_id.startswith("score_")


# ── 持久化 ─────────────────────────────────────────────

def test_persists_to_store():
    store = _FakeStore()
    judge = TaskQualityJudge(_FakeLLM(_GOOD_RESPONSE), store)
    _run(judge.judge_task("t1", "trace", "output"))
    assert len(store.scores) == 1
    assert store.scores[0].task_id == "t1"


def test_store_none_does_not_crash():
    judge = TaskQualityJudge(_FakeLLM(_GOOD_RESPONSE), None)
    score = _run(judge.judge_task("t1", "trace", "output"))
    assert score is not None


# ── 降级 ───────────────────────────────────────────────

def test_llm_none_returns_none():
    judge = TaskQualityJudge(None, _FakeStore())
    score = _run(judge.judge_task("t1", "trace", "output"))
    assert score is None


def test_llm_exception_returns_none():
    class _ExplodingLLM:
        def invoke(self, messages):
            raise RuntimeError("LLM down")

    judge = TaskQualityJudge(_ExplodingLLM(), _FakeStore())
    score = _run(judge.judge_task("t1", "trace", "output"))
    assert score is None


def test_invalid_json_returns_none():
    judge = TaskQualityJudge(_FakeLLM("not json"), _FakeStore())
    score = _run(judge.judge_task("t1", "trace", "output"))
    assert score is None


def test_json_with_markdown_fences():
    content = '```json\n' + _GOOD_RESPONSE + '\n```'
    judge = TaskQualityJudge(_FakeLLM(content), _FakeStore())
    score = _run(judge.judge_task("t1", "trace", "output"))
    assert score is not None
    assert score.task_completion == 0.9


# ── 维度值 clamp ───────────────────────────────────────

def test_dims_clamped_to_01():
    """LLM 返超范围值 → clamp 到 [0, 1]。"""
    response = json.dumps({
        "task_completion": 1.5, "response_quality": -0.3,
        "efficiency": 0.5, "tool_usage": 0.5,
    })
    judge = TaskQualityJudge(_FakeLLM(response), _FakeStore())
    score = _run(judge.judge_task("t1", "trace", "output"))
    assert score is not None
    assert score.task_completion == 1.0
    assert score.response_quality == 0.0


def test_missing_dim_defaults_to_05():
    """LLM 漏返某维 → 默认 0.5。"""
    response = json.dumps({"task_completion": 0.9})
    judge = TaskQualityJudge(_FakeLLM(response), _FakeStore())
    score = _run(judge.judge_task("t1", "trace", "output"))
    assert score is not None
    assert score.response_quality == 0.5
    assert score.efficiency == 0.5
