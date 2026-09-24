"""L3-E5 单测：SkillJudgmentAnalyzer — 产 SkillJudgment + EvolutionSuggestion + 更新计数器。

用 asyncio.run() 跑 async 方法（无 pytest-asyncio 依赖）。
"""
from __future__ import annotations

import asyncio
import json

from agent4ml.backend.agents.skill.eval.analyzers.skill_judgment_analyzer import (
    SkillJudgmentAnalyzer,
)
from agent4ml.backend.agents.skill.eval.types import SkillJudgment


class _FakeLLM:
    def __init__(self, content: str):
        self._content = content

    def invoke(self, messages):
        return type("R", (), {"content": self._content})()


class _FakeStore:
    def __init__(self):
        self.judgments: list[SkillJudgment] = []
        self.outcomes: list[dict] = []

    def save_judgment(self, judgment):
        self.judgments.append(judgment)
        return judgment.judgment_id

    def record_outcome(self, skill_id, run_id, applied, task_completed, note=""):
        self.outcomes.append({
            "skill_id": skill_id, "run_id": run_id,
            "applied": applied, "task_completed": task_completed, "note": note,
        })


_SKILLS = [
    {"skill_id": "sv__imp", "name": "source-verification", "description": "验证信源"},
    {"skill_id": "qd__imp", "name": "question-decomposition", "description": "分解问题"},
]

_GOOD_RESPONSE = json.dumps({
    "judgments": [
        {"skill_id": "sv__imp", "skill_applied": True, "deviation_note": "验证了 2/3 URL"},
        {"skill_id": "qd__imp", "skill_applied": False, "deviation_note": "agent 忽略了分解步骤"},
    ],
    "suggestions": [
        {"evolution_type": "FIX", "target_skill_ids": ["qd__imp"], "direction": "加强 MUST 强制分解"},
    ],
})


def _run(coro):
    return asyncio.run(coro)


# ── 基本产出 ────────────────────────────────────────────

def test_produces_judgments_and_suggestions():
    analyzer = SkillJudgmentAnalyzer(_FakeLLM(_GOOD_RESPONSE), _FakeStore())
    judgments, suggestions = _run(analyzer.analyze_execution(
        "t1", [{"type": "skill.select"}], "agent 执行了任务", _SKILLS,
    ))
    assert len(judgments) == 2
    assert judgments[0].skill_id == "sv__imp"
    assert judgments[0].skill_applied is True
    assert judgments[0].deviation_note == "验证了 2/3 URL"
    assert judgments[1].skill_applied is False
    assert len(suggestions) == 1
    assert suggestions[0].evolution_type == "FIX"
    assert "qd__imp" in suggestions[0].target_skill_ids


def test_persists_judgments_and_updates_counters():
    store = _FakeStore()
    analyzer = SkillJudgmentAnalyzer(_FakeLLM(_GOOD_RESPONSE), store)
    _run(analyzer.analyze_execution("t1", [], "summary", _SKILLS, task_completed=True))
    assert len(store.judgments) == 2
    assert len(store.outcomes) == 2
    assert store.outcomes[0]["skill_id"] == "sv__imp"
    assert store.outcomes[0]["applied"] is True
    assert store.outcomes[0]["task_completed"] is True
    assert store.outcomes[1]["applied"] is False


def test_task_completed_false_propagates():
    store = _FakeStore()
    analyzer = SkillJudgmentAnalyzer(_FakeLLM(_GOOD_RESPONSE), store)
    _run(analyzer.analyze_execution("t1", [], "summary", _SKILLS, task_completed=False))
    assert all(o["task_completed"] is False for o in store.outcomes)


# ── 降级 ───────────────────────────────────────────────

def test_llm_none_returns_empty():
    analyzer = SkillJudgmentAnalyzer(None, _FakeStore())
    judgments, suggestions = _run(analyzer.analyze_execution("t1", [], "s", _SKILLS))
    assert judgments == []
    assert suggestions == []


def test_no_injected_skills_returns_empty():
    analyzer = SkillJudgmentAnalyzer(_FakeLLM(_GOOD_RESPONSE), _FakeStore())
    judgments, suggestions = _run(analyzer.analyze_execution("t1", [], "s", []))
    assert judgments == []
    assert suggestions == []


def test_llm_exception_returns_empty():
    class _ExplodingLLM:
        def invoke(self, messages):
            raise RuntimeError("LLM down")

    analyzer = SkillJudgmentAnalyzer(_ExplodingLLM(), _FakeStore())
    judgments, suggestions = _run(analyzer.analyze_execution("t1", [], "s", _SKILLS))
    assert judgments == []
    assert suggestions == []


# ── JSON 解析 ──────────────────────────────────────────

def test_invalid_json_returns_empty():
    analyzer = SkillJudgmentAnalyzer(_FakeLLM("not json"), _FakeStore())
    judgments, _ = _run(analyzer.analyze_execution("t1", [], "s", _SKILLS))
    assert judgments == []


def test_json_with_markdown_fences():
    content = '```json\n' + _GOOD_RESPONSE + '\n```'
    analyzer = SkillJudgmentAnalyzer(_FakeLLM(content), _FakeStore())
    judgments, _ = _run(analyzer.analyze_execution("t1", [], "s", _SKILLS))
    assert len(judgments) == 2


def test_unknown_skill_id_filtered():
    response = json.dumps({
        "judgments": [
            {"skill_id": "sv__imp", "skill_applied": True, "deviation_note": ""},
            {"skill_id": "unknown__id", "skill_applied": True, "deviation_note": ""},
        ],
        "suggestions": [],
    })
    analyzer = SkillJudgmentAnalyzer(_FakeLLM(response), _FakeStore())
    judgments, _ = _run(analyzer.analyze_execution("t1", [], "s", _SKILLS))
    assert len(judgments) == 1
    assert judgments[0].skill_id == "sv__imp"


def test_empty_suggestions_ok():
    response = json.dumps({
        "judgments": [{"skill_id": "sv__imp", "skill_applied": True, "deviation_note": ""}],
        "suggestions": [],
    })
    analyzer = SkillJudgmentAnalyzer(_FakeLLM(response), _FakeStore())
    judgments, suggestions = _run(analyzer.analyze_execution("t1", [], "s", _SKILLS))
    assert len(judgments) == 1
    assert suggestions == []


def test_invalid_evolution_type_defaults_fix():
    response = json.dumps({
        "judgments": [],
        "suggestions": [{"evolution_type": "INVALID", "target_skill_ids": [], "direction": "x"}],
    })
    analyzer = SkillJudgmentAnalyzer(_FakeLLM(response), _FakeStore())
    _, suggestions = _run(analyzer.analyze_execution("t1", [], "s", _SKILLS))
    assert suggestions[0].evolution_type == "FIX"


def test_store_none_does_not_crash():
    analyzer = SkillJudgmentAnalyzer(_FakeLLM(_GOOD_RESPONSE), None)
    judgments, suggestions = _run(analyzer.analyze_execution("t1", [], "s", _SKILLS))
    assert len(judgments) == 2
    assert len(suggestions) == 1
