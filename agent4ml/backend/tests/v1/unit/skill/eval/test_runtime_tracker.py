"""L3-E7 单测：RuntimeTracker — 趋势判定 + 退化检测 + advice。

验证：
- health_report 从 L1 计数器 + SkillJudgment 产出
- 趋势判定（improving/stable/degrading/insufficient_data）
- degraded_skills 返退化列表
- advice 产出
- skill 不存在 → insufficient_data
"""
from __future__ import annotations

from agent4ml.backend.agents.skill.eval.runtime_tracker import RuntimeTracker
from agent4ml.backend.agents.skill.eval.types import SkillJudgment
from agent4ml.backend.agents.skill.types import SkillLineage, SkillMetrics, SkillRecord


class _FakeStore:
    """Mock store with metrics + judgments + list_active + get。"""

    def __init__(self):
        self._records: dict[str, SkillRecord] = {}
        self._metrics: dict[str, SkillMetrics] = {}
        self._judgments: dict[str, list[SkillJudgment]] = {}

    def add_skill(self, skill_id: str, name: str = "test",
                  sel=10, app=6, comp=4, fb=2):
        self._records[skill_id] = SkillRecord(
            skill_id=skill_id, name=name, path="/p", content_hash="h",
            lineage=SkillLineage(origin="IMPORTED"),
            total_selections=sel, total_applied=app,
            total_completions=comp, total_fallbacks=fb,
        )
        self._metrics[skill_id] = SkillMetrics(
            skill_id=skill_id, selections=sel, applied=app,
            completions=comp, fallbacks=fb,
            applied_rate=app / sel if sel else 0.0,
            completion_rate=comp / app if app else 0.0,
            effective_rate=comp / sel if sel else 0.0,
            fallback_rate=fb / sel if sel else 0.0,
        )

    def add_judgments(self, skill_id: str, applied_list: list[bool]):
        """添加 judgments（timestamp 从旧到新）。store.get_judgments 返 DESC。"""
        js = [
            SkillJudgment(
                judgment_id=f"j_{skill_id}_{i}",
                skill_id=skill_id, skill_name="test",
                task_id=f"t{i}",
                skill_applied=a,
                timestamp=f"2026-07-21T10:{i:02d}:00Z",
            )
            for i, a in enumerate(applied_list)
        ]
        self._judgments[skill_id] = list(reversed(js))  # DESC (newest first)

    def get_metrics(self, skill_id):
        return self._metrics.get(skill_id)

    def get_judgments(self, skill_id, limit=20):
        return self._judgments.get(skill_id, [])[:limit]

    def list_active(self):
        return list(self._records.values())

    def get(self, skill_id):
        return self._records.get(skill_id)


# ── health_report ──────────────────────────────────────

def test_health_report_basic():
    store = _FakeStore()
    store.add_skill("s1", "test-skill")
    tracker = RuntimeTracker(store)
    report = tracker.health_report("s1")
    assert report.skill_id == "s1"
    assert report.skill_name == "test-skill"
    assert report.window_selections == 10
    assert report.applied_rate == 0.6
    assert report.completion_rate > 0.0


def test_health_report_insufficient_data():
    """judgments < 4 → insufficient_data。"""
    store = _FakeStore()
    store.add_skill("s1")
    store.add_judgments("s1", [True, False])  # only 2
    tracker = RuntimeTracker(store)
    report = tracker.health_report("s1")
    assert report.trend == "insufficient_data"


def test_health_report_improving():
    """recent applied_rate > older + delta → improving。"""
    store = _FakeStore()
    store.add_skill("s1")
    # 6 judgments: older 3 = [F, F, F], recent 3 = [T, T, T]
    store.add_judgments("s1", [False, False, False, True, True, True])
    tracker = RuntimeTracker(store, degradation_delta=0.15)
    report = tracker.health_report("s1")
    assert report.trend == "improving"


def test_health_report_degrading():
    """recent applied_rate < older - delta → degrading。"""
    store = _FakeStore()
    store.add_skill("s1")
    # 6 judgments: older 3 = [T, T, T], recent 3 = [F, F, F]
    store.add_judgments("s1", [True, True, True, False, False, False])
    tracker = RuntimeTracker(store, degradation_delta=0.15)
    report = tracker.health_report("s1")
    assert report.trend == "degrading"


def test_health_report_stable():
    """recent ≈ older → stable。"""
    store = _FakeStore()
    store.add_skill("s1")
    # 6 judgments: older 3 = [T, F, T], recent 3 = [T, F, T]
    store.add_judgments("s1", [True, False, True, True, False, True])
    tracker = RuntimeTracker(store, degradation_delta=0.15)
    report = tracker.health_report("s1")
    assert report.trend == "stable"


def test_health_report_skill_not_found():
    store = _FakeStore()
    tracker = RuntimeTracker(store)
    report = tracker.health_report("nonexistent")
    assert report.trend == "insufficient_data"
    assert report.window_selections == 0


def test_health_report_recent_judgments_in_report():
    store = _FakeStore()
    store.add_skill("s1")
    store.add_judgments("s1", [True, False, True, False, True, False])
    tracker = RuntimeTracker(store)
    report = tracker.health_report("s1")
    assert len(report.recent_judgments) <= 5


# ── advice ─────────────────────────────────────────────

def test_advice_high_fallback_rate():
    store = _FakeStore()
    store.add_skill("s1", sel=10, app=2, comp=1, fb=5)  # fallback_rate=0.5
    tracker = RuntimeTracker(store)
    report = tracker.health_report("s1")
    assert "fallback_rate" in report.advice


def test_advice_low_completion_rate():
    store = _FakeStore()
    store.add_skill("s1", sel=10, app=5, comp=1, fb=2)  # completion_rate=0.2
    tracker = RuntimeTracker(store)
    report = tracker.health_report("s1")
    assert "completion_rate" in report.advice


def test_advice_degrading_mentions_rollback():
    store = _FakeStore()
    store.add_skill("s1")
    store.add_judgments("s1", [True, True, True, False, False, False])
    tracker = RuntimeTracker(store)
    report = tracker.health_report("s1")
    assert "回滚" in report.advice or "退化" in report.advice


def test_advice_improving_mentions_good():
    store = _FakeStore()
    store.add_skill("s1")
    store.add_judgments("s1", [False, False, False, True, True, True])
    tracker = RuntimeTracker(store)
    report = tracker.health_report("s1")
    assert "改善" in report.advice


# ── degraded_skills ────────────────────────────────────

def test_degraded_skills_finds_degrading():
    store = _FakeStore()
    store.add_skill("good_skill", "good")
    store.add_skill("bad_skill", "bad")
    # good: improving
    store.add_judgments("good_skill", [False, False, True, True, True, True])
    # bad: degrading
    store.add_judgments("bad_skill", [True, True, True, False, False, False])
    tracker = RuntimeTracker(store, degradation_delta=0.15)
    degraded = tracker.degraded_skills()
    assert "bad_skill" in degraded
    assert "good_skill" not in degraded


def test_degraded_skills_empty_when_all_stable():
    """recent ≈ older → stable → not degraded。"""
    store = _FakeStore()
    store.add_skill("s1")
    # 8 judgments: older 4 = [T,T,F,F], recent 4 = [T,T,F,F] → same rate → stable
    store.add_judgments("s1", [True, True, False, False, True, True, False, False])
    tracker = RuntimeTracker(store)
    assert tracker.degraded_skills() == []


def test_degraded_skills_skips_insufficient_data():
    """judgments < 4 → not degraded（insufficient_data）。"""
    store = _FakeStore()
    store.add_skill("s1")
    store.add_judgments("s1", [False, False])  # only 2
    tracker = RuntimeTracker(store)
    assert tracker.degraded_skills() == []
