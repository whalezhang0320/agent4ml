"""SQLiteSkillStore 打点 / 查询测试（B3）。

覆盖 spec Scenario:
- record_selection 递增（调 3 次 → total_selections == 3）
- record_outcome applied=True + task_completed=True → applied+1, completions+1
- record_outcome applied=False + task_completed=False → fallbacks+1
- record_outcome applied=None（guidance-skill）→ 三计数器不变
- record_outcome applied=True + task_completed=False → applied+1 only
- record_outcome applied=False + task_completed=True → 三计数器不变
- get_top_skills min_selections 过滤
- health_check degraded 两分支（数据不足 False / 低 effective_rate True）

INVARIANT #5-#9, #12.
"""
from __future__ import annotations

import pytest

from agent4ml.backend.agents.skill.store import SQLiteSkillStore
from agent4ml.backend.agents.skill.types import SkillLineage, SkillRecord


def _make_record(
    skill_id: str = "sv__imp_a1b2c3d4",
    name: str = "source-verification",
    path: str = "/skills/sv/SKILL.md",
    content_hash: str = "hash_aaaa",
) -> SkillRecord:
    return SkillRecord(
        skill_id=skill_id,
        name=name,
        path=path,
        content_hash=content_hash,
        lineage=SkillLineage(generation=0, origin="IMPORTED"),
        description="verify sources",
        allowed_tools=("web_search", "browse_page"),
    )


# ── record_selection 递增 ──────────────────────────────────

def test_record_selection_increments(tmp_path):
    """调 record_selection 3 次 → total_selections == 3。"""
    store = SQLiteSkillStore(tmp_path / "skills.db")
    rec = _make_record()
    store.register(rec)

    for _ in range(3):
        store.record_selection(rec.skill_id)

    got = store.get(rec.skill_id)
    assert got is not None
    assert got.total_selections == 3
    store.close()


def test_record_selection_nonexistent_silent(tmp_path):
    """skill_id 不存在时 record_selection 静默不抛。"""
    store = SQLiteSkillStore(tmp_path / "skills.db")
    # 不抛异常
    store.record_selection("nonexistent__imp_x")
    store.close()


# ── record_outcome applied=True + task_completed=True ──────

def test_record_outcome_applied_true_completed(tmp_path):
    """applied=True + task_completed=True → applied+1, completions+1, fallbacks 不变。"""
    store = SQLiteSkillStore(tmp_path / "skills.db")
    rec = _make_record()
    store.register(rec)

    store.record_outcome(rec.skill_id, "run-1", applied=True, task_completed=True)

    got = store.get(rec.skill_id)
    assert got is not None
    assert got.total_applied == 1
    assert got.total_completions == 1
    assert got.total_fallbacks == 0
    store.close()


# ── record_outcome applied=False + task_completed=False ────

def test_record_outcome_applied_false_not_completed(tmp_path):
    """applied=False + task_completed=False → fallbacks+1, applied/completions 不变。"""
    store = SQLiteSkillStore(tmp_path / "skills.db")
    rec = _make_record()
    store.register(rec)

    store.record_outcome(rec.skill_id, "run-1", applied=False, task_completed=False)

    got = store.get(rec.skill_id)
    assert got is not None
    assert got.total_fallbacks == 1
    assert got.total_applied == 0
    assert got.total_completions == 0
    store.close()


# ── record_outcome applied=None（guidance-skill）──────────

def test_record_outcome_applied_none_no_attribution(tmp_path):
    """applied=None（guidance-skill）→ 三计数器均不变。"""
    store = SQLiteSkillStore(tmp_path / "skills.db")
    rec = _make_record()
    store.register(rec)

    store.record_outcome(rec.skill_id, "run-1", applied=None, task_completed=True)

    got = store.get(rec.skill_id)
    assert got is not None
    assert got.total_applied == 0
    assert got.total_completions == 0
    assert got.total_fallbacks == 0
    store.close()


# ── record_outcome applied=True + task_completed=False ─────

def test_record_outcome_applied_true_not_completed(tmp_path):
    """applied=True + task_completed=False → applied+1, completions/fallbacks 不变。"""
    store = SQLiteSkillStore(tmp_path / "skills.db")
    rec = _make_record()
    store.register(rec)

    store.record_outcome(rec.skill_id, "run-1", applied=True, task_completed=False)

    got = store.get(rec.skill_id)
    assert got is not None
    assert got.total_applied == 1
    assert got.total_completions == 0
    assert got.total_fallbacks == 0
    store.close()


# ── record_outcome applied=False + task_completed=True（边界）──

def test_record_outcome_applied_false_completed_no_attribution(tmp_path):
    """applied=False + task_completed=True → 三计数器均不变（不算 fallback 也不算 completion）。"""
    store = SQLiteSkillStore(tmp_path / "skills.db")
    rec = _make_record()
    store.register(rec)

    store.record_outcome(rec.skill_id, "run-1", applied=False, task_completed=True)

    got = store.get(rec.skill_id)
    assert got is not None
    assert got.total_applied == 0
    assert got.total_completions == 0
    assert got.total_fallbacks == 0
    store.close()


# ── record_outcome skill_judgments 插入 ────────────────────

def test_record_outcome_inserts_judgment(tmp_path):
    """record_outcome 插入 skill_judgments 行（applied NULL for None）。"""
    store = SQLiteSkillStore(tmp_path / "skills.db")
    rec = _make_record()
    store.register(rec)

    store.record_outcome(rec.skill_id, "run-1", applied=None, task_completed=True)

    with store._mu:
        row = store._conn.execute(
            "SELECT * FROM skill_judgments WHERE skill_id=?", (rec.skill_id,)
        ).fetchone()
    assert row is not None
    assert row["run_id"] == "run-1"
    assert row["applied"] is None  # NULL in DB
    assert row["task_completed"] == 1
    store.close()


def test_record_outcome_nonexistent_silent(tmp_path):
    """skill_id 不存在时 record_outcome 静默跳过（不插孤立 judgment）。"""
    store = SQLiteSkillStore(tmp_path / "skills.db")
    store.record_outcome("nonexistent__imp_x", "run-1", applied=True, task_completed=True)

    with store._mu:
        rows = store._conn.execute("SELECT * FROM skill_judgments").fetchall()
    assert len(rows) == 0
    store.close()


# ── get_metrics ────────────────────────────────────────────

def test_get_metrics_returns_snapshot(tmp_path):
    """get_metrics 返回 4 计数器 + 4 rate（零除保护）。"""
    store = SQLiteSkillStore(tmp_path / "skills.db")
    rec = _make_record()
    store.register(rec)

    # 3 selections, 2 applied, 1 completion
    store.record_selection(rec.skill_id)
    store.record_selection(rec.skill_id)
    store.record_selection(rec.skill_id)
    store.record_outcome(rec.skill_id, "r1", applied=True, task_completed=True)
    store.record_outcome(rec.skill_id, "r2", applied=True, task_completed=False)

    m = store.get_metrics(rec.skill_id)
    assert m is not None
    assert m.selections == 3
    assert m.applied == 2
    assert m.completions == 1
    assert m.fallbacks == 0
    assert m.applied_rate == pytest.approx(2 / 3)
    assert m.completion_rate == pytest.approx(1 / 2)
    assert m.effective_rate == pytest.approx(1 / 3)
    assert m.fallback_rate == pytest.approx(0.0)
    store.close()


def test_get_metrics_nonexistent_returns_none(tmp_path):
    """get_metrics 不存在 skill_id 返 None。"""
    store = SQLiteSkillStore(tmp_path / "skills.db")
    assert store.get_metrics("nope") is None
    store.close()


# ── get_top_skills min_selections 过滤 ─────────────────────

def test_get_top_skills_filters_below_min(tmp_path):
    """selections < min_selections 的 skill 不参与排序。"""
    store = SQLiteSkillStore(tmp_path / "skills.db")
    # skill_a: 10 selections, effective_rate 高
    a = _make_record(skill_id="a__imp_1", name="alpha")
    store.register(a)
    for _ in range(10):
        store.record_selection(a.skill_id)
    store.record_outcome(a.skill_id, "r", applied=True, task_completed=True)

    # skill_b: 2 selections（< min=5），不参与排序
    b = _make_record(skill_id="b__imp_2", name="beta", content_hash="h_b")
    store.register(b)
    store.record_selection(b.skill_id)
    store.record_selection(b.skill_id)

    top = store.get_top_skills(n=10, min_selections=5)
    assert len(top) == 1
    assert top[0].skill_id == "a__imp_1"
    store.close()


def test_get_top_skills_orders_by_metric_desc(tmp_path):
    """按 effective_rate 降序排。"""
    store = SQLiteSkillStore(tmp_path / "skills.db")
    # skill_a: effective_rate = 1.0 (5/5)
    a = _make_record(skill_id="a__imp_1", name="alpha")
    store.register(a)
    for _ in range(5):
        store.record_selection(a.skill_id)
    store.record_outcome(a.skill_id, "r", applied=True, task_completed=True)

    # skill_b: effective_rate = 0.0 (0/5)
    b = _make_record(skill_id="b__imp_2", name="beta", content_hash="h_b")
    store.register(b)
    for _ in range(5):
        store.record_selection(b.skill_id)

    top = store.get_top_skills(n=10, min_selections=5)
    assert len(top) == 2
    assert top[0].skill_id == "a__imp_1"
    assert top[1].skill_id == "b__imp_2"
    store.close()


# ── health_check degraded 两分支 ───────────────────────────

def test_health_check_new_skill_not_degraded(tmp_path):
    """selections < min → degraded=False（数据不足不判）。"""
    store = SQLiteSkillStore(tmp_path / "skills.db")
    rec = _make_record()
    store.register(rec)
    store.record_selection(rec.skill_id)  # 1 selection < min=5

    healths = store.health_check(threshold=0.4, min_selections=5)
    assert len(healths) == 1
    h = healths[0]
    assert h.degraded is False
    assert h.skill_id == rec.skill_id
    store.close()


def test_health_check_low_effective_rate_degraded(tmp_path):
    """selections >= min 且 effective_rate < threshold → degraded=True。"""
    store = SQLiteSkillStore(tmp_path / "skills.db")
    rec = _make_record()
    store.register(rec)
    # 10 selections, 0 completions → effective_rate=0.0 < 0.4
    for _ in range(10):
        store.record_selection(rec.skill_id)

    healths = store.health_check(threshold=0.4, min_selections=5)
    assert len(healths) == 1
    h = healths[0]
    assert h.degraded is True
    assert h.effective_rate == pytest.approx(0.0)
    store.close()


def test_health_check_high_effective_rate_not_degraded(tmp_path):
    """selections >= min 且 effective_rate >= threshold → degraded=False。"""
    store = SQLiteSkillStore(tmp_path / "skills.db")
    rec = _make_record()
    store.register(rec)
    # 10 selections, 5 completions → effective_rate=0.5 >= 0.4
    for _ in range(10):
        store.record_selection(rec.skill_id)
    for _ in range(5):
        store.record_outcome(rec.skill_id, "r", applied=True, task_completed=True)

    healths = store.health_check(threshold=0.4, min_selections=5)
    assert len(healths) == 1
    assert healths[0].degraded is False
    store.close()
