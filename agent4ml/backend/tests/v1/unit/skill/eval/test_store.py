"""L3 eval store 持久化单测（L3-E2）— v2→v3 迁移 + CRUD。

验证：
- v2→v3 迁移幂等（老 v2 DB 升级不丢数据）
- save_judgment / get_judgments CRUD
- save_task_score / get_task_scores CRUD
- save_eval_run CRUD
- 三表 schema 存在
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from agent4ml.backend.agents.skill.eval.types import (
    EvalRun,
    SkillJudgment,
    TaskQualityScore,
)
from agent4ml.backend.agents.skill.store import SQLiteSkillStore


@pytest.fixture()
def store(tmp_path: Path) -> SQLiteSkillStore:
    return SQLiteSkillStore(tmp_path / "test_eval.db")


def _make_judgment(jid: str = "j1", skill_applied: bool = True) -> SkillJudgment:
    return SkillJudgment(
        judgment_id=jid,
        skill_id="skill-1__v0_abc",
        skill_name="test-skill",
        task_id="task-1",
        skill_applied=skill_applied,
        deviation_note="skipped validate step",
        timestamp="2026-07-21T10:00:00Z",
    )


def _make_score(sid: str = "q1") -> TaskQualityScore:
    return TaskQualityScore(
        score_id=sid,
        task_id="task-1",
        task_completion=0.9,
        response_quality=0.8,
        efficiency=0.7,
        tool_usage=0.8,
        overall_score=0.835,
        rationale="good analysis",
        timestamp="2026-07-21T10:00:00Z",
    )


def _make_run(rid: str = "e1") -> EvalRun:
    return EvalRun(
        eval_run_id=rid,
        eval_layer="execution",
        skill_ids=("skill-1__v0_abc",),
        candidate_id=None,
        baseline_id=None,
        result_json='{"score": 0.875}',
        timestamp="2026-07-21T10:00:00Z",
    )


# ── schema 版本 ──────────────────────────────────────────

def test_schema_version_is_3(store: SQLiteSkillStore):
    ver = store._conn.execute("PRAGMA user_version").fetchone()[0]
    assert ver == 3


def test_three_eval_tables_exist(store: SQLiteSkillStore):
    tables = {
        r[0]
        for r in store._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    assert "skill_eval_judgments" in tables
    assert "task_quality_scores" in tables
    assert "skill_eval_runs" in tables


def test_existing_tables_preserved(store: SQLiteSkillStore):
    """L1/L2 表不丢。"""
    tables = {
        r[0]
        for r in store._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    assert "skill_records" in tables
    assert "skill_lineage_parents" in tables
    assert "skill_judgments" in tables  # L1 原有
    assert "skill_evolutions" in tables


# ── v2→v3 迁移幂等 ───────────────────────────────────────

def test_v2_to_v3_migration(tmp_path: Path):
    """手动建 v2 DB（无 L3 三表），用 SQLiteSkillStore 打开应自动迁移到 v3。"""
    db_path = tmp_path / "v2.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA user_version = 2")
    # 建 v2 已有的表（完整 schema，含 timestamp 供索引创建）
    conn.execute("""
        CREATE TABLE skill_evolutions (
            evolution_id        TEXT PRIMARY KEY,
            skill_name          TEXT NOT NULL,
            evolution_type      TEXT NOT NULL,
            trigger             TEXT NOT NULL,
            baseline_id         TEXT,
            candidate_id        TEXT NOT NULL,
            failure_focus       TEXT NOT NULL DEFAULT '',
            mutation_diff       TEXT NOT NULL DEFAULT '',
            eval_score          REAL NOT NULL DEFAULT 0.0,
            gate_decision       TEXT NOT NULL,
            created_version_id  TEXT,
            timestamp           TEXT NOT NULL DEFAULT ''
        )
    """)
    conn.execute(
        "INSERT INTO skill_evolutions VALUES ('e1','test','FIX','METRIC',NULL,'c1','','',0.0,'accept',NULL,'')"
    )
    conn.commit()
    conn.close()

    store = SQLiteSkillStore(db_path)
    ver = store._conn.execute("PRAGMA user_version").fetchone()[0]
    assert ver == 3

    # 存量数据不丢
    row = store._conn.execute(
        "SELECT * FROM skill_evolutions WHERE evolution_id='e1'"
    ).fetchone()
    assert row["skill_name"] == "test"

    # 新三表存在
    tables = {
        r[0]
        for r in store._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    assert "skill_eval_judgments" in tables
    assert "task_quality_scores" in tables
    assert "skill_eval_runs" in tables


# ── SkillJudgment CRUD ──────────────────────────────────

def test_save_and_get_judgment(store: SQLiteSkillStore):
    j = _make_judgment()
    jid = store.save_judgment(j)
    assert jid == "j1"

    got = store.get_judgments("skill-1__v0_abc")
    assert len(got) == 1
    assert got[0].judgment_id == "j1"
    assert got[0].skill_applied is True
    assert got[0].deviation_note == "skipped validate step"


def test_get_judgments_empty(store: SQLiteSkillStore):
    got = store.get_judgments("nonexistent")
    assert got == []


def test_get_judgments_limit(store: SQLiteSkillStore):
    for i in range(5):
        store.save_judgment(_make_judgment(jid=f"j{i}", skill_applied=bool(i % 2)))
    got = store.get_judgments("skill-1__v0_abc", limit=3)
    assert len(got) == 3


def test_save_judgment_applied_false(store: SQLiteSkillStore):
    j = _make_judgment(jid="j_false", skill_applied=False)
    store.save_judgment(j)
    got = store.get_judgments("skill-1__v0_abc")
    assert any(not g.skill_applied for g in got)


def test_save_judgment_overwrite(store: SQLiteSkillStore):
    store.save_judgment(_make_judgment(jid="j1", skill_applied=True))
    store.save_judgment(_make_judgment(jid="j1", skill_applied=False))
    got = store.get_judgments("skill-1__v0_abc")
    assert len(got) == 1
    assert got[0].skill_applied is False


# ── TaskQualityScore CRUD ───────────────────────────────

def test_save_and_get_task_score(store: SQLiteSkillStore):
    s = _make_score()
    sid = store.save_task_score(s)
    assert sid == "q1"

    got = store.get_task_scores("task-1")
    assert got is not None
    assert got.overall_score == pytest.approx(0.835)
    assert got.rationale == "good analysis"


def test_get_task_score_none(store: SQLiteSkillStore):
    got = store.get_task_scores("nonexistent")
    assert got is None


# ── EvalRun CRUD ────────────────────────────────────────

def test_save_eval_run(store: SQLiteSkillStore):
    run = _make_run()
    rid = store.save_eval_run(run)
    assert rid == "e1"

    row = store._conn.execute(
        "SELECT * FROM skill_eval_runs WHERE eval_run_id='e1'"
    ).fetchone()
    assert row["eval_layer"] == "execution"
    assert row["result_json"] == '{"score": 0.875}'


def test_save_eval_run_response_layer(store: SQLiteSkillStore):
    run = EvalRun(
        eval_run_id="e2",
        eval_layer="response",
        skill_ids=("skill-1",),
        candidate_id="cand-1",
        baseline_id="base-1",
        result_json='{"score": 0.9}',
        timestamp="2026-07-21T11:00:00Z",
    )
    store.save_eval_run(run)
    row = store._conn.execute(
        "SELECT * FROM skill_eval_runs WHERE eval_run_id='e2'"
    ).fetchone()
    assert row["candidate_id"] == "cand-1"
    assert row["baseline_id"] == "base-1"
