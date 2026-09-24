"""E2 store skill_evolutions 表单测 — record_evolution 持久 + get_evolution_history 回溯 + v1→v2 迁移。"""
from __future__ import annotations

import pytest

from agent4ml.backend.agents.skill.evolution.types import EvolutionRecord
from agent4ml.backend.agents.skill.store import SQLiteSkillStore


def _record(
    evolution_id: str = "e1",
    skill_name: str = "sv",
    evolution_type: str = "FIX",
    timestamp: str = "2026-07-17T10:00:00",
    eval_score: float = 0.8,
    gate_decision: str = "accept",
) -> EvolutionRecord:
    return EvolutionRecord(
        evolution_id=evolution_id,
        skill_name=skill_name,
        evolution_type=evolution_type,
        trigger="METRIC",
        baseline_id="sv__imp_a1b2",
        candidate_id="sv__v1_x",
        failure_focus="指令不清",
        mutation_diff="- old\n+ new",
        eval_score=eval_score,
        gate_decision=gate_decision,
        created_version_id="sv__v1_x" if gate_decision == "accept" else None,
        timestamp=timestamp,
    )


def test_record_evolution_persists(tmp_path):
    store = SQLiteSkillStore(tmp_path / "evo.db")
    rec = _record()
    sid = store.record_evolution(rec)
    assert sid == "e1"

    history = store.get_evolution_history("sv")
    assert len(history) == 1
    row = history[0]
    assert row["evolution_id"] == "e1"
    assert row["skill_name"] == "sv"
    assert row["evolution_type"] == "FIX"
    assert row["trigger"] == "METRIC"
    assert row["baseline_id"] == "sv__imp_a1b2"
    assert row["candidate_id"] == "sv__v1_x"
    assert row["failure_focus"] == "指令不清"
    assert row["eval_score"] == 0.8
    assert row["gate_decision"] == "accept"
    assert row["created_version_id"] == "sv__v1_x"
    store.close()


def test_record_evolution_captured_no_baseline(tmp_path):
    """CAPTURED：baseline_id=None。"""
    store = SQLiteSkillStore(tmp_path / "evo.db")
    rec = EvolutionRecord(
        evolution_id="e2", skill_name="source-cross-check", evolution_type="CAPTURED",
        trigger="CAPTURE", baseline_id=None, candidate_id="scc__v1",
        failure_focus="信源交叉核验模式", mutation_diff="+ full",
        eval_score=0.75, gate_decision="accept", created_version_id="scc__v1",
        timestamp="t",
    )
    store.record_evolution(rec)
    history = store.get_evolution_history("source-cross-check")
    assert history[0]["baseline_id"] is None
    assert history[0]["evolution_type"] == "CAPTURED"
    store.close()


def test_get_evolution_history_order_desc(tmp_path):
    """降序：最新 timestamp 在前。"""
    store = SQLiteSkillStore(tmp_path / "evo.db")
    store.record_evolution(_record("e1", timestamp="2026-07-17T10:00:00"))
    store.record_evolution(_record("e2", timestamp="2026-07-17T12:00:00"))
    store.record_evolution(_record("e3", timestamp="2026-07-17T11:00:00"))

    history = store.get_evolution_history("sv")
    assert [h["evolution_id"] for h in history] == ["e2", "e3", "e1"]  # 降序
    store.close()


def test_get_evolution_history_limit(tmp_path):
    store = SQLiteSkillStore(tmp_path / "evo.db")
    for i in range(5):
        store.record_evolution(_record(f"e{i}", timestamp=f"2026-07-17T10:0{i}:00"))
    history = store.get_evolution_history("sv", limit=2)
    assert len(history) == 2
    store.close()


def test_get_evolution_history_empty(tmp_path):
    store = SQLiteSkillStore(tmp_path / "evo.db")
    assert store.get_evolution_history("nonexistent") == []
    store.close()


def test_record_evolution_replace_same_id(tmp_path):
    """同 evolution_id 重写（INSERT OR REPLACE）。"""
    store = SQLiteSkillStore(tmp_path / "evo.db")
    store.record_evolution(_record("e1", eval_score=0.5, gate_decision="reject", timestamp="t1"))
    store.record_evolution(_record("e1", eval_score=0.9, gate_decision="accept", timestamp="t2"))
    history = store.get_evolution_history("sv")
    assert len(history) == 1  # 同 id 不重复
    assert history[0]["eval_score"] == 0.9  # 覆盖
    assert history[0]["gate_decision"] == "accept"
    store.close()


def test_migration_v1_to_v2(tmp_path):
    """v1 DB（无 skill_evolutions）开新代码 → 迁移 v1→v2→v3 建表。"""
    db = tmp_path / "v1.db"
    # 先用新代码建到 v3
    store = SQLiteSkillStore(db)
    store._conn.execute("DROP TABLE skill_evolutions")
    store._conn.execute("PRAGMA user_version = 1")
    store._conn.commit()
    store.close()

    # 重新开 → _init_schema 应迁移 v1→v2→v3
    store2 = SQLiteSkillStore(db)
    assert store2._conn.execute("PRAGMA user_version").fetchone()[0] == 3
    # skill_evolutions 表存在（迁移重建）
    store2.record_evolution(_record("e1"))
    assert len(store2.get_evolution_history("sv")) == 1
    store2.close()


def test_migration_idempotent(tmp_path):
    """重复实例化（已 v3）不重复迁移。"""
    db = tmp_path / "e.db"
    SQLiteSkillStore(db).close()
    store2 = SQLiteSkillStore(db)
    assert store2._conn.execute("PRAGMA user_version").fetchone()[0] == 3
    # 表仍可用
    store2.record_evolution(_record("e1"))
    assert len(store2.get_evolution_history("sv")) == 1
    store2.close()


def test_fresh_db_has_skill_evolutions_table(tmp_path):
    """全新 DB：_SCHEMA_SQL 建表（v3）。"""
    store = SQLiteSkillStore(tmp_path / "fresh.db")
    assert store._conn.execute("PRAGMA user_version").fetchone()[0] == 3
    tables = {r[0] for r in store._conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()}
    assert "skill_evolutions" in tables
    store.close()
