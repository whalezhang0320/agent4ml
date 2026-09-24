"""SQLiteSkillStore schema + 迁移单测（B1b）。"""
from __future__ import annotations

from agent4ml.backend.agents.skill.store import _SCHEMA_VERSION, SQLiteSkillStore


def _tables(conn) -> set[str]:
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    ).fetchall()
    return {r[0] for r in rows}


def _indices(conn) -> set[str]:
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='index' AND name LIKE 'idx_%' ORDER BY name"
    ).fetchall()
    return {r[0] for r in rows}


def test_schema_creates_tables_indices_and_version(tmp_path):
    store = SQLiteSkillStore(tmp_path / "skills.db")
    try:
        conn = store._conn
        assert _tables(conn) >= {"skill_records", "skill_lineage_parents", "skill_judgments"}
        assert _indices(conn) >= {"idx_sr_name", "idx_sr_active", "idx_sj_skill", "idx_sj_run"}
        v = conn.execute("PRAGMA user_version").fetchone()[0]
        assert v == _SCHEMA_VERSION
    finally:
        store.close()


def test_wal_mode(tmp_path):
    store = SQLiteSkillStore(tmp_path / "skills.db")
    try:
        mode = store._conn.execute("PRAGMA journal_mode").fetchone()[0]
        assert mode == "wal"
    finally:
        store.close()


def test_migration_idempotent(tmp_path):
    db = tmp_path / "skills.db"
    s1 = SQLiteSkillStore(db)
    s1.close()
    s2 = SQLiteSkillStore(db)
    try:
        v = s2._conn.execute("PRAGMA user_version").fetchone()[0]
        assert v == _SCHEMA_VERSION
        assert _tables(s2._conn) >= {"skill_records", "skill_lineage_parents", "skill_judgments"}
    finally:
        s2.close()


def test_skill_records_columns_no_content(tmp_path):
    store = SQLiteSkillStore(tmp_path / "skills.db")
    try:
        cols = {
            r[1] for r in store._conn.execute("PRAGMA table_info(skill_records)").fetchall()
        }
        for c in (
            "skill_id", "name", "path", "content_hash", "is_active",
            "generation", "origin", "total_selections", "total_applied",
            "total_completions", "total_fallbacks", "allowed_tools",
        ):
            assert c in cols
        # 内容/索引分离：不存全文列
        assert "content" not in cols
        assert "text" not in cols
    finally:
        store.close()


def test_parent_dir_created(tmp_path):
    store = SQLiteSkillStore(tmp_path / "nested" / "dir" / "skills.db")
    try:
        assert (tmp_path / "nested" / "dir" / "skills.db").exists()
    finally:
        store.close()
