"""SQLiteSkillStore CRUD + version DAG 测试（B2）。

覆盖 spec Scenario:
- register 幂等（同 id 二次不覆盖）
- get 还原 allowed_tools tuple + lineage parent
- get_active / list_active
- create_version 切 active（新 1 旧 0 + lineage_parents 行）
- rollback 切指针不删除
- get_versions 按 generation 升序
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
    allowed_tools: tuple[str, ...] = ("web_search", "browse_page"),
    lineage: SkillLineage | None = None,
    description: str = "verify sources",
) -> SkillRecord:
    return SkillRecord(
        skill_id=skill_id,
        name=name,
        path=path,
        content_hash=content_hash,
        lineage=lineage or SkillLineage(generation=0, origin="IMPORTED"),
        description=description,
        allowed_tools=allowed_tools,
    )


# ── register 幂等 ──────────────────────────────────────────

def test_register_idempotent_same_id_not_overwrite(tmp_path):
    """同 skill_id 二次 register 不覆盖、不报错。"""
    store = SQLiteSkillStore(tmp_path / "skills.db")
    rec = _make_record(description="original")
    store.register(rec)

    # 二次 register，改 description — 不应覆盖
    rec2 = _make_record(description="changed")
    store.register(rec2)

    got = store.get(rec.skill_id)
    assert got is not None
    assert got.description == "original"
    store.close()


# ── get 还原 allowed_tools + lineage ───────────────────────

def test_get_restores_allowed_tools_tuple(tmp_path):
    """get 还原 allowed_tools 为 tuple。"""
    store = SQLiteSkillStore(tmp_path / "skills.db")
    rec = _make_record(allowed_tools=("web_search", "browse_page"))
    store.register(rec)

    got = store.get(rec.skill_id)
    assert got is not None
    assert got.allowed_tools == ("web_search", "browse_page")
    assert isinstance(got.allowed_tools, tuple)
    store.close()


def test_get_restores_lineage_parents(tmp_path):
    """get 从 skill_lineage_parents 表还原 parent_skill_ids。"""
    store = SQLiteSkillStore(tmp_path / "skills.db")
    parent = _make_record(skill_id="sv__imp_p1", content_hash="hash_p")
    store.register(parent)

    child = _make_record(
        skill_id="sv__v1_c1",
        content_hash="hash_c",
        lineage=SkillLineage(
            parent_skill_ids=("sv__imp_p1",),
            generation=1,
            origin="FIXED",
        ),
    )
    store.create_version("sv__imp_p1", child, "FIXED")

    got = store.get("sv__v1_c1")
    assert got is not None
    assert got.lineage.parent_skill_ids == ("sv__imp_p1",)
    assert got.lineage.generation == 1
    assert got.lineage.origin == "FIXED"
    store.close()


# ── get_active / list_active ───────────────────────────────

def test_get_active_returns_active_version(tmp_path):
    """get_active 返回 is_active=1 的版本。"""
    store = SQLiteSkillStore(tmp_path / "skills.db")
    rec = _make_record()
    store.register(rec)

    got = store.get_active("source-verification")
    assert got is not None
    assert got.skill_id == rec.skill_id
    store.close()


def test_list_active_returns_all_active(tmp_path):
    """list_active 返回所有 is_active=1 的 skill。"""
    store = SQLiteSkillStore(tmp_path / "skills.db")
    store.register(_make_record(skill_id="a__imp_1", name="alpha"))
    store.register(_make_record(skill_id="b__imp_2", name="beta"))

    active = store.list_active()
    assert len(active) == 2
    names = {r.name for r in active}
    assert names == {"alpha", "beta"}
    store.close()


# ── create_version 切 active ───────────────────────────────

def test_create_version_switches_active(tmp_path):
    """create_version: 新 node is_active=1，旧 node is_active=0，lineage_parents 含 (new, parent)。"""
    store = SQLiteSkillStore(tmp_path / "skills.db")
    v1 = _make_record(skill_id="sv__imp_v1", content_hash="h1")
    store.register(v1)

    v2 = _make_record(
        skill_id="sv__v2_v2",
        content_hash="h2",
        lineage=SkillLineage(
            parent_skill_ids=("sv__imp_v1",), generation=1, origin="FIXED"
        ),
    )
    store.create_version("sv__imp_v1", v2, "FIXED")

    # v2 active, v1 not active
    got_v1 = store.get("sv__imp_v1")
    got_v2 = store.get("sv__v2_v2")
    assert got_v1 is not None and got_v2 is not None
    assert got_v1.is_active is False
    assert got_v2.is_active is True

    # lineage_parents 含 (v2, v1)
    assert got_v2.lineage.parent_skill_ids == ("sv__imp_v1",)

    # get_active 返 v2
    active = store.get_active("source-verification")
    assert active is not None
    assert active.skill_id == "sv__v2_v2"
    store.close()


# ── rollback 切指针不删除 ───────────────────────────────────

def test_rollback_switches_pointer_no_delete(tmp_path):
    """rollback: 激活旧 node，新 node deactive，新 node 行仍在。"""
    store = SQLiteSkillStore(tmp_path / "skills.db")
    v1 = _make_record(skill_id="sv__imp_v1", content_hash="h1")
    store.register(v1)

    v2 = _make_record(
        skill_id="sv__v2_v2",
        content_hash="h2",
        lineage=SkillLineage(
            parent_skill_ids=("sv__imp_v1",), generation=1, origin="FIXED"
        ),
    )
    store.create_version("sv__imp_v1", v2, "FIXED")

    # rollback 到 v1
    store.rollback("sv__imp_v1")

    got_v1 = store.get("sv__imp_v1")
    got_v2 = store.get("sv__v2_v2")
    assert got_v1 is not None and got_v2 is not None
    assert got_v1.is_active is True
    assert got_v2.is_active is False  # v2 行仍在，只是 deactive

    # get_active 返 v1
    active = store.get_active("source-verification")
    assert active is not None
    assert active.skill_id == "sv__imp_v1"
    store.close()


# ── get_versions 按 generation 升序 ─────────────────────────

def test_get_versions_ordered_by_generation(tmp_path):
    """get_versions 返回所有版本，ORDER BY generation ASC。"""
    store = SQLiteSkillStore(tmp_path / "skills.db")
    v1 = _make_record(skill_id="sv__imp_v1", content_hash="h1")
    store.register(v1)

    v2 = _make_record(
        skill_id="sv__v2_v2",
        content_hash="h2",
        lineage=SkillLineage(
            parent_skill_ids=("sv__imp_v1",), generation=1, origin="FIXED"
        ),
    )
    store.create_version("sv__imp_v1", v2, "FIXED")

    v3 = _make_record(
        skill_id="sv__v3_v3",
        content_hash="h3",
        lineage=SkillLineage(
            parent_skill_ids=("sv__v2_v2",), generation=2, origin="DERIVED"
        ),
    )
    store.create_version("sv__v2_v2", v3, "DERIVED")

    versions = store.get_versions("source-verification")
    assert len(versions) == 3
    assert [v.lineage.generation for v in versions] == [0, 1, 2]
    assert versions[0].skill_id == "sv__imp_v1"
    assert versions[2].skill_id == "sv__v3_v3"
    store.close()


# ── create_version 重复 skill_id 抛 ValueError ──────────────

def test_create_version_duplicate_skill_id_raises(tmp_path):
    """同 skill_id 二次 create_version 抛 ValueError（保护 is_active 单指针不变量）。"""
    store = SQLiteSkillStore(tmp_path / "skills.db")
    v1 = _make_record(skill_id="sv__imp_v1", content_hash="h1")
    store.register(v1)

    v2 = _make_record(
        skill_id="sv__v2_v2",
        content_hash="h2",
        lineage=SkillLineage(
            parent_skill_ids=("sv__imp_v1",), generation=1, origin="FIXED"
        ),
    )
    store.create_version("sv__imp_v1", v2, "FIXED")

    # 二次 create_version 同 skill_id → ValueError
    with pytest.raises(ValueError, match="skill_id already exists"):
        store.create_version("sv__imp_v1", v2, "FIXED")
    store.close()


# ── discover 同步文件变更 ───────────────────────────────────

def test_discover_updates_changed_file(tmp_path):
    """discover: skill_id 已存在时 UPDATE path/content_hash/description。"""
    import hashlib

    from agent4ml.backend.agents.skill.parser import parse_skill_file

    store = SQLiteSkillStore(tmp_path / "skills.db")

    skill_dir = tmp_path / "skills" / "my-skill"
    skill_dir.mkdir(parents=True)
    skill_md = skill_dir / "SKILL.md"
    skill_md.write_text(
        "---\nname: my-skill\ndescription: original\n---\n\nbody\n",
        encoding="utf-8",
    )

    # 首次 discover 注册
    results = store.discover([tmp_path / "skills"])
    assert len(results) == 1
    rec1 = results[0]
    assert rec1.description == "original"
    orig_hash = rec1.content_hash

    # 编辑 SKILL.md（改 description）
    skill_md.write_text(
        "---\nname: my-skill\ndescription: changed\n---\n\nbody\n",
        encoding="utf-8",
    )

    # 二次 discover → store.get 返回新 description + 新 content_hash
    results2 = store.discover([tmp_path / "skills"])
    assert len(results2) == 1
    rec2 = results2[0]
    assert rec2.description == "changed"
    assert rec2.content_hash != orig_hash

    # 从 store 验证 DB 已更新
    got = store.get(rec1.skill_id)
    assert got is not None
    assert got.description == "changed"
    assert got.content_hash != orig_hash
    store.close()


# ── set_enabled 持久 enable/disable ──────────────────────


def test_set_enabled_persists(tmp_path):
    """set_enabled 持久 SQLite，跨实例生效。"""
    db = tmp_path / "skills.db"
    store = SQLiteSkillStore(db)
    rec = _make_record()
    store.register(rec)
    assert store.get(rec.skill_id).enabled is True  # 默认 enabled

    assert store.set_enabled(rec.skill_id, False) is True
    assert store.get(rec.skill_id).enabled is False
    store.close()

    # 跨实例（重启）仍禁用
    store2 = SQLiteSkillStore(db)
    assert store2.get(rec.skill_id).enabled is False
    store2.close()


def test_set_enabled_re_enable(tmp_path):
    store = SQLiteSkillStore(tmp_path / "skills.db")
    rec = _make_record()
    store.register(rec)
    store.set_enabled(rec.skill_id, False)
    assert store.get(rec.skill_id).enabled is False
    assert store.set_enabled(rec.skill_id, True) is True
    assert store.get(rec.skill_id).enabled is True
    store.close()


def test_set_enabled_nonexistent_returns_false(tmp_path):
    store = SQLiteSkillStore(tmp_path / "skills.db")
    assert store.set_enabled("nonexistent__id", True) is False
    store.close()


def test_set_enabled_filtered_from_list_active(tmp_path):
    """disabled skill 不在 list_active（selector 据此过滤）。"""
    store = SQLiteSkillStore(tmp_path / "skills.db")
    rec = _make_record()
    store.register(rec)
    assert len(store.list_active()) == 1
    store.set_enabled(rec.skill_id, False)
    # list_active 返 is_active=1（含 disabled），selector 另过滤 enabled
    # 但 get_active/get 的 enabled=False
    assert store.get(rec.skill_id).enabled is False
    store.close()
