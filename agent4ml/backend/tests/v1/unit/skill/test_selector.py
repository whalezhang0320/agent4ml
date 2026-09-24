"""SkillSelector 单测（B6）— override 强制 + quality filter + LLM select + fallback。"""
from __future__ import annotations

from agent4ml.backend.agents.skill.selector import SkillSelector
from agent4ml.backend.agents.skill.types import SkillRecord


def _rec(
    skill_id: str, name: str, *,
    selections: int = 0, applied: int = 0, completions: int = 0,
    enabled: bool = True, description: str = "d",
) -> SkillRecord:
    return SkillRecord(
        skill_id=skill_id, name=name, path=f"/p/{name}", content_hash="h",
        enabled=enabled, description=description,
        total_selections=selections, total_applied=applied,
        total_completions=completions,
    )


class FakeStore:
    def __init__(self, records: list[SkillRecord]) -> None:
        self._records = {r.skill_id: r for r in records}

    def list_active(self) -> list[SkillRecord]:
        return [r for r in self._records.values() if r.is_active]

    def get_active(self, name: str) -> SkillRecord | None:
        for r in self._records.values():
            if r.name == name and r.is_active:
                return r
        return None


class FakeLLM:
    def __init__(self, content: str) -> None:
        self._content = content

    def invoke(self, messages, **kwargs):
        return type("Resp", (), {"content": self._content})()


def test_override_forced_not_filtered():
    # skill_a: 低 effective_rate + selections>=min → 正常会被淘汰，但 override 强制包含
    store = FakeStore([_rec("a", "skill-a", selections=10, completions=1)])  # eff 0.1
    sel = SkillSelector(store, llm=None, max_skills=3, quality_threshold=0.3, min_selections=5)
    result = sel.select_for_task("task", overrides=["skill-a"])
    assert any(r.skill_id == "a" for r in result)


def test_quality_filter_excludes_low_rate_mature_skill():
    store = FakeStore([
        _rec("a", "skill-a", selections=10, completions=1),   # eff 0.1 < 0.3, sel>=5 → 淘汰
        _rec("b", "skill-b", selections=10, completions=8),   # eff 0.8 → 保留
    ])
    sel = SkillSelector(store, llm=None, max_skills=3, quality_threshold=0.3, min_selections=5)
    result = sel.select_for_task("task")
    ids = {r.skill_id for r in result}
    assert "b" in ids
    assert "a" not in ids


def test_new_skill_not_filtered():
    # selections < min → 不淘汰（给数据积累）
    store = FakeStore([_rec("a", "skill-a", selections=2, completions=0)])  # eff 0.0 but sel<5
    sel = SkillSelector(store, llm=None, max_skills=3, quality_threshold=0.3, min_selections=5)
    result = sel.select_for_task("task")
    assert any(r.skill_id == "a" for r in result)


def test_le_max_returns_all_skips_llm():
    store = FakeStore([_rec("a", "a"), _rec("b", "b")])
    llm = FakeLLM('{"skills": ["a"]}')  # 不应被调用
    sel = SkillSelector(store, llm=llm, max_skills=3)
    result = sel.select_for_task("task")
    assert len(result) == 2


def test_gt_max_with_llm_selects():
    store = FakeStore([
        _rec("a", "a", description="alpha"),
        _rec("b", "b", description="beta"),
        _rec("c", "c", description="gamma"),
        _rec("d", "d", description="delta"),
    ])
    llm = FakeLLM('{"skills": ["b", "d"]}')
    sel = SkillSelector(store, llm=llm, max_skills=2)
    result = sel.select_for_task("task")
    ids = {r.skill_id for r in result}
    assert ids == {"b", "d"}


def test_gt_max_no_llm_fallback_effective_rate():
    store = FakeStore([
        _rec("a", "a", selections=10, completions=5),   # eff 0.5
        _rec("b", "b", selections=10, completions=8),   # eff 0.8
        _rec("c", "c", selections=10, completions=2),   # eff 0.2
        _rec("d", "d", selections=10, completions=9),   # eff 0.9
    ])
    sel = SkillSelector(store, llm=None, max_skills=2)
    result = sel.select_for_task("task")
    ids = [r.skill_id for r in result]
    assert ids == ["d", "b"]  # top 2 by effective_rate


def test_llm_failure_fallback_ranking():
    store = FakeStore([
        _rec("a", "a", selections=10, completions=5),
        _rec("b", "b", selections=10, completions=9),
        _rec("c", "c", selections=10, completions=1),
        _rec("d", "d", selections=10, completions=3),
    ])
    llm = FakeLLM("not json at all")  # 解析失败
    sel = SkillSelector(store, llm=llm, max_skills=2)
    result = sel.select_for_task("task")
    ids = [r.skill_id for r in result]
    assert ids == ["b", "a"]  # fallback effective_rate 排序


def test_no_store_returns_empty():
    sel = SkillSelector(store=None, llm=None, max_skills=3)
    assert sel.select_for_task("task") == []


def test_dedup_override_and_filtered():
    # override skill-a 同时也在 active 列表 → 不重复
    store = FakeStore([_rec("a", "skill-a"), _rec("b", "skill-b")])
    sel = SkillSelector(store, llm=None, max_skills=3)
    result = sel.select_for_task("task", overrides=["skill-a"])
    ids = [r.skill_id for r in result]
    assert ids.count("a") == 1


def test_disabled_skill_excluded():
    store = FakeStore([
        _rec("a", "skill-a", enabled=False),
        _rec("b", "skill-b", enabled=True),
    ])
    sel = SkillSelector(store, llm=None, max_skills=3)
    result = sel.select_for_task("task")
    ids = {r.skill_id for r in result}
    assert "a" not in ids
    assert "b" in ids


def test_llm_select_caps_at_max():
    store = FakeStore([_rec("a", "a"), _rec("b", "b"), _rec("c", "c"), _rec("d", "d")])
    llm = FakeLLM('{"skills": ["a", "b", "c", "d"]}')  # LLM 返 4 个，应截到 max=2
    sel = SkillSelector(store, llm=llm, max_skills=2)
    result = sel.select_for_task("task")
    assert len(result) == 2
