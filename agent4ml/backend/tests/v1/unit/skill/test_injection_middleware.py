"""SkillInjectionMiddleware 单测（B7）— 注入 + 打点 + provenance + 静默降级。"""
from __future__ import annotations

from langchain_core.messages import SystemMessage

from agent4ml.backend.agents.middlewares.skill_injection_middleware import SkillInjectionMiddleware
from agent4ml.backend.agents.skill.types import SkillRecord


class FakeStore:
    def __init__(self) -> None:
        self.selections: list[str] = []
        self._raise = False

    def record_selection(self, skill_id: str) -> None:
        if self._raise:
            raise RuntimeError("boom")
        self.selections.append(skill_id)


class FakeSelector:
    def __init__(self, records: list[SkillRecord]) -> None:
        self._records = records
        self.last_overrides = None

    def select_for_task(self, task: str, overrides=None):
        self.last_overrides = overrides
        return self._records


class FakeJournal:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []

    def append(self, event_type: str, payload: dict) -> None:
        self.events.append((event_type, payload))


def _rec(skill_id: str, name: str, path: str) -> SkillRecord:
    return SkillRecord(skill_id=skill_id, name=name, path=path, content_hash="h")


def _write_skill(tmp_path, name: str, body: str) -> str:
    p = tmp_path / name / "SKILL.md"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(f"---\nname: {name}\ndescription: d\n---\n{body}", encoding="utf-8")
    return str(p)


def _runtime(journal=None, run_id="r1"):
    class _R:
        pass
    r = _R()
    r.context = {"journal": journal, "run_id": run_id} if journal is not None else {"run_id": run_id}
    return r


def test_injects_systemmessage_and_points_and_provenance(tmp_path):
    path = _write_skill(tmp_path, "skill-a", "# Body A")
    store = FakeStore()
    sel = FakeSelector([_rec("a", "skill-a", path)])
    mw = SkillInjectionMiddleware(store, sel)
    state = {"user_input": "task", "metadata": {}}
    result = mw.before_model(state, _runtime(FakeJournal()))
    assert result is not None
    assert len(result["messages"]) == 1
    assert isinstance(result["messages"][0], SystemMessage)
    assert "### Skill: skill-a" in result["messages"][0].content
    assert "# Body A" in result["messages"][0].content
    assert result["metadata"]["active_skills"] == ["a"]
    assert result["metadata"]["skill_applied"] == {"a": None}
    assert store.selections == ["a"]


def test_journal_skill_select_event(tmp_path):
    path = _write_skill(tmp_path, "skill-a", "body")
    journal = FakeJournal()
    mw = SkillInjectionMiddleware(FakeStore(), FakeSelector([_rec("a", "skill-a", path)]))
    mw.before_model({"user_input": "t", "metadata": {}}, _runtime(journal))
    assert len(journal.events) == 1
    assert journal.events[0][0] == "skill.select"
    assert journal.events[0][1]["skill_id"] == "a"


def test_no_active_returns_none():
    mw = SkillInjectionMiddleware(FakeStore(), FakeSelector([]))
    result = mw.before_model({"user_input": "t", "metadata": {}}, _runtime())
    assert result is None


def test_overrides_passed_to_selector(tmp_path):
    path = _write_skill(tmp_path, "skill-a", "body")
    sel = FakeSelector([_rec("a", "skill-a", path)])
    mw = SkillInjectionMiddleware(FakeStore(), sel)
    mw.before_model({"user_input": "t", "metadata": {"skill_override": ["skill-a"]}}, _runtime())
    assert sel.last_overrides == ["skill-a"]


def test_no_journal_silent(tmp_path):
    path = _write_skill(tmp_path, "skill-a", "body")
    mw = SkillInjectionMiddleware(FakeStore(), FakeSelector([_rec("a", "skill-a", path)]))
    # runtime 无 journal
    result = mw.before_model({"user_input": "t", "metadata": {}}, _runtime(journal=None))
    assert result is not None  # 不抛，仍注入


def test_store_raises_silent(tmp_path):
    path = _write_skill(tmp_path, "skill-a", "body")
    store = FakeStore()
    store._raise = True
    mw = SkillInjectionMiddleware(store, FakeSelector([_rec("a", "skill-a", path)]))
    result = mw.before_model({"user_input": "t", "metadata": {}}, _runtime(FakeJournal()))
    assert result is not None  # 不抛，仍注入 + provenance
    assert result["metadata"]["active_skills"] == ["a"]


def test_multiple_skills_injected(tmp_path):
    p1 = _write_skill(tmp_path, "skill-a", "body a")
    p2 = _write_skill(tmp_path, "skill-b", "body b")
    mw = SkillInjectionMiddleware(FakeStore(), FakeSelector([
        _rec("a", "skill-a", p1), _rec("b", "skill-b", p2),
    ]))
    result = mw.before_model({"user_input": "t", "metadata": {}}, _runtime())
    assert result["metadata"]["active_skills"] == ["a", "b"]
    assert "skill-a" in result["messages"][0].content
    assert "skill-b" in result["messages"][0].content


def test_missing_skill_file_injects_header_only(tmp_path):
    # path 不存在 → body 空，仍注入 header
    mw = SkillInjectionMiddleware(FakeStore(), FakeSelector([_rec("a", "skill-a", "/nonexistent.md")]))
    result = mw.before_model({"user_input": "t", "metadata": {}}, _runtime())
    assert result is not None
    assert "### Skill: skill-a" in result["messages"][0].content
