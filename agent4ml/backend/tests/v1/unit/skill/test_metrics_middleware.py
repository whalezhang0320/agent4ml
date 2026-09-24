"""SkillMetricsMiddleware 单测（B8）— applied 打点 + 归因 + task_completed + 降级。"""
from __future__ import annotations

import asyncio

import pytest

from agent4ml.backend.agents.middlewares.skill_metrics_middleware import SkillMetricsMiddleware
from agent4ml.backend.agents.skill._ctx import _active_skills_ctx, _applied_ctx
from agent4ml.backend.agents.skill.types import SkillRecord


@pytest.fixture(autouse=True)
def _reset_ctx():
    """每个测试后重置 provenance ContextVar。"""
    tok_a = _active_skills_ctx.set(None)
    tok_b = _applied_ctx.set(None)
    yield
    _active_skills_ctx.reset(tok_a)
    _applied_ctx.reset(tok_b)


def _rec(skill_id: str, allowed_tools: tuple[str, ...] = ()) -> SkillRecord:
    return SkillRecord(skill_id=skill_id, name=skill_id, path="/p", content_hash="h",
                       allowed_tools=allowed_tools)


class FakeStore:
    def __init__(self, records=None) -> None:
        self._records = {r.skill_id: r for r in (records or [])}
        self.outcomes: list[tuple] = []

    def get(self, sid: str):
        return self._records.get(sid)

    def record_outcome(self, sid, run_id, applied, task_completed, note=""):
        self.outcomes.append((sid, applied, task_completed))


class FakeJournal:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []

    def append(self, event_type, payload):
        self.events.append((event_type, payload))


class FakeReq:
    def __init__(self, tool_name: str, journal=None) -> None:
        self.tool_call = {"name": tool_name, "args": {}, "id": "1"}
        r = type("R", (), {})()
        r.context = {"journal": journal} if journal is not None else {}
        self.runtime = r


def _runtime(journal=None, run_id="r1"):
    r = type("R", (), {})()
    r.context = {"journal": journal, "run_id": run_id} if journal is not None else {"run_id": run_id}
    return r


async def _ok(_r):
    return "result"


def test_awrap_marks_tool_skill_applied_and_journal():
    _active_skills_ctx.set([("a", ("web_search",)), ("b", ())])
    _applied_ctx.set({"a": None, "b": None})
    journal = FakeJournal()
    mw = SkillMetricsMiddleware(FakeStore([]))
    asyncio.run(mw.awrap_tool_call(FakeReq("web_search", journal), _ok))
    assert _applied_ctx.get()["a"] is True
    assert _applied_ctx.get()["b"] is None
    assert journal.events == [("skill.apply", {"skill_id": "a", "tool_name": "web_search"})]


def test_awrap_unrelated_tool_no_mark():
    _active_skills_ctx.set([("a", ("web_search",))])
    _applied_ctx.set({"a": None})
    journal = FakeJournal()
    mw = SkillMetricsMiddleware(FakeStore([]))
    asyncio.run(mw.awrap_tool_call(FakeReq("other_tool", journal), _ok))
    assert _applied_ctx.get()["a"] is None
    assert journal.events == []


def test_awrap_no_ctx_degradation_no_crash():
    # 无 injection 设 ctx → awrap 不标，不崩
    mw = SkillMetricsMiddleware(FakeStore([]))
    asyncio.run(mw.awrap_tool_call(FakeReq("web_search"), _ok))  # 不抛


def test_after_agent_tool_skill_hit_applied_true():
    _applied_ctx.set({"a": True})
    store = FakeStore([_rec("a", ("web_search",))])
    mw = SkillMetricsMiddleware(store)
    mw.after_agent({"metadata": {"active_skills": ["a"]}, "errors": [], "final_report": "x"}, _runtime())
    assert store.outcomes == [("a", True, True)]


def test_after_agent_tool_skill_not_hit_applied_false():
    _applied_ctx.set({"a": None})  # 未被 awrap 标
    store = FakeStore([_rec("a", ("web_search",))])
    mw = SkillMetricsMiddleware(store)
    mw.after_agent({"metadata": {"active_skills": ["a"]}, "errors": [], "final_report": "x"}, _runtime())
    assert store.outcomes == [("a", False, True)]  # tool-skill 有工具没用 → False


def test_after_agent_guidance_skill_applied_none():
    _applied_ctx.set({"b": None})
    store = FakeStore([_rec("b", ())])  # 无 allowed_tools = guidance
    mw = SkillMetricsMiddleware(store)
    mw.after_agent({"metadata": {"active_skills": ["b"]}, "errors": [], "final_report": "x"}, _runtime())
    assert store.outcomes == [("b", None, True)]  # guidance → None，不归因 completion/fallback


def test_after_agent_task_completed_hard_failure_false():
    _applied_ctx.set({"a": True})
    store = FakeStore([_rec("a", ("web_search",))])
    mw = SkillMetricsMiddleware(store)
    mw.after_agent(
        {"metadata": {"active_skills": ["a"]}, "errors": [{"kind": "failure"}], "final_report": "x"},
        _runtime(),
    )
    assert store.outcomes == [("a", True, False)]  # task_completed False


def test_after_agent_task_completed_chat_mode_no_report():
    _applied_ctx.set({"a": True})
    store = FakeStore([_rec("a", ("web_search",))])
    mw = SkillMetricsMiddleware(store)
    mw.after_agent({"metadata": {"active_skills": ["a"]}, "errors": [], "final_report": None}, _runtime())
    assert store.outcomes == [("a", True, True)]  # chat 宽松：无硬失败 → True


def test_after_agent_degradation_no_applied_ctx():
    _applied_ctx.set(None)  # 无 injection
    store = FakeStore([_rec("a", ("web_search",))])
    mw = SkillMetricsMiddleware(store)
    mw.after_agent({"metadata": {"active_skills": ["a"]}, "errors": [], "final_report": "x"}, _runtime())
    # applied None → tool-skill 有 allowed_tools 且 None → False
    assert store.outcomes == [("a", False, True)]


def test_after_agent_no_store_returns_none():
    mw = SkillMetricsMiddleware(None)
    assert mw.after_agent({"metadata": {"active_skills": ["a"]}, "errors": []}, _runtime()) is None


def test_after_agent_no_active_returns_none():
    mw = SkillMetricsMiddleware(FakeStore([]))
    assert mw.after_agent({"metadata": {}, "errors": [], "final_report": "x"}, _runtime()) is None


def test_after_agent_success_error_not_hard_failure():
    _applied_ctx.set({"a": True})
    store = FakeStore([_rec("a", ("web_search",))])
    mw = SkillMetricsMiddleware(store)
    # errors 含 success 条目（非硬失败）→ task_completed True
    mw.after_agent(
        {"metadata": {"active_skills": ["a"]}, "errors": [{"kind": "success"}], "final_report": "x"},
        _runtime(),
    )
    assert store.outcomes == [("a", True, True)]
