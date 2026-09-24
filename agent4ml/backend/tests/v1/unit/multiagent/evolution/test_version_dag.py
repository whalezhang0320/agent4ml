"""VersionDAG 单测 — commit + get_active + get_history + rollback + hash 防环 + is_active 单指针 + 持久化跨 session.

设计（spec.md VersionDAG Requirement + 42 文档 §7.8 + R1）:
- commit 写 DB + is_active 单指针更新（同 template_id 仅 1 行 active）
- get_active 每次查 DB（不缓存，保 hot swap，INV-12）
- rollback is_active 指针回退
- hash_exists_in_recent 近 5 版防环（INV-27）
- reject candidate 也存（防重复尝试）
- 持久化跨 session（SQLite）
- 演化失败保持旧 is_active（INV-13）
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import pytest

from agent4ml.backend.agents.multiagent.evolution.types import (
    ContextSummaryTemplate,
    SkillInjectionTemplate,
)
from agent4ml.backend.agents.multiagent.evolution.version_dag import VersionDAG


@dataclass(frozen=True)
class _FakeEvalResult:
    """测试用 EvalResult（与 VersionDAG.commit 的 eval_result 参数兼容）."""

    candidate_score: float = 0.8
    baseline_score: float = 0.4
    ci_low: float = 0.6
    ci_high: float = 0.88
    sample_size: int = 15
    success: bool = True


def _make_template(version: str = "v1", skeleton: str = "s1") -> ContextSummaryTemplate:
    return ContextSummaryTemplate(
        version=version, template_id="default",
        extractors=(), filters=(), max_tokens=2000, prompt_skeleton=skeleton,
    )


def _make_skill_template(version: str = "v1") -> SkillInjectionTemplate:
    return SkillInjectionTemplate(
        version=version, template_id="default",
        skill_selector=_DummySelector(), injection_format="fmt",
    )


class _DummySelector:
    def select(self, goal, available_skills):
        return ()


@pytest.fixture
def dag(tmp_path):
    """每个测试用独立 db 文件（tmp_path fixture 隔离）."""
    db_path = str(tmp_path / "test_multiagent.db")
    return VersionDAG(db_path=db_path)


# ── commit + get_active ───────────────────────────────────────────────────────


def test_commit_and_get_active(dag):
    """commit artifact → get_active 返回该 artifact（is_active=1）."""
    t1 = _make_template(version="v1")
    art_id = dag.commit(t1, _FakeEvalResult(), decision="accept")
    assert art_id.startswith("art_")

    active = dag.get_active(ContextSummaryTemplate)
    assert active is not None
    assert active.version == "v1"
    assert active.template_id == "default"


def test_commit_updates_is_active_single_pointer(dag):
    """同 template_id 仅 1 行 is_active=1（INV-7）."""
    t1 = _make_template(version="v1", skeleton="s1")
    t2 = _make_template(version="v2", skeleton="s2")

    dag.commit(t1, _FakeEvalResult(), decision="accept")
    dag.commit(t2, _FakeEvalResult(), decision="accept")

    history = dag.get_history(ContextSummaryTemplate, "default")
    assert len(history) == 2
    active_rows = [r for r in history if r.is_active]
    assert len(active_rows) == 1  # 仅 1 行 active
    assert active_rows[0].version == "v2"  # 最新 commit 是 active


def test_get_active_returns_none_when_empty(dag):
    """无记录时 get_active 返 None."""
    assert dag.get_active(ContextSummaryTemplate) is None
    assert dag.get_active(SkillInjectionTemplate) is None


def test_get_active_queries_db_each_call_not_cached(dag):
    """get_active 每次查 DB（不缓存，保 hot swap，INV-12）."""
    t1 = _make_template(version="v1")
    dag.commit(t1, _FakeEvalResult(), decision="accept")

    # 第一次查
    active1 = dag.get_active(ContextSummaryTemplate)
    assert active1 is not None
    assert active1.version == "v1"

    # commit v2
    t2 = _make_template(version="v2", skeleton="s2")
    dag.commit(t2, _FakeEvalResult(), decision="accept")

    # 第二次查 → 立即看到 v2（不缓存）
    active2 = dag.get_active(ContextSummaryTemplate)
    assert active2 is not None
    assert active2.version == "v2"  # hot swap 生效


# ── reject candidate 也存（防重复尝试）──────────────────────────────────────


def test_reject_candidate_stored(dag):
    """reject candidate 也存（is_active 不更新，防重复尝试）."""
    t1 = _make_template(version="v1")
    dag.commit(t1, _FakeEvalResult(), decision="accept")

    t2 = _make_template(version="v2", skeleton="s2")
    dag.commit(t2, _FakeEvalResult(), decision="reject")

    history = dag.get_history(ContextSummaryTemplate, "default")
    assert len(history) == 2  # v1 + v2 都存
    # v1 仍是 active（reject 不更新 is_active）
    active = [r for r in history if r.is_active]
    assert len(active) == 1
    assert active[0].version == "v1"


# ── rollback ───────────────────────────────────────────────────────────────────


def test_rollback_is_active_pointer(dag):
    """rollback is_active 指针回退到指定 artifact."""
    t1 = _make_template(version="v1", skeleton="s1")
    t2 = _make_template(version="v2", skeleton="s2")
    t3 = _make_template(version="v3", skeleton="s3")

    id1 = dag.commit(t1, _FakeEvalResult(), decision="accept")
    dag.commit(t2, _FakeEvalResult(), decision="accept")
    dag.commit(t3, _FakeEvalResult(), decision="accept")

    # 当前 active 是 v3
    active = dag.get_active(ContextSummaryTemplate)
    assert active is not None
    assert active.version == "v3"

    # rollback 到 v1
    dag.rollback(id1)
    active = dag.get_active(ContextSummaryTemplate)
    assert active is not None
    assert active.version == "v1"


def test_rollback_nonexistent_artifact_no_op(dag):
    """rollback 不存在的 artifact_id → 无操作（不报错）."""
    dag.rollback("nonexistent_id")
    # 无 active（因为没 commit 过）
    assert dag.get_active(ContextSummaryTemplate) is None


# ── hash_exists_in_recent 防环 ─────────────────────────────────────────────────


def test_hash_exists_in_recent_window_5(dag):
    """近 5 版内 hash 命中 → 返 True（防环，INV-27）."""
    # commit 5 个版本
    for i in range(5):
        t = _make_template(version=f"v{i+1}", skeleton=f"s{i+1}")
        dag.commit(t, _FakeEvalResult(), decision="accept")

    # 取最新版本的 hash，查近 5 版应命中
    history = dag.get_history(ContextSummaryTemplate, "default")
    latest_hash = history[0].artifact_hash
    assert dag.hash_exists_in_recent(latest_hash, window=5) is True


def test_hash_not_in_recent_beyond_window(dag):
    """超过 5 版窗口 → 返 False."""
    # commit 6 个版本（hash 各不同）
    for i in range(6):
        t = _make_template(version=f"v{i+1}", skeleton=f"s{i+1}")
        dag.commit(t, _FakeEvalResult(), decision="accept")

    # 取第 6 版（最旧）hash，查近 5 版应不命中
    history = dag.get_history(ContextSummaryTemplate, "default")
    # history 按 created_at DESC，最旧是最后一个
    oldest_hash = history[-1].artifact_hash
    assert dag.hash_exists_in_recent(oldest_hash, window=5) is False


def test_hash_exists_empty_db_returns_false(dag):
    """空 DB → hash_exists_in_recent 返 False."""
    assert dag.hash_exists_in_recent("any_hash", window=5) is False


# ── 持久化跨 session ─────────────────────────────────────────────────────────


def test_persistence_across_sessions(dag, tmp_path):
    """进程重启后 get_active 返上次 commit 的 is_active 模板（SQLite 持久化）."""
    t1 = _make_template(version="v1", skeleton="persistent")
    dag.commit(t1, _FakeEvalResult(), decision="accept")

    # 新建 VersionDAG 实例（模拟进程重启）
    dag2 = VersionDAG(db_path=dag._db_path)
    active = dag2.get_active(ContextSummaryTemplate)
    assert active is not None
    assert active.version == "v1"
    assert active.prompt_skeleton == "persistent"


# ── get_history ───────────────────────────────────────────────────────────────


def test_get_history_returns_all_versions(dag):
    """get_history 返同 template_id 所有版本（按 created_at 降序）."""
    t1 = _make_template(version="v1", skeleton="s1")
    t2 = _make_template(version="v2", skeleton="s2")
    t3 = _make_template(version="v3", skeleton="s3")

    dag.commit(t1, _FakeEvalResult(), decision="accept")
    dag.commit(t2, _FakeEvalResult(), decision="reject")
    dag.commit(t3, _FakeEvalResult(), decision="accept")

    history = dag.get_history(ContextSummaryTemplate, "default")
    assert len(history) == 3
    # 最新在前
    assert history[0].version == "v3"
    assert history[1].version == "v2"
    assert history[2].version == "v1"


def test_get_history_empty_template_id(dag):
    """不存在的 template_id → 返空列表."""
    history = dag.get_history(ContextSummaryTemplate, "nonexistent")
    assert history == []


# ── SkillInjectionTemplate ─────────────────────────────────────────────────────


def test_commit_and_get_active_skill_injection(dag):
    """SkillInjectionTemplate commit + get_active."""
    t1 = _make_skill_template(version="v1")
    dag.commit(t1, _FakeEvalResult(), decision="accept")

    active = dag.get_active(SkillInjectionTemplate)
    assert active is not None
    assert active.version == "v1"
    assert active.template_id == "default"


def test_context_summary_and_skill_injection_independent(dag):
    """ContextSummaryTemplate 和 SkillInjectionTemplate is_active 独立."""
    ctx_t = _make_template(version="v1")
    skill_t = _make_skill_template(version="v1")
    dag.commit(ctx_t, _FakeEvalResult(), decision="accept")
    dag.commit(skill_t, _FakeEvalResult(), decision="accept")

    ctx_active = dag.get_active(ContextSummaryTemplate)
    skill_active = dag.get_active(SkillInjectionTemplate)
    assert ctx_active is not None
    assert skill_active is not None
    assert ctx_active.version == "v1"
    assert skill_active.version == "v1"


# ── 演化失败保持旧 is_active（INV-13）──────────────────────────────────────


def test_failed_decision_keeps_old_is_active(dag):
    """decision=failed 时 is_active 不更新（保持旧 active，INV-13）."""
    t1 = _make_template(version="v1", skeleton="s1")
    dag.commit(t1, _FakeEvalResult(), decision="accept")

    t2 = _make_template(version="v2", skeleton="s2")
    dag.commit(t2, _FakeEvalResult(), decision="failed")

    active = dag.get_active(ContextSummaryTemplate)
    assert active is not None
    assert active.version == "v1"  # 仍是 v1（failed 不更新）
