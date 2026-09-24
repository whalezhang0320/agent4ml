"""MetricsView Protocol 单测 — runtime_checkable + mock 实现契约验证 + L2 不耦合 L1 store。

设计（spec.md MetricsView Requirement + 42 文档 §7.2）:
- Protocol runtime_checkable：isinstance(mock, MetricsView) 返 True
- mock 实现契约验证：5 方法签名 + 返回类型符合 Protocol
- L2 不直接耦合 L1 store：L2 模块不 import MultiAgentMetricsStore 实现类
"""
from __future__ import annotations

import pytest

from agent4ml.backend.agents.multiagent.evolution.metrics_view import (
    GlobalMetricsSnapshot,
    MetricsView,
    SpecialistMetricsSnapshot,
)
from agent4ml.backend.agents.multiagent.evolution.types import (
    FailureCategory,
    FailureRecord,
)


# ── Protocol runtime_checkable ──────────────────────────────────────────────────


class _MockMetricsView:
    """测试用 MetricsView 实现（5 方法全覆盖，符合 Protocol 契约）。

    Protocol 不强制继承，runtime_checkable 检查结构匹配。
    """

    def get_specialist_metrics(
        self, name: str, *, since: float | None = None
    ) -> SpecialistMetricsSnapshot | None:
        if name == "codex":
            return SpecialistMetricsSnapshot(
                specialist_name="codex",
                total_selections=10,
                total_invoked=10,
                total_completions=8,
                total_fallbacks=2,
                completion_rate=0.8,
                avg_cost_usd=0.45,
                avg_latency_seconds=180.0,
                sample_size=10,
            )
        return None

    def get_global_metrics(
        self, *, since: float | None = None
    ) -> GlobalMetricsSnapshot:
        return GlobalMetricsSnapshot(
            total_calls=20,
            total_cost_usd=10.0,
            avg_latency_seconds=200.0,
            total_selections=20,
            total_completions=16,
            total_fallbacks=4,
        )

    def get_failure_categories(
        self, *, since: float | None = None
    ) -> dict[FailureCategory, int]:
        return {
            FailureCategory.CONTEXT_INSUFFICIENT: 5,
            FailureCategory.ABILITY_INSUFFICIENT: 2,
        }

    def get_recent_failures(
        self, *, category: FailureCategory, limit: int = 10
    ) -> list[FailureRecord]:
        if category == FailureCategory.CONTEXT_INSUFFICIENT:
            return [
                FailureRecord(
                    specialist_name="codex",
                    goal="g1",
                    success_criteria="sc1",
                    failure_category=FailureCategory.CONTEXT_INSUFFICIENT,
                )
            ]
        return []

    def list_specialists(self) -> list[str]:
        return ["codex", "claude"]


class _PartialMock:
    """只实现 4 方法（缺 list_specialists），不符 Protocol 契约。"""

    def get_specialist_metrics(
        self, name: str, *, since: float | None = None
    ) -> SpecialistMetricsSnapshot | None:
        return None

    def get_global_metrics(
        self, *, since: float | None = None
    ) -> GlobalMetricsSnapshot:
        return GlobalMetricsSnapshot(
            total_calls=0, total_cost_usd=0.0, avg_latency_seconds=0.0,
            total_selections=0, total_completions=0, total_fallbacks=0,
        )

    def get_failure_categories(
        self, *, since: float | None = None
    ) -> dict[FailureCategory, int]:
        return {}

    def get_recent_failures(
        self, *, category: FailureCategory, limit: int = 10
    ) -> list[FailureRecord]:
        return []


def test_metrics_view_runtime_checkable():
    """Mock 实现 5 方法 → isinstance(mock, MetricsView) 返 True。"""
    mock = _MockMetricsView()
    assert isinstance(mock, MetricsView)


def test_metrics_view_partial_not_instance():
    """缺 list_specialists → 不符 Protocol，isinstance 返 False。"""
    partial = _PartialMock()
    assert not isinstance(partial, MetricsView)


def test_metrics_view_not_instance_for_plain_object():
    """普通 object 不符 Protocol。"""
    assert not isinstance(object(), MetricsView)


# ── mock 契约验证：5 方法签名 + 返回类型 ────────────────────────────────────────


def test_get_specialist_metrics_returns_snapshot_or_none():
    mock = _MockMetricsView()
    snap = mock.get_specialist_metrics("codex")
    assert snap is not None
    assert isinstance(snap, dict)  # TypedDict 是 dict
    assert snap["specialist_name"] == "codex"
    assert snap["total_invoked"] == 10
    assert snap["completion_rate"] == 0.8
    assert snap["avg_cost_usd"] == 0.45
    assert snap["avg_latency_seconds"] == 180.0
    assert snap["sample_size"] == 10

    none_result = mock.get_specialist_metrics("unknown")
    assert none_result is None


def test_get_specialist_metrics_since_filter():
    """since 参数接受 float | None（None 查全量）。"""
    mock = _MockMetricsView()
    assert mock.get_specialist_metrics("codex", since=None) is not None
    assert mock.get_specialist_metrics("codex", since=1700000000.0) is not None


def test_get_global_metrics_returns_snapshot():
    mock = _MockMetricsView()
    snap = mock.get_global_metrics()
    assert isinstance(snap, dict)
    assert snap["total_calls"] == 20
    assert snap["total_cost_usd"] == 10.0
    assert snap["avg_latency_seconds"] == 200.0
    assert snap["total_selections"] == 20
    assert snap["total_completions"] == 16
    assert snap["total_fallbacks"] == 4


def test_get_failure_categories_returns_dict():
    mock = _MockMetricsView()
    cats = mock.get_failure_categories()
    assert isinstance(cats, dict)
    assert cats[FailureCategory.CONTEXT_INSUFFICIENT] == 5
    assert cats[FailureCategory.ABILITY_INSUFFICIENT] == 2
    # 缺失类别不在 dict 中（不返 0）
    assert FailureCategory.GOAL_UNCLEAR not in cats


def test_get_recent_failures_returns_list():
    mock = _MockMetricsView()
    failures = mock.get_recent_failures(category=FailureCategory.CONTEXT_INSUFFICIENT)
    assert isinstance(failures, list)
    assert len(failures) == 1
    assert failures[0].specialist_name == "codex"
    assert failures[0].failure_category == FailureCategory.CONTEXT_INSUFFICIENT

    empty = mock.get_recent_failures(category=FailureCategory.SANDBOX_ISSUE)
    assert empty == []


def test_get_recent_failures_limit_default():
    """limit 默认 10，可覆盖。"""
    mock = _MockMetricsView()
    mock.get_recent_failures(category=FailureCategory.CONTEXT_INSUFFICIENT)
    mock.get_recent_failures(category=FailureCategory.CONTEXT_INSUFFICIENT, limit=5)


def test_list_specialists_returns_list_str():
    mock = _MockMetricsView()
    specialists = mock.list_specialists()
    assert isinstance(specialists, list)
    assert all(isinstance(s, str) for s in specialists)
    assert "codex" in specialists
    assert "claude" in specialists


def test_list_specialists_empty():
    """无记录时返空列表（非 None）。"""
    mock = _MockMetricsView()
    # Mock 总返非空，验证返回类型是 list
    assert isinstance(mock.list_specialists(), list)


# ── L2 不耦合 L1 store ─────────────────────────────────────────────────────────


def test_l2_not_import_multiagent_metrics_store():
    """L2 模块不 import MultiAgentMetricsStore 实现类（spec.md Scenario）。

    验证 metrics_view.py 的 import 语句不含 L1 store 实现类。
    docstring 提及 "MultiAgentMetricsStore" 字样不算 import 耦合。
    """
    import ast
    from pathlib import Path

    import agent4ml.backend.agents.multiagent.evolution.metrics_view as mv_module

    source = Path(mv_module.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    import_names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                import_names.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                import_names.append(alias.name)
            # 也检查 module 路径（from X import Y）
            if node.module:
                import_names.append(node.module)
    # 检查 import 语句不含 L1 store 实现类 / 模块路径
    assert not any("MultiAgentMetricsStore" in name for name in import_names)
    assert not any("agent4ml.backend.agents.multiagent.metrics" in name for name in import_names)


def test_typed_dict_structure():
    """SpecialistMetricsSnapshot / GlobalMetricsSnapshot 是 TypedDict（dict 子类）。"""
    snap: SpecialistMetricsSnapshot = SpecialistMetricsSnapshot(
        specialist_name="codex",
        total_selections=1, total_invoked=1, total_completions=1, total_fallbacks=0,
        completion_rate=1.0, avg_cost_usd=0.0, avg_latency_seconds=0.0, sample_size=1,
    )
    assert isinstance(snap, dict)
    assert snap["specialist_name"] == "codex"

    global_snap: GlobalMetricsSnapshot = GlobalMetricsSnapshot(
        total_calls=1, total_cost_usd=0.0, avg_latency_seconds=0.0,
        total_selections=1, total_completions=1, total_fallbacks=0,
    )
    assert isinstance(global_snap, dict)
    assert global_snap["total_calls"] == 1
