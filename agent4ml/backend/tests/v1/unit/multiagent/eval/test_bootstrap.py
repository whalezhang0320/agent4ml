"""L3 bootstrap 单测.

测试要点（L3 组件装配）:
- setup_l3 config.l3.enabled=false 返 None
- setup_l3 config.l3.enabled=true 构造 L3 组件 + 启动 daemon thread
- L3Setup 字段完整（bridge/registry/runtime_tracker/decision_log_writer/reader/facade/health_thread）
- registry 注册 3 adapter

注：setup_l2 集成测试需 langchain 环境（evolution/bootstrap.py import trigger_middleware
依赖 langchain），在此环境跳过，langchain 环境补测.
"""
from __future__ import annotations

import queue
from typing import Any

import pytest

from agent4ml.backend.agents.multiagent.config import L3Config, MultiAgentConfig
from agent4ml.backend.agents.multiagent.eval.bootstrap import L3Setup, setup_l3
from agent4ml.backend.agents.multiagent.evolution.metrics_view import (
    GlobalMetricsSnapshot,
    SpecialistMetricsSnapshot,
)
from agent4ml.backend.agents.multiagent.evolution.types import FailureCategory


class _MockMetricsStore:
    """Mock L1 MultiAgentMetricsStore（实现 MetricsView Protocol + _DecisionLogStore Protocol）."""

    def get_specialist_metrics(
        self, name: str, *, since: float | None = None,
    ) -> SpecialistMetricsSnapshot | None:
        return None

    def get_global_metrics(
        self, *, since: float | None = None,
    ) -> GlobalMetricsSnapshot:
        return {
            "total_calls": 0, "total_cost_usd": 0.0,
            "avg_latency_seconds": 0.0, "total_selections": 0,
            "total_completions": 0, "total_fallbacks": 0,
        }

    def get_failure_categories(
        self, *, since: float | None = None,
    ) -> dict[FailureCategory, int]:
        return {}

    def get_recent_failures(
        self, *, category: FailureCategory, limit: int = 10,
    ) -> list[Any]:
        return []

    def list_specialists(self) -> list[str]:
        return []

    def save_decision_log(self, record: Any) -> None:
        pass

    def get_decision_logs(
        self, specialist_name: str, failure_category: Any | None, limit: int,
    ) -> list[Any]:
        return []

    def archive_decision_logs(self, retention_days: int) -> int:
        return 0


class TestSetupL3:
    def test_disabled_returns_none(self):
        """config.l3.enabled=false 返 None."""
        config = MultiAgentConfig(l3=L3Config(enabled=False))
        result = setup_l3(config, _MockMetricsStore(), queue.Queue())
        assert result is None

    def test_enabled_returns_l3setup(self):
        """config.l3.enabled=true 构造 L3 组件 + 启动 daemon thread."""
        config = MultiAgentConfig(l3=L3Config(enabled=True))
        result = setup_l3(config, _MockMetricsStore(), queue.Queue())
        assert result is not None
        assert isinstance(result, L3Setup)
        assert result.bridge is not None
        assert result.registry is not None
        assert result.runtime_tracker is not None
        assert result.decision_log_writer is not None
        assert result.decision_log_reader is not None
        assert result.facade is not None
        assert result.health_thread is not None
        assert result.health_thread.daemon is True

    def test_registry_has_3_adapters(self):
        """registry 注册 3 adapter（programmatic/llm_judge/longitudinal_pairs）."""
        config = MultiAgentConfig(l3=L3Config(enabled=True))
        result = setup_l3(config, _MockMetricsStore(), queue.Queue())
        assert result is not None
        methods = result.registry.list_methods()
        assert "programmatic" in methods
        assert "llm_judge" in methods
        assert "longitudinal_pairs" in methods
