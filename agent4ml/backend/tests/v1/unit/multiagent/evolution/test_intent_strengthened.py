"""IntentEngineStrengthened 单测 — IntentTree 命中 + miss + llm=None + LLM 兜底 + 失败 fallback + candidate metadata 不含 capability_match + MVP llm=None 行为 + 不调 detect_and_dispatch.

设计（spec.md IntentEngineStrengthened Requirement + 42 文档 §7.10 + R6）:
- IntentTreeStrategy 薄包装 ReportIntentStrategy.match（不调 detect_and_dispatch 副作用）
- IntentEngineStrengthened 分层：IntentTree 优先 + confidence>=0.7 或 llm=None 返 tree + 否则调 llm + 失败 fallback
- candidate metadata 不含 capability_match（R6.2 修正，INV-35）
- MVP llm=None 行为（intent_llm_enabled=false）
"""
from __future__ import annotations

from pathlib import Path

import pytest

from agent4ml.backend.agents.multiagent.evolution.intent_strengthened import (
    IntentAnalysis,
    IntentEngineStrengthened,
    IntentTreeStrategy,
    LLMIntentStrategy,
)
from agent4ml.backend.agents.multiagent.evolution.metrics_view import (
    GlobalMetricsSnapshot,
    SpecialistMetricsSnapshot,
)
from agent4ml.backend.agents.multiagent.evolution.types import SpecialistCandidate


class _MockMetricsView:
    """测试用 MetricsView."""

    def __init__(self, specialists=None):
        self._specialists = specialists or {}

    def get_specialist_metrics(self, name, *, since=None):
        return self._specialists.get(name)

    def get_global_metrics(self, *, since=None):
        return GlobalMetricsSnapshot(
            total_calls=0, total_cost_usd=0.0, avg_latency_seconds=0.0,
            total_selections=0, total_completions=0, total_fallbacks=0,
        )

    def get_failure_categories(self, *, since=None):
        return {}

    def get_recent_failures(self, *, category, limit=10):
        return []

    def list_specialists(self):
        return list(self._specialists.keys())


def _make_tree(threshold=0.7):
    return IntentTreeStrategy(confidence_threshold=threshold)


# ── IntentTreeStrategy ───────────────────────────────────────────────────────


def test_tree_match_report_intent():
    """ReportIntentStrategy.match 命中 → IntentAnalysis.confidence>=0.7."""
    tree = _make_tree()
    result = tree.analyze("生成报告", ["codex", "claude"])
    assert result.confidence >= 0.7
    assert result.intent_type == "report"
    assert result.candidates == []


def test_tree_match_with_topic():
    """带 topic 的报告意图匹配."""
    tree = _make_tree()
    result = tree.analyze("生成报告 天气", ["codex"])
    assert result.confidence >= 0.7


def test_tree_miss_non_report():
    """非报告意图 → confidence=0.0."""
    tree = _make_tree()
    result = tree.analyze("帮我写代码", ["codex"])
    assert result.confidence == 0.0
    assert result.intent_type == "unknown"
    assert result.candidates == []


def test_tree_miss_how_to_write_report():
    """"如何写报告" 不匹配（^锚定，防误触发）."""
    tree = _make_tree()
    result = tree.analyze("如何写报告", ["codex"])
    assert result.confidence == 0.0


def test_tree_not_call_detect_and_dispatch():
    """IntentTreeStrategy 不调 detect_and_dispatch（避免 report action 副作用）."""
    import ast

    import agent4ml.backend.agents.multiagent.evolution.intent_strengthened as mod

    source = Path(mod.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    # 检查方法调用不含 detect_and_dispatch
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr == "detect_and_dispatch":
            pytest.fail("IntentTreeStrategy should not call detect_and_dispatch")
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Attribute) and func.attr == "detect_and_dispatch":
                pytest.fail("IntentTreeStrategy should not call detect_and_dispatch")


# ── IntentEngineStrengthened 分层 ───────────────────────────────────────────


def test_engine_tree_hit_returns_tree():
    """IntentTree 命中（confidence>=0.7）→ 返 tree."""
    engine = IntentEngineStrengthened(tree=_make_tree(), llm=None)
    result = engine.analyze("生成报告", ["codex"])
    assert result.confidence >= 0.7


def test_engine_miss_llm_none_returns_tree():
    """IntentTree miss + llm=None → 返 tree（fallback）."""
    engine = IntentEngineStrengthened(tree=_make_tree(), llm=None)
    result = engine.analyze("帮我写代码", ["codex"])
    assert result.confidence == 0.0  # tree miss


def test_engine_miss_llm_not_none_calls_llm():
    """IntentTree miss + llm 非 None → 调 llm."""

    class _FakeLLM:
        def analyze(self, user_message, available_specialists):
            return IntentAnalysis(confidence=0.9, candidates=["codex"], intent_type="coding")

    engine = IntentEngineStrengthened(tree=_make_tree(), llm=_FakeLLM())
    result = engine.analyze("帮我写代码", ["codex"])
    assert result.confidence == 0.9
    assert result.candidates == ["codex"]


def test_engine_llm_not_implemented_fallback_tree():
    """LLM 兜底抛 NotImplementedError → fallback tree（MVP 行为）."""
    engine = IntentEngineStrengthened(tree=_make_tree(), llm=LLMIntentStrategy())
    result = engine.analyze("帮我写代码", ["codex"])
    # LLMIntentStrategy.analyze 抛 NotImplementedError → fallback tree
    assert result.confidence == 0.0  # tree miss


def test_engine_llm_exception_fallback_tree():
    """LLM 兜底抛异常 → fallback tree."""
    class _ExceptionLLM:
        def analyze(self, user_message, available_specialists):
            raise RuntimeError("LLM call failed")

    engine = IntentEngineStrengthened(tree=_make_tree(), llm=_ExceptionLLM())
    result = engine.analyze("帮我写代码", ["codex"])
    assert result.confidence == 0.0  # fallback tree


# ── MVP llm=None 行为 ───────────────────────────────────────────────────────


def test_mvp_llm_none_analyze_always_returns_tree():
    """intent_llm_enabled=false（MVP 默认）→ llm=None → analyze 始终返 tree."""
    engine = IntentEngineStrengthened(tree=_make_tree(), llm=None)
    # 任意输入都返 tree（不调 LLM）
    assert engine.analyze("生成报告", ["codex"]).confidence >= 0.7
    assert engine.analyze("帮我写代码", ["codex"]).confidence == 0.0


def test_mvp_llm_not_instantiated():
    """MVP 阶段 LLMIntentStrategy 不实例化（intent_llm_enabled=false）."""
    engine = IntentEngineStrengthened(tree=_make_tree(), llm=None)
    assert engine._llm is None


# ── get_candidate_metadata ───────────────────────────────────────────────────


def test_candidate_metadata_no_capability_match():
    """get_candidate_metadata 不含 capability_match（R6.2 修正，INV-35）."""
    mv = _MockMetricsView(specialists={
        "codex": SpecialistMetricsSnapshot(
            specialist_name="codex",
            total_selections=10, total_invoked=10, total_completions=8,
            total_fallbacks=2, completion_rate=0.8,
            avg_cost_usd=0.45, avg_latency_seconds=180.0, sample_size=10,
        )
    })
    engine = IntentEngineStrengthened(tree=_make_tree(), llm=None)
    candidates = engine.get_candidate_metadata("写代码", ["codex"], mv)

    assert len(candidates) == 1
    c = candidates[0]
    fields = {f.name for f in __import__("dataclasses").fields(c)}
    assert "capability_match" not in fields
    assert "name" in fields
    assert "historical_success_rate" in fields
    assert "avg_cost_usd" in fields
    assert "avg_latency_seconds" in fields
    assert "sample_size" in fields


def test_candidate_metadata_from_metrics():
    """candidate metadata 数据来源 metrics_view.get_specialist_metrics."""
    mv = _MockMetricsView(specialists={
        "codex": SpecialistMetricsSnapshot(
            specialist_name="codex",
            total_selections=20, total_invoked=20, total_completions=16,
            total_fallbacks=4, completion_rate=0.8,
            avg_cost_usd=0.45, avg_latency_seconds=180.0, sample_size=20,
        ),
        "claude": SpecialistMetricsSnapshot(
            specialist_name="claude",
            total_selections=15, total_invoked=15, total_completions=13,
            total_fallbacks=2, completion_rate=0.87,
            avg_cost_usd=0.62, avg_latency_seconds=220.0, sample_size=15,
        ),
    })
    engine = IntentEngineStrengthened(tree=_make_tree(), llm=None)
    candidates = engine.get_candidate_metadata("写代码", ["codex", "claude"], mv)

    assert len(candidates) == 2
    codex = next(c for c in candidates if c.name == "codex")
    assert codex.historical_success_rate == 0.8
    assert codex.avg_cost_usd == 0.45
    assert codex.avg_latency_seconds == 180.0
    assert codex.sample_size == 20


def test_candidate_metadata_no_record_specialist():
    """无记录的 specialist 用默认值（sample_size=0，LLM 可判断可信度）."""
    mv = _MockMetricsView(specialists={})
    engine = IntentEngineStrengthened(tree=_make_tree(), llm=None)
    candidates = engine.get_candidate_metadata("写代码", ["new_specialist"], mv)

    assert len(candidates) == 1
    c = candidates[0]
    assert c.name == "new_specialist"
    assert c.historical_success_rate == 0.0
    assert c.avg_cost_usd == 0.0
    assert c.avg_latency_seconds == 0.0
    assert c.sample_size == 0


def test_candidate_metadata_empty_specialists():
    """空 specialist 列表 → 返空列表."""
    mv = _MockMetricsView()
    engine = IntentEngineStrengthened(tree=_make_tree(), llm=None)
    candidates = engine.get_candidate_metadata("写代码", [], mv)
    assert candidates == []


# ── LLMIntentStrategy MVP 不实现 ───────────────────────────────────────────


def test_llm_intent_strategy_not_implemented():
    """LLMIntentStrategy MVP 不实现 → 抛 NotImplementedError."""
    strategy = LLMIntentStrategy()
    with pytest.raises(NotImplementedError):
        strategy.analyze("test", ["codex"])


# ── IntentAnalysis frozen ───────────────────────────────────────────────────


def test_intent_analysis_frozen():
    import dataclasses
    a = IntentAnalysis(confidence=0.9, candidates=["codex"])
    with pytest.raises(dataclasses.FrozenInstanceError):
        a.confidence = 0.5


# ── 不作为 middleware ───────────────────────────────────────────────────────


def test_intent_engine_not_middleware():
    """IntentEngineStrengthened 不继承 AgentMiddleware（R6.5 架构修正）."""
    import inspect
    from langchain.agents.middleware.types import AgentMiddleware

    engine_cls = IntentEngineStrengthened
    bases = inspect.getmro(engine_cls)
    assert AgentMiddleware not in bases
