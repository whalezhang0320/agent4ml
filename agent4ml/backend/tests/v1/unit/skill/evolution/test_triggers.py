"""E3 触发器单测 — Protocol + MetricMonitor(两阶段+anti-loop) + Capture(手动)。"""
from __future__ import annotations

from agent4ml.backend.agents.skill.evolution.protocols import (
    EvalBridge,
    FailureFocuser,
    Mutator,
    PromotionGate,
    Trigger,
)
from agent4ml.backend.agents.skill.evolution.triggers.capture_trigger import CaptureTrigger
from agent4ml.backend.agents.skill.evolution.triggers.metric_monitor import (
    MetricMonitorTrigger,
)
from agent4ml.backend.agents.skill.types import SkillLineage, SkillRecord


def _rec(
    name: str = "sv",
    selections: int = 10,
    applied: int = 0,
    completions: int = 0,
    fallbacks: int = 0,
    enabled: bool = True,
) -> SkillRecord:
    return SkillRecord(
        skill_id=f"{name}__imp", name=name, path="/p", content_hash="h",
        lineage=SkillLineage(generation=0, origin="IMPORTED"),
        total_selections=selections, total_applied=applied,
        total_completions=completions, total_fallbacks=fallbacks, enabled=enabled,
    )


class _FakeStore:
    def __init__(self, records):
        self._records = records

    def list_active(self):
        return list(self._records)


# ── Protocol runtime_checkable ──────────────────────────


def test_trigger_protocol_runtime_checkable():
    assert isinstance(MetricMonitorTrigger(), Trigger)
    assert isinstance(CaptureTrigger(), Trigger)


def test_all_protocols_runtime_checkable():
    """5 Protocol 均可 isinstance 检查（2b/L3 impl 注册时校验）。"""
    # 具体实现类在后续 batch，这里仅验证 Protocol 可用于 runtime check
    assert isinstance(MetricMonitorTrigger(), Trigger)


# ── MetricMonitor 触发条件 ──────────────────────────────


def test_metric_high_fallback_triggers_fix():
    """fallback_rate > 0.4 → FIX。"""
    rec = _rec("a", selections=10, fallbacks=5)  # fallback_rate=0.5
    store = _FakeStore([rec])
    trig = MetricMonitorTrigger(min_selections=5, cooldown_turns=0, llm=None)
    results = trig.should_trigger(store)
    assert len(results) == 1
    assert results[0].evolution_type == "FIX"
    assert results[0].trigger == "METRIC"
    assert results[0].target_skill is rec


def test_metric_low_completion_high_applied_triggers_fix():
    """applied_rate>0.4 且 completion_rate<0.35 → FIX。"""
    rec = _rec("a", selections=10, applied=5, completions=1)  # applied 0.5, completion 0.2
    store = _FakeStore([rec])
    trig = MetricMonitorTrigger(min_selections=5, cooldown_turns=0, llm=None)
    results = trig.should_trigger(store)
    assert len(results) == 1
    assert results[0].evolution_type == "FIX"


def test_metric_derived_skipped_in_2a():
    """effective<0.55 且 applied>0.25 → DERIVED，2a 跳过（留 2b）。"""
    rec = _rec("a", selections=10, applied=3, completions=2)  # eff 0.2, applied 0.3
    store = _FakeStore([rec])
    trig = MetricMonitorTrigger(min_selections=5, cooldown_turns=0, llm=None)
    results = trig.should_trigger(store)
    assert len(results) == 0  # DERIVED 2a 不产


def test_metric_healthy_not_trigger():
    """健康 skill 不触发。"""
    rec = _rec("a", selections=10, applied=8, completions=7)  # eff 0.7, fallback 0
    store = _FakeStore([rec])
    trig = MetricMonitorTrigger(min_selections=5, cooldown_turns=0, llm=None)
    assert trig.should_trigger(store) == []


# ── anti-loop ───────────────────────────────────────────


def test_metric_anti_loop_min_selections():
    """total_selections < min 不触发（新进化 skill selections=0）。"""
    rec = _rec("a", selections=3, fallbacks=2)  # fallback 0.67 但 selections<5
    store = _FakeStore([rec])
    trig = MetricMonitorTrigger(min_selections=5, cooldown_turns=0, llm=None)
    assert trig.should_trigger(store) == []


def test_metric_anti_loop_cooldown():
    """自上次进化需 cooldown_turns 次新 selections 才重评。"""
    rec = _rec("a", selections=10, fallbacks=5)
    store = _FakeStore([rec])
    trig = MetricMonitorTrigger(min_selections=5, cooldown_turns=10, llm=None)
    # 首次触发
    assert len(trig.should_trigger(store)) == 1
    trig.mark_evolved("a", 10)
    # 刚进化，selections 未增 → cooldown 不过
    assert trig.should_trigger(store) == []
    # selections 增到 25 → 25-10=15 >= 10 → 再触发
    rec2 = _rec("a", selections=25, fallbacks=13)
    store2 = _FakeStore([rec2])
    assert len(trig.should_trigger(store2)) == 1


def test_metric_disabled_skill_skipped():
    """disabled skill 不触发。"""
    rec = _rec("a", selections=10, fallbacks=5, enabled=False)
    store = _FakeStore([rec])
    trig = MetricMonitorTrigger(min_selections=5, cooldown_turns=0, llm=None)
    assert trig.should_trigger(store) == []


# ── LLM 确认（Phase 2） ─────────────────────────────────


class _FakeLLM:
    def __init__(self, answer: str):
        self._answer = answer

    def invoke(self, messages):
        return type("R", (), {"content": self._answer})()


def test_metric_llm_confirm_yes():
    rec = _rec("a", selections=10, fallbacks=5)
    store = _FakeStore([rec])
    trig = MetricMonitorTrigger(min_selections=5, cooldown_turns=0, llm=_FakeLLM("yes"))
    assert len(trig.should_trigger(store)) == 1


def test_metric_llm_confirm_no_skips():
    """LLM 确认返 no → 不触发（过滤误报）。"""
    rec = _rec("a", selections=10, fallbacks=5)
    store = _FakeStore([rec])
    trig = MetricMonitorTrigger(min_selections=5, cooldown_turns=0, llm=_FakeLLM("no"))
    assert trig.should_trigger(store) == []


def test_metric_llm_failure_conservative_skip():
    """LLM 调用失败 → 保守不触发。"""
    class _BoomLLM:
        def invoke(self, messages):
            raise RuntimeError("boom")
    rec = _rec("a", selections=10, fallbacks=5)
    store = _FakeStore([rec])
    trig = MetricMonitorTrigger(min_selections=5, cooldown_turns=0, llm=_BoomLLM())
    assert trig.should_trigger(store) == []


# ── CaptureTrigger ──────────────────────────────────────


def test_capture_should_trigger_empty_in_2a():
    """2a 自动信号未实现，should_trigger 返空。"""
    trig = CaptureTrigger()
    assert trig.should_trigger(_FakeStore([])) == []


def test_capture_manual_capture_produces_context():
    """手动 capture 产 CAPTURED context。"""
    trig = CaptureTrigger()
    ctx = trig.manual_capture("信源交叉核验", "source-cross-check")
    assert ctx.trigger == "CAPTURE"
    assert ctx.evolution_type == "CAPTURED"
    assert ctx.target_skill is None
    assert ctx.capture_pattern == "信源交叉核验"
    assert ctx.suggested_name == "source-cross-check"


def test_capture_is_trigger_protocol():
    assert isinstance(CaptureTrigger(), Trigger)
