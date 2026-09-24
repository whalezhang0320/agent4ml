"""E4 IVEFocuser 单测 — 5 问诊断 + impl 累计升级 + LLM=None 降级 + CAPTURED 透传。"""
from __future__ import annotations

import json

from agent4ml.backend.agents.skill.evolution.focus.ive_focuser import IVEFocuser
from agent4ml.backend.agents.skill.evolution.types import EvolutionContext, FailureEvidence
from agent4ml.backend.agents.skill.types import SkillLineage, SkillRecord


def _skill(name: str = "sv") -> SkillRecord:
    return SkillRecord(
        skill_id=f"{name}__imp", name=name, path="/p", content_hash="h",
        lineage=SkillLineage(generation=0, origin="IMPORTED"),
        description="信源验证",
    )


def _evidence(desc: str = "工具调用失败") -> FailureEvidence:
    return FailureEvidence(
        turn_index=2, tool_name="web_search",
        failure_class="IMPLEMENTATION", description=desc,
    )


def _fix_ctx(evidence=None) -> EvolutionContext:
    return EvolutionContext(
        trigger="METRIC", evolution_type="FIX",
        target_skill=_skill(), failure_evidence=evidence or (_evidence(),),
    )


class _FakeLLM:
    def __init__(self, content: str):
        self._content = content

    def invoke(self, messages):
        return type("R", (), {"content": self._content})()


# ── CAPTURED 透传 ───────────────────────────────────────


def test_captured_no_evidence_passthrough():
    """CAPTURED 无失败证据，直接返原 ctx（沉淀非修复）。"""
    ctx = EvolutionContext(
        trigger="CAPTURE", evolution_type="CAPTURED", target_skill=None,
        capture_pattern="模式", suggested_name="new-skill",
    )
    focuser = IVEFocuser(llm=None)
    result = focuser.focus(ctx, store=None)
    assert result is ctx  # 透传


# ── LLM=None 降级 ───────────────────────────────────────


def test_llm_none_degrades_to_summary():
    """LLM=None 降级全量摘要，默认 IMPLEMENTATION。"""
    ctx = _fix_ctx((_evidence("失败A"), _evidence("失败B")))
    focuser = IVEFocuser(llm=None)
    result = focuser.focus(ctx, store=None)
    assert "失败A" in result.fix_direction
    assert "失败B" in result.fix_direction
    assert result.failure_evidence[0].failure_class == "IMPLEMENTATION"


# ── LLM 5 问诊断 ────────────────────────────────────────


def test_llm_diagnose_implementation():
    """LLM 返 IMPLEMENTATION → focus 更新 failure_class。"""
    ctx = _fix_ctx()
    llm = _FakeLLM(json.dumps({"class": "IMPLEMENTATION", "direction": "微调步骤2措辞"}))
    focuser = IVEFocuser(llm=llm)
    result = focuser.focus(ctx, store=None)
    assert result.failure_evidence[0].failure_class == "IMPLEMENTATION"
    assert result.fix_direction == "微调步骤2措辞"


def test_llm_diagnose_fundamental():
    """LLM 返 FUNDAMENTAL → focus 更新 + 重置 impl 计数。"""
    ctx = _fix_ctx()
    llm = _FakeLLM(json.dumps({"class": "FUNDAMENTAL", "direction": "skill 方向错，需重写"}))
    focuser = IVEFocuser(llm=llm)
    result = focuser.focus(ctx, store=None)
    assert result.failure_evidence[0].failure_class == "FUNDAMENTAL"
    assert result.fix_direction == "skill 方向错，需重写"


# ── implementation 累计升级 ─────────────────────────────


def test_impl_accumulate_upgrade_to_fundamental():
    """implementation 累计 3 次升级 fundamental。"""
    llm = _FakeLLM(json.dumps({"class": "IMPLEMENTATION", "direction": "微调"}))
    focuser = IVEFocuser(llm=llm, impl_fail_threshold=3)

    # 第 1 次：IMPLEMENTATION
    r1 = focuser.focus(_fix_ctx(), store=None)
    assert r1.failure_evidence[0].failure_class == "IMPLEMENTATION"
    assert r1.failure_evidence[0].impl_fail_count == 1

    # 第 2 次：仍 IMPLEMENTATION
    r2 = focuser.focus(_fix_ctx(), store=None)
    assert r2.failure_evidence[0].failure_class == "IMPLEMENTATION"
    assert r2.failure_evidence[0].impl_fail_count == 2

    # 第 3 次：升级 FUNDAMENTAL
    r3 = focuser.focus(_fix_ctx(), store=None)
    assert r3.failure_evidence[0].failure_class == "FUNDAMENTAL"
    assert "升级 fundamental" in r3.fix_direction


def test_fundamental_resets_impl_count():
    """FUNDAMENTAL 重置 implementation 计数。"""
    focuser = IVEFocuser(llm=_FakeLLM(json.dumps({"class": "IMPLEMENTATION"})), impl_fail_threshold=3)
    focuser.focus(_fix_ctx(), store=None)  # impl=1
    focuser.focus(_fix_ctx(), store=None)  # impl=2

    # FUNDAMENTAL
    focuser2 = IVEFocuser(llm=_FakeLLM(json.dumps({"class": "FUNDAMENTAL"})), impl_fail_threshold=3)
    # 复用计数（同 skill_name "sv"）—— 需同 focuser 实例
    focuser._llm = _FakeLLM(json.dumps({"class": "FUNDAMENTAL", "direction": "重写"}))
    focuser.focus(_fix_ctx(), store=None)
    assert focuser._impl_fail_counts.get("sv") == 0


# ── LLM 失败保守 ────────────────────────────────────────


def test_llm_failure_conservative_implementation():
    """LLM 调用失败 → 保守返 IMPLEMENTATION（不轻易判 fundamental 杀 skill）。"""
    class _BoomLLM:
        def invoke(self, messages):
            raise RuntimeError("boom")
    ctx = _fix_ctx()
    focuser = IVEFocuser(llm=_BoomLLM())
    result = focuser.focus(ctx, store=None)
    assert result.failure_evidence[0].failure_class == "IMPLEMENTATION"


def test_llm_invalid_json_conservative():
    """LLM 返非 JSON → 保守 IMPLEMENTATION。"""
    ctx = _fix_ctx()
    focuser = IVEFocuser(llm=_FakeLLM("not json at all"))
    result = focuser.focus(ctx, store=None)
    assert result.failure_evidence[0].failure_class == "IMPLEMENTATION"


def test_llm_invalid_class_value_defaults():
    """LLM 返非法 class 值 → 默认 IMPLEMENTATION。"""
    ctx = _fix_ctx()
    llm = _FakeLLM(json.dumps({"class": "UNKNOWN", "direction": "d"}))
    focuser = IVEFocuser(llm=llm)
    result = focuser.focus(ctx, store=None)
    assert result.failure_evidence[0].failure_class == "IMPLEMENTATION"
