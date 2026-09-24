"""Skill 数据模型单测（B1a）— frozen + rate property 边界。"""
from __future__ import annotations

import dataclasses

import pytest

from agent4ml.backend.agents.skill.types import (
    SkillHealth,
    SkillLineage,
    SkillMetrics,
    SkillRecord,
)


def test_skill_lineage_defaults():
    lin = SkillLineage()
    assert lin.parent_skill_ids == ()
    assert lin.generation == 0
    assert lin.origin == "IMPORTED"
    assert lin.version_hash == ""
    assert lin.created_by is None


def test_skill_lineage_fixed_origin():
    lin = SkillLineage(parent_skill_ids=("prev__v0_x",), generation=1, origin="FIXED")
    assert lin.parent_skill_ids == ("prev__v0_x",)
    assert lin.origin == "FIXED"


def test_skill_record_frozen():
    rec = SkillRecord(skill_id="s1", name="n", path="/p", content_hash="h")
    with pytest.raises(dataclasses.FrozenInstanceError):
        rec.name = "x"


def test_skill_record_required_fields():
    rec = SkillRecord(skill_id="s1", name="n", path="/p", content_hash="h")
    assert rec.is_active is True
    assert rec.lineage == SkillLineage()
    assert rec.allowed_tools == ()
    assert rec.enabled is True
    assert rec.total_selections == 0


def test_skill_record_rate_zero_division_safe():
    rec = SkillRecord(skill_id="s1", name="n", path="/p", content_hash="h")
    assert rec.applied_rate == 0.0
    assert rec.completion_rate == 0.0
    assert rec.effective_rate == 0.0
    assert rec.fallback_rate == 0.0


def test_skill_record_rate_values():
    rec = SkillRecord(
        skill_id="s1",
        name="n",
        path="/p",
        content_hash="h",
        total_selections=10,
        total_applied=8,
        total_completions=6,
        total_fallbacks=2,
    )
    assert rec.applied_rate == 0.8
    assert rec.completion_rate == 0.75
    assert rec.effective_rate == 0.6
    assert rec.fallback_rate == 0.2


def test_skill_metrics_frozen():
    m = SkillMetrics(
        skill_id="s", selections=1, applied=1, completions=1, fallbacks=0,
        applied_rate=1.0, completion_rate=1.0, effective_rate=1.0, fallback_rate=0.0,
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        m.selections = 2


def test_skill_health_frozen():
    h = SkillHealth(
        skill_id="s", name="n", effective_rate=0.1, fallback_rate=0.5,
        total_selections=10, degraded=True,
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        h.degraded = False
