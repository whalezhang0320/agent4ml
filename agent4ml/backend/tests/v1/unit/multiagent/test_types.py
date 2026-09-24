"""Multi-Agent types 单测 — frozen + 默认值 + ArtifactRef 结构。"""
from __future__ import annotations

import dataclasses

import pytest

from agent4ml.backend.agents.multiagent.types import (
    ArtifactRef,
    SpecialistCapabilities,
    SpecialistCapability,
    SpecialistRawResult,
    SpecialistRequest,
    SpecialistResult,
    SubagentRequest,
    SubagentResult,
    TokenUsage,
)


def test_specialist_request_frozen():
    req = SpecialistRequest(
        goal="g", success_criteria="sc", context_summary="cs",
        sandbox_id="sb1", artifacts_path="/p",
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        req.goal = "x"


def test_specialist_request_defaults():
    req = SpecialistRequest(
        goal="g", success_criteria="sc", context_summary="cs",
        sandbox_id=None, artifacts_path=None,
    )
    assert req.max_steps == 50
    assert req.timeout_seconds == 600
    assert req.allowed_tools == ()
    assert req.skill_injection is None


def test_specialist_result_frozen():
    r = SpecialistResult(specialist_name="codex", summary="ok")
    with pytest.raises(dataclasses.FrozenInstanceError):
        r.success = True


def test_specialist_result_defaults():
    r = SpecialistResult(specialist_name="codex", summary="ok")
    assert r.success is False
    assert r.artifacts == ()
    assert r.gap_analysis == ""
    assert r.usage is None
    assert r.error is None


def test_specialist_raw_result_frozen():
    raw = SpecialistRawResult(raw_output="out")
    with pytest.raises(dataclasses.FrozenInstanceError):
        raw.raw_output = "x"


def test_specialist_raw_result_defaults():
    raw = SpecialistRawResult(raw_output="out")
    assert raw.artifacts == ()
    assert raw.usage is None
    assert raw.duration_seconds == 0.0
    assert raw.exit_code == 0


def test_subagent_request_defaults_lower_than_specialist():
    """subagent max_steps/timeout 默认小于 specialist（leaf role，递归控制）。"""
    req = SubagentRequest(
        goal="g", success_criteria="sc", context_summary="cs",
        sandbox_id=None, artifacts_path=None,
    )
    assert req.max_steps == 20
    assert req.timeout_seconds == 300


def test_subagent_result_frozen():
    r = SubagentResult(summary="ok")
    with pytest.raises(dataclasses.FrozenInstanceError):
        r.summary = "x"


def test_artifact_ref_structure():
    ref = ArtifactRef(path="/p", artifact_type="code", specialist_name="codex")
    assert ref.path == "/p"
    assert ref.artifact_type == "code"
    assert ref.specialist_name == "codex"
    assert ref.description == ""
    assert ref.created_at is None


def test_artifact_ref_frozen():
    ref = ArtifactRef(path="/p", artifact_type="code", specialist_name="codex")
    with pytest.raises(dataclasses.FrozenInstanceError):
        ref.path = "/x"


def test_token_usage_defaults():
    u = TokenUsage()
    assert u.prompt_tokens == 0
    assert u.completion_tokens == 0
    assert u.total_tokens == 0


def test_specialist_capabilities_has():
    caps = SpecialistCapabilities(capabilities=(SpecialistCapability.CODING,))
    assert caps.has(SpecialistCapability.CODING)
    assert not caps.has(SpecialistCapability.RESEARCH)


def test_specialist_capabilities_empty_default():
    caps = SpecialistCapabilities()
    assert caps.capabilities == ()
    assert not caps.has(SpecialistCapability.CODING)
