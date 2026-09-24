"""CapabilityRegistry multiagent 扩展单测 — specialist_registry + subagent_provider slot。"""
from __future__ import annotations

import dataclasses

import pytest

from agent4ml.backend.agents.capabilities.registry import (
    CapabilityMissingError,
    CapabilityRegistry,
)


def test_default_specialist_registry_none():
    reg = CapabilityRegistry()
    assert reg.specialist_registry is None


def test_default_subagent_provider_none():
    reg = CapabilityRegistry()
    assert reg.subagent_provider is None


def test_get_specialist_registry_missing_raises():
    reg = CapabilityRegistry()
    with pytest.raises(CapabilityMissingError, match="specialist_registry not registered"):
        reg.get_specialist_registry()


def test_get_subagent_provider_missing_raises():
    reg = CapabilityRegistry()
    with pytest.raises(CapabilityMissingError, match="subagent_provider not registered"):
        reg.get_subagent_provider()


def test_get_specialist_registry_returns_injected():
    sentinel = object()
    reg = CapabilityRegistry(specialist_registry=sentinel)
    assert reg.get_specialist_registry() is sentinel


def test_get_subagent_provider_returns_injected():
    sentinel = object()
    reg = CapabilityRegistry(subagent_provider=sentinel)
    assert reg.get_subagent_provider() is sentinel


def test_existing_api_not_broken():
    """回归：加新 slot 后既有 API 不破坏。"""
    reg = CapabilityRegistry(
        models={"m1": object()},
        tools={"t1": object()},
        reporter=object(),
        artifact_store=object(),
        sandbox_provider=object(),
        skill_store=object(),
    )
    assert reg.get_model("m1") is not None
    assert reg.get_tool("t1") is not None
    assert reg.get_reporter() is not None
    assert reg.get_artifact_store() is not None
    assert reg.get_sandbox_provider() is not None
    assert reg.get_skill_store() is not None


def test_registry_frozen():
    """frozen dataclass：不可重新赋值 specialist_registry。"""
    reg = CapabilityRegistry()
    with pytest.raises(dataclasses.FrozenInstanceError):
        reg.specialist_registry = object()


def test_registry_construct_without_new_fields():
    """既有构造方式不破坏（不传新 slot）。"""
    reg = CapabilityRegistry(models={}, tools={})
    assert reg.specialist_registry is None
    assert reg.subagent_provider is None


def test_registry_construct_with_all_slots():
    """全 slot 构造 OK。"""
    reg = CapabilityRegistry(
        models={},
        tools={},
        reporter=None,
        artifact_store=None,
        sandbox_provider=None,
        skill_store=None,
        specialist_registry=object(),
        subagent_provider=object(),
    )
    assert reg.get_specialist_registry() is not None
    assert reg.get_subagent_provider() is not None
