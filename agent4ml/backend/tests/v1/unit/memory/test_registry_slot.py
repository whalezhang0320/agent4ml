from __future__ import annotations

import dataclasses

import pytest

from agent4ml.backend.agents.capabilities.registry import (
    CapabilityMissingError,
    CapabilityRegistry,
)


class TestMemoryProviderSlot:
    def test_default_none(self) -> None:
        registry = CapabilityRegistry()
        assert registry.memory_provider is None

    def test_get_missing_raises(self) -> None:
        registry = CapabilityRegistry()
        with pytest.raises(CapabilityMissingError, match="memory_provider not registered"):
            registry.get_memory_provider()

    def test_get_returns_injected(self) -> None:
        mock_provider = object()
        registry = CapabilityRegistry(memory_provider=mock_provider)
        assert registry.get_memory_provider() is mock_provider

    def test_field_is_ninth_capability(self) -> None:
        """memory_provider 是第 9 个 capability（与 sandbox/skill/specialist/subagent 同级）。"""
        fields = [f.name for f in dataclasses.fields(CapabilityRegistry)]
        assert "memory_provider" in fields
        expected = [
            "models", "tools", "reporter", "artifact_store",
            "sandbox_provider", "skill_store", "specialist_registry",
            "subagent_provider", "memory_provider",
        ]
        assert fields == expected

    def test_frozen(self) -> None:
        registry = CapabilityRegistry()
        with pytest.raises(dataclasses.FrozenInstanceError):
            registry.memory_provider = "x"


class TestBackwardCompat:
    def test_existing_construction_without_memory_provider(self) -> None:
        """既有构造不传 memory_provider 时默认 None（既有测试兼容）。"""
        registry = CapabilityRegistry(
            models={"researcher": "m1"},
            tools={"t1": "tool"},
            reporter="reporter",
        )
        assert registry.memory_provider is None
        assert registry.get_model("researcher") == "m1"
        assert registry.get_tool("t1") == "tool"
        assert registry.get_reporter() == "reporter"

    def test_existing_getters_unaffected(self) -> None:
        """既有 getter 不受 memory_provider 字段影响。"""
        registry = CapabilityRegistry(
            models={"researcher": "m1"},
            sandbox_provider="sandbox",
            skill_store="skill",
            specialist_registry="spec",
            subagent_provider="sub",
        )
        assert registry.get_sandbox_provider() == "sandbox"
        assert registry.get_skill_store() == "skill"
        assert registry.get_specialist_registry() == "spec"
        assert registry.get_subagent_provider() == "sub"

    def test_all_getters_work_when_all_filled(self) -> None:
        registry = CapabilityRegistry(
            models={"researcher": "m1"},
            tools={"t1": "tool"},
            reporter="reporter",
            artifact_store="store",
            sandbox_provider="sandbox",
            skill_store="skill",
            specialist_registry="spec",
            subagent_provider="sub",
            memory_provider="mem",
        )
        assert registry.get_memory_provider() == "mem"
        assert registry.get_artifact_store() == "store"
