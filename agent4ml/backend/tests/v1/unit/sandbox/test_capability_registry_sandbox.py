from __future__ import annotations

import dataclasses

import pytest

from agent4ml.backend.agents.capabilities.registry import (
    CapabilityMissingError,
    CapabilityRegistry,
)


class TestSandboxProviderSlot:
    def test_default_none(self) -> None:
        registry = CapabilityRegistry()
        assert registry.sandbox_provider is None

    def test_get_sandbox_provider_missing_raises(self) -> None:
        registry = CapabilityRegistry()
        with pytest.raises(CapabilityMissingError, match="sandbox_provider not registered"):
            registry.get_sandbox_provider()

    def test_get_sandbox_provider_returns_injected(self) -> None:
        mock_provider = object()
        registry = CapabilityRegistry(sandbox_provider=mock_provider)
        assert registry.get_sandbox_provider() is mock_provider

    def test_frozen_immutable(self) -> None:
        registry = CapabilityRegistry()
        with pytest.raises(dataclasses.FrozenInstanceError):
            registry.sandbox_provider = object()  # type: ignore[misc]


class TestBackwardCompat:
    def test_existing_get_model_works(self) -> None:
        mock_model = object()
        registry = CapabilityRegistry(models={"researcher": mock_model})
        assert registry.get_model("researcher") is mock_model

    def test_existing_get_model_missing_raises(self) -> None:
        registry = CapabilityRegistry()
        with pytest.raises(CapabilityMissingError):
            registry.get_model("unknown")

    def test_existing_get_reporter_works(self) -> None:
        mock_reporter = object()
        registry = CapabilityRegistry(reporter=mock_reporter)
        assert registry.get_reporter() is mock_reporter

    def test_existing_get_artifact_store_works(self) -> None:
        mock_store = object()
        registry = CapabilityRegistry(artifact_store=mock_store)
        assert registry.get_artifact_store() is mock_store

    def test_construct_with_all_fields(self) -> None:
        mock_provider = object()
        mock_model = object()
        registry = CapabilityRegistry(
            models={"m": mock_model},
            tools={},
            reporter=None,
            artifact_store=None,
            sandbox_provider=mock_provider,
        )
        assert registry.get_sandbox_provider() is mock_provider
        assert registry.get_model("m") is mock_model
