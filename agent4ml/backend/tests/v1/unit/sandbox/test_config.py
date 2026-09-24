from __future__ import annotations

import dataclasses

import pytest

from agent4ml.backend.agents.sandbox.integration.config import (
    STARTUP_ONLY_FIELDS,
    SandboxConfig,
    SandboxMountConfig,
)


class TestSandboxConfig:
    def test_default_not_enabled(self) -> None:
        config = SandboxConfig()
        assert config.use == ""

    def test_default_allow_host_bash_true(self) -> None:
        config = SandboxConfig()
        assert config.allow_host_bash is True

    def test_default_mounts_empty(self) -> None:
        config = SandboxConfig()
        assert config.mounts == []

    def test_default_environment_empty(self) -> None:
        config = SandboxConfig()
        assert config.environment == {}

    def test_enabled_with_use(self) -> None:
        config = SandboxConfig(use="agent4ml.backend.agents.sandbox.local:LocalSandboxProvider")
        assert config.use != ""

    def test_frozen_immutable(self) -> None:
        config = SandboxConfig()
        with pytest.raises(dataclasses.FrozenInstanceError):
            config.use = "x"  # type: ignore[misc]

    def test_docker_fields_reserved(self) -> None:
        config = SandboxConfig(
            image="sandbox:latest",
            port=9090,
            container_prefix="test-sandbox",
            idle_timeout=300,
            replicas=5,
            provisioner_url="http://k8s:8002",
        )
        assert config.image == "sandbox:latest"
        assert config.port == 9090
        assert config.container_prefix == "test-sandbox"
        assert config.idle_timeout == 300
        assert config.replicas == 5
        assert config.provisioner_url == "http://k8s:8002"


class TestSandboxMountConfig:
    def test_construct(self) -> None:
        mount = SandboxMountConfig(host_path="/host", container_path="/container")
        assert mount.host_path == "/host"
        assert mount.container_path == "/container"
        assert mount.read_only is False

    def test_read_only(self) -> None:
        mount = SandboxMountConfig("/h", "/c", read_only=True)
        assert mount.read_only is True

    def test_frozen(self) -> None:
        mount = SandboxMountConfig("/h", "/c")
        with pytest.raises(dataclasses.FrozenInstanceError):
            mount.host_path = "/other"  # type: ignore[misc]


class TestStartupOnlyFields:
    def test_contains_sandbox(self) -> None:
        assert "sandbox" in STARTUP_ONLY_FIELDS

    def test_is_set(self) -> None:
        assert isinstance(STARTUP_ONLY_FIELDS, set)
