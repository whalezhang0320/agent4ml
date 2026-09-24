from __future__ import annotations

import dataclasses
from copy import deepcopy

import pytest

from agent4ml.backend.agents.sandbox.types import (
    GrepMatch,
    PathMapping,
    ResolvedPath,
    SandboxInfo,
)


class TestPathMapping:
    def test_construct(self) -> None:
        m = PathMapping(
            container_path="/mnt/user-data/workspace",
            local_path="/tmp/ws",
        )
        assert m.container_path == "/mnt/user-data/workspace"
        assert m.local_path == "/tmp/ws"
        assert m.read_only is False

    def test_read_only(self) -> None:
        m = PathMapping("/c", "/l", read_only=True)
        assert m.read_only is True

    def test_frozen_immutable(self) -> None:
        m = PathMapping("/c", "/l")
        with pytest.raises(dataclasses.FrozenInstanceError):
            m.container_path = "/other"  # type: ignore[misc]

    def test_frozen_immutable_read_only(self) -> None:
        m = PathMapping("/c", "/l")
        with pytest.raises(dataclasses.FrozenInstanceError):
            m.read_only = True  # type: ignore[misc]


class TestResolvedPath:
    def test_with_mapping(self) -> None:
        m = PathMapping("/c", "/l")
        rp = ResolvedPath(path="/l/file", mapping=m)
        assert rp.path == "/l/file"
        assert rp.mapping is m

    def test_without_mapping(self) -> None:
        rp = ResolvedPath(path="/raw", mapping=None)
        assert rp.path == "/raw"
        assert rp.mapping is None

    def test_namedtuple_unpack(self) -> None:
        m = PathMapping("/c", "/l")
        rp = ResolvedPath("/l", m)
        path, mapping = rp
        assert path == "/l"
        assert mapping is m


class TestGrepMatch:
    def test_construct(self) -> None:
        gm = GrepMatch(path="/mnt/user-data/x.py", line_number=42, line="print('hi')")
        assert gm.path == "/mnt/user-data/x.py"
        assert gm.line_number == 42
        assert gm.line == "print('hi')"

    def test_mutable(self) -> None:
        gm = GrepMatch("/x", 1, "line")
        gm.line = "modified"
        assert gm.line == "modified"


class TestSandboxInfo:
    def test_construct_minimal(self) -> None:
        info = SandboxInfo(sandbox_id="abc", sandbox_url="http://localhost:8080")
        assert info.sandbox_id == "abc"
        assert info.sandbox_url == "http://localhost:8080"
        assert info.container_name is None
        assert info.container_id is None
        assert info.created_at  # non-empty

    def test_construct_full(self) -> None:
        info = SandboxInfo(
            sandbox_id="abc",
            sandbox_url="http://x",
            container_name="sandbox-abc",
            container_id="docker-123",
            created_at="2026-07-13T12:00:00Z",
        )
        assert info.container_name == "sandbox-abc"
        assert info.container_id == "docker-123"
        assert info.created_at == "2026-07-13T12:00:00Z"

    def test_created_at_is_string_not_float(self) -> None:
        info = SandboxInfo(sandbox_id="x", sandbox_url="http://x")
        assert isinstance(info.created_at, str)
        assert not isinstance(info.created_at, float)

    def test_created_at_iso_format(self) -> None:
        info = SandboxInfo(sandbox_id="x", sandbox_url="http://x")
        assert "T" in info.created_at

    def test_to_dict(self) -> None:
        info = SandboxInfo(
            sandbox_id="abc",
            sandbox_url="http://x",
            container_name="c",
            container_id="d",
            created_at="2026-07-13T12:00:00Z",
        )
        d = info.to_dict()
        assert d == {
            "sandbox_id": "abc",
            "sandbox_url": "http://x",
            "container_name": "c",
            "container_id": "d",
            "created_at": "2026-07-13T12:00:00Z",
        }

    def test_from_dict_roundtrip(self) -> None:
        original = SandboxInfo(
            sandbox_id="abc",
            sandbox_url="http://x",
            container_name="c",
            container_id="d",
            created_at="2026-07-13T12:00:00Z",
        )
        d = original.to_dict()
        restored = SandboxInfo.from_dict(deepcopy(d))
        assert restored == original

    def test_from_dict_base_url_compat(self) -> None:
        data = {
            "sandbox_id": "abc",
            "base_url": "http://legacy",
        }
        info = SandboxInfo.from_dict(data)
        assert info.sandbox_url == "http://legacy"

    def test_from_dict_sandbox_url_preferred_over_base_url(self) -> None:
        data = {
            "sandbox_id": "abc",
            "sandbox_url": "http://new",
            "base_url": "http://legacy",
        }
        info = SandboxInfo.from_dict(data)
        assert info.sandbox_url == "http://new"

    def test_from_dict_missing_created_at_uses_now(self) -> None:
        data = {"sandbox_id": "abc", "sandbox_url": "http://x"}
        info = SandboxInfo.from_dict(data)
        assert info.created_at
        assert isinstance(info.created_at, str)

    def test_from_dict_missing_sandbox_id_raises(self) -> None:
        with pytest.raises(KeyError):
            SandboxInfo.from_dict({"sandbox_url": "http://x"})
