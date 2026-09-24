from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from agent4ml.backend.agents.sandbox.contracts import (
    PathTranslator,
    SandboxRuntime,
    SecurityGuard,
)
from agent4ml.backend.agents.sandbox.sandbox import Sandbox
from agent4ml.backend.agents.sandbox.types import GrepMatch


def _make_sandbox() -> tuple[Sandbox, MagicMock, MagicMock, MagicMock, list[str]]:
    """构造 Sandbox + mock 三组件 + 共享 calls 记录。"""
    calls: list[str] = []
    runtime = MagicMock(spec=SandboxRuntime)
    translator = MagicMock(spec=PathTranslator)
    guard = MagicMock(spec=SecurityGuard)

    def t_path(p: str) -> str:
        calls.append("t.translate_path")
        return p

    def t_cmd(c: str) -> str:
        calls.append("t.translate_command")
        return c

    def t_mask(o: str) -> str:
        calls.append("t.mask")
        return o

    translator.translate_path.side_effect = t_path
    translator.translate_command.side_effect = t_cmd
    translator.mask_output.side_effect = t_mask

    def g_path(path: str, *, write: bool = False) -> None:
        calls.append(f"g.validate_path(w={write})")

    def g_cmd(cmd: str) -> None:
        calls.append("g.validate_command")

    guard.validate_path.side_effect = g_path
    guard.validate_command.side_effect = g_cmd

    runtime.exec_command.side_effect = lambda c: (calls.append("r.exec"), "out")[1]
    runtime.read_file.side_effect = lambda p: (calls.append("r.read"), "content")[1]
    runtime.write_file.side_effect = lambda p, c, append=False: calls.append("r.write")
    runtime.list_dir.side_effect = lambda p, max_depth=2, **kw: (calls.append("r.listdir"), ["/a", "/b"])[1]
    runtime.glob.side_effect = lambda *a, **k: (calls.append("r.glob"), (["/x"], False))[1]
    runtime.download_file.side_effect = lambda p: (calls.append("r.download"), b"bytes")[1]
    runtime.update_file.side_effect = lambda p, c: calls.append("r.update")

    sandbox = Sandbox("sb-1", runtime, translator, guard)
    return sandbox, runtime, translator, guard, calls


class TestExecuteCommand:
    def test_orchestration_order(self) -> None:
        sb, _, _, _, calls = _make_sandbox()
        result = sb.execute_command("ls")
        assert result == "out"
        assert calls == ["g.validate_command", "t.translate_command", "r.exec", "t.mask"]


class TestReadFile:
    def test_orchestration_order(self) -> None:
        sb, _, _, _, calls = _make_sandbox()
        result = sb.read_file("/mnt/x")
        assert result == "content"
        assert calls == ["g.validate_path(w=False)", "t.translate_path", "r.read", "t.mask"]


class TestWriteFile:
    def test_orchestration_order_write_true(self) -> None:
        sb, _, _, _, calls = _make_sandbox()
        sb.write_file("/mnt/x", "data")
        assert calls == ["g.validate_path(w=True)", "t.translate_path", "r.write"]

    def test_append_passed_through(self) -> None:
        sb, runtime, _, _, _ = _make_sandbox()
        sb.write_file("/mnt/x", "data", append=True)
        runtime.write_file.assert_called_once_with("/mnt/x", "data", append=True)


class TestListDir:
    def test_orchestration_with_element_mask(self) -> None:
        sb, _, translator, _, calls = _make_sandbox()
        result = sb.list_dir("/mnt/x")
        assert result == ["/a", "/b"]
        assert calls == ["g.validate_path(w=False)", "t.translate_path", "r.listdir", "t.mask", "t.mask"]
        assert translator.mask_output.call_count == 2


class TestGlob:
    def test_orchestration_with_element_mask(self) -> None:
        sb, _, translator, _, calls = _make_sandbox()
        result, truncated = sb.glob("/mnt/x", "*.py")
        assert result == ["/x"]
        assert truncated is False
        assert "r.glob" in calls
        assert "t.mask" in calls


class TestGrep:
    def test_grep_match_line_path_masked(self) -> None:
        sb, runtime, translator, _, _ = _make_sandbox()
        runtime.grep.return_value = ([GrepMatch("/host/secret", 42, "/host/leak")], False)
        translator.mask_output.side_effect = lambda o: o.replace("/host", "/mnt")
        result, truncated = sb.grep("/mnt/x", "pattern")
        assert truncated is False
        assert len(result) == 1
        assert result[0].path == "/mnt/secret"
        assert result[0].line == "/mnt/leak"
        assert result[0].line_number == 42

    def test_grep_preserves_line_number(self) -> None:
        sb, runtime, translator, _, _ = _make_sandbox()
        runtime.grep.return_value = ([GrepMatch("/p", 99, "line")], True)
        result, truncated = sb.grep("/mnt/x", "p")
        assert result[0].line_number == 99
        assert truncated is True


class TestGuardBlocksRuntime:
    def test_execute_command_guard_raises_runtime_not_called(self) -> None:
        sb, runtime, _, guard, _ = _make_sandbox()
        guard.validate_command.side_effect = PermissionError("denied")
        with pytest.raises(PermissionError):
            sb.execute_command("ls")
        runtime.exec_command.assert_not_called()

    def test_read_file_guard_raises_runtime_not_called(self) -> None:
        sb, runtime, _, guard, _ = _make_sandbox()
        guard.validate_path.side_effect = PermissionError("denied")
        with pytest.raises(PermissionError):
            sb.read_file("/mnt/x")
        runtime.read_file.assert_not_called()


class TestCloseAndId:
    def test_close_delegates_runtime(self) -> None:
        sb, runtime, _, _, _ = _make_sandbox()
        sb.close()
        runtime.close.assert_called_once()

    def test_id_property(self) -> None:
        sb, _, _, _, _ = _make_sandbox()
        assert sb.id == "sb-1"

    def test_id_readonly(self) -> None:
        sb, _, _, _, _ = _make_sandbox()
        with pytest.raises(AttributeError):
            sb.id = "other"  # type: ignore[misc]


class TestDownloadAndUpdate:
    def test_download_orchestration(self) -> None:
        sb, _, _, _, calls = _make_sandbox()
        result = sb.download_file("/mnt/x")
        assert result == b"bytes"
        assert calls == ["g.validate_path(w=False)", "t.translate_path", "r.download"]

    def test_update_orchestration_write_true(self) -> None:
        sb, _, _, _, calls = _make_sandbox()
        sb.update_file("/mnt/x", b"data")
        assert calls == ["g.validate_path(w=True)", "t.translate_path", "r.update"]
