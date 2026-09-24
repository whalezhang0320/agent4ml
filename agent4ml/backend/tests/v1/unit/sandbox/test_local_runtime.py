from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from agent4ml.backend.agents.sandbox.exceptions import (
    SandboxCommandError,
    SandboxError,
    SandboxFileNotFoundError,
    SandboxPermissionError,
    SandboxRuntimeError,
)
from agent4ml.backend.agents.sandbox.runtimes.local_runtime import LocalRuntime
from agent4ml.backend.agents.sandbox.types import GrepMatch


@pytest.fixture
def runtime() -> LocalRuntime:
    return LocalRuntime()


class TestExecCommand:
    def test_success(self, runtime: LocalRuntime) -> None:
        assert runtime.exec_command("echo hello").strip() == "hello"

    def test_failure_raises(self, runtime: LocalRuntime) -> None:
        with pytest.raises(SandboxCommandError) as exc_info:
            runtime.exec_command("exit 1")
        assert exc_info.value.details["exit_code"] == 1

    def test_failure_is_sandbox_error(self, runtime: LocalRuntime) -> None:
        with pytest.raises(SandboxError):
            runtime.exec_command("exit 1")

    def test_command_truncated_in_error(self, runtime: LocalRuntime) -> None:
        long_cmd = "x" * 150
        with pytest.raises(SandboxCommandError) as exc_info:
            runtime.exec_command(f"{long_cmd}; exit 1")
        assert "..." in exc_info.value.details["command"]


class TestAllowHostBashGate:
    """S2 安全加固：allow_host_bash=False 时 exec_command 被拒。"""

    def test_disabled_raises_runtime_error(self) -> None:
        rt = LocalRuntime(allow_host_bash=False)
        with pytest.raises(SandboxRuntimeError, match="host bash is disabled"):
            rt.exec_command("echo hello")

    def test_enabled_executes_normally(self) -> None:
        rt = LocalRuntime(allow_host_bash=True)
        assert rt.exec_command("echo hello").strip() == "hello"

    def test_default_is_enabled(self) -> None:
        rt = LocalRuntime()
        assert rt.exec_command("echo hello").strip() == "hello"


class TestReadFile:
    def test_success(self, runtime: LocalRuntime, tmp_path) -> None:
        f = tmp_path / "test.txt"
        f.write_text("content", encoding="utf-8")
        assert runtime.read_file(str(f)) == "content"

    def test_not_found(self, runtime: LocalRuntime) -> None:
        with pytest.raises(SandboxFileNotFoundError):
            runtime.read_file("/nonexistent/path/file.txt")

    def test_not_found_is_sandbox_error(self, runtime: LocalRuntime) -> None:
        with pytest.raises(SandboxError):
            runtime.read_file("/nonexistent/file")


class TestWriteFile:
    def test_write(self, runtime: LocalRuntime, tmp_path) -> None:
        f = tmp_path / "out.txt"
        runtime.write_file(str(f), "data")
        assert f.read_text() == "data"

    def test_creates_parent_dirs(self, runtime: LocalRuntime, tmp_path) -> None:
        f = tmp_path / "sub" / "dir" / "out.txt"
        runtime.write_file(str(f), "data")
        assert f.read_text() == "data"

    def test_append(self, runtime: LocalRuntime, tmp_path) -> None:
        f = tmp_path / "out.txt"
        runtime.write_file(str(f), "a")
        runtime.write_file(str(f), "b", append=True)
        assert f.read_text() == "ab"


class TestListDir:
    def test_returns_relative_paths(self, runtime: LocalRuntime, tmp_path) -> None:
        (tmp_path / "a.txt").write_text("x")
        (tmp_path / "b.txt").write_text("x")
        entries = runtime.list_dir(str(tmp_path))
        assert "a.txt" in entries
        assert "b.txt" in entries

    def test_max_depth(self, runtime: LocalRuntime, tmp_path) -> None:
        (tmp_path / "d1").mkdir()
        (tmp_path / "d1" / "d2").mkdir()
        (tmp_path / "d1" / "d2" / "deep.txt").write_text("x")
        entries = runtime.list_dir(str(tmp_path), max_depth=1)
        assert all(len(Path(e).parts) <= 1 for e in entries)

    def test_max_entries_truncation(self, runtime: LocalRuntime, tmp_path) -> None:
        """S9: max_entries 截断——超限不继续扫描。"""
        for i in range(50):
            (tmp_path / f"f{i:02d}.txt").write_text("x")
        entries = runtime.list_dir(str(tmp_path), max_entries=10)
        assert len(entries) <= 10

    def test_bfs_does_not_traverse_deep_unnecessarily(self, runtime: LocalRuntime, tmp_path) -> None:
        """S9: max_depth=1 时不递归进子目录（BFS 剪枝）。"""
        (tmp_path / "d1").mkdir()
        (tmp_path / "d1" / "deep.txt").write_text("x")
        (tmp_path / "d1" / "d2").mkdir()
        (tmp_path / "d1" / "d2" / "deeper.txt").write_text("x")
        entries = runtime.list_dir(str(tmp_path), max_depth=1)
        # d1 在 entries（depth 1），d1/deep.txt 不在（depth 2 > max_depth=1）
        assert "d1" in entries
        assert all(not e.startswith("d1/") for e in entries)

    def test_large_dir_no_explosion(self, runtime: LocalRuntime, tmp_path) -> None:
        """S9: 大目录不爆内存——max_entries 截断后立即停止。"""
        for i in range(500):
            (tmp_path / f"f{i:04d}.txt").write_text("x")
        entries = runtime.list_dir(str(tmp_path), max_entries=50)
        assert len(entries) <= 50  # 截断生效


class TestGlob:
    def test_match(self, runtime: LocalRuntime, tmp_path) -> None:
        (tmp_path / "a.py").write_text("x")
        (tmp_path / "b.txt").write_text("x")
        matches, truncated = runtime.glob(str(tmp_path), "*.py")
        assert "a.py" in matches
        assert "b.txt" not in matches
        assert truncated is False

    def test_truncated(self, runtime: LocalRuntime, tmp_path) -> None:
        for i in range(5):
            (tmp_path / f"f{i}.txt").write_text("x")
        matches, truncated = runtime.glob(str(tmp_path), "*.txt", max_results=2)
        assert len(matches) == 2
        assert truncated is True


class TestGrep:
    def test_match(self, runtime: LocalRuntime, tmp_path) -> None:
        (tmp_path / "a.py").write_text("print('hello')\nprint('world')\n")
        matches, truncated = runtime.grep(str(tmp_path), "hello")
        assert len(matches) == 1
        assert "hello" in matches[0].line

    def test_ignore_patterns(self, runtime: LocalRuntime, tmp_path) -> None:
        (tmp_path / ".git").mkdir()
        (tmp_path / ".git" / "config").write_text("hello")
        (tmp_path / "main.py").write_text("hello")
        matches, _ = runtime.grep(str(tmp_path), "hello")
        paths = [m.path for m in matches]
        assert any(".git" not in p for p in paths)
        assert not any(".git" in p for p in paths)

    def test_max_results(self, runtime: LocalRuntime, tmp_path) -> None:
        for i in range(5):
            (tmp_path / f"f{i}.py").write_text("match\n")
        matches, truncated = runtime.grep(str(tmp_path), "match", max_results=2)
        assert len(matches) == 2
        assert truncated is True

    def test_returns_grep_match(self, runtime: LocalRuntime, tmp_path) -> None:
        (tmp_path / "a.py").write_text("pattern here\n")
        matches, _ = runtime.grep(str(tmp_path), "pattern")
        assert len(matches) == 1
        assert isinstance(matches[0], GrepMatch)
        assert matches[0].line_number == 1


class TestGrepReDoSProtection:
    """S10: grep ReDoS 防护——超长 pattern + 嵌套量词被拒。"""

    def test_overlong_pattern_rejected(self, runtime: LocalRuntime, tmp_path) -> None:
        long_pattern = "a" * 201
        with pytest.raises(ValueError, match="too long"):
            runtime.grep(str(tmp_path), long_pattern, literal=False)

    def test_nested_quantifier_rejected(self, runtime: LocalRuntime, tmp_path) -> None:
        with pytest.raises(ValueError, match="ReDoS"):
            runtime.grep(str(tmp_path), "(a+)+", literal=False)

    def test_nested_star_rejected(self, runtime: LocalRuntime, tmp_path) -> None:
        with pytest.raises(ValueError, match="ReDoS"):
            runtime.grep(str(tmp_path), "(a*)*", literal=False)

    def test_mixed_nested_quantifier_rejected(self, runtime: LocalRuntime, tmp_path) -> None:
        with pytest.raises(ValueError, match="ReDoS"):
            runtime.grep(str(tmp_path), "(a+)*", literal=False)

    def test_normal_regex_passes(self, runtime: LocalRuntime, tmp_path) -> None:
        """正常 regex pattern 不被拒。"""
        (tmp_path / "a.py").write_text("hello123\n")
        matches, _ = runtime.grep(str(tmp_path), r"hello\d+", literal=False)
        assert len(matches) == 1

    def test_literal_mode_skips_validation(self, runtime: LocalRuntime, tmp_path) -> None:
        """literal=True 时不做 ReDoS 校验（pattern 被 re.escape 转义）。"""
        (tmp_path / "a.py").write_text("(a+)+ here\n")
        matches, _ = runtime.grep(str(tmp_path), "(a+)+", literal=True)
        assert len(matches) == 1

    def test_max_length_boundary_passes(self, runtime: LocalRuntime, tmp_path) -> None:
        """正好 200 字符的 pattern 通过。"""
        pattern = "a" * 200
        (tmp_path / "a.py").write_text(pattern + "\n")
        matches, _ = runtime.grep(str(tmp_path), pattern, literal=False)
        assert len(matches) == 1


class TestDownloadUpdate:
    def test_download(self, runtime: LocalRuntime, tmp_path) -> None:
        f = tmp_path / "bin.dat"
        f.write_bytes(b"\x00\x01\x02")
        assert runtime.download_file(str(f)) == b"\x00\x01\x02"

    def test_download_not_found(self, runtime: LocalRuntime) -> None:
        with pytest.raises(SandboxFileNotFoundError):
            runtime.download_file("/nonexistent")

    def test_update(self, runtime: LocalRuntime, tmp_path) -> None:
        f = tmp_path / "out.bin"
        runtime.update_file(str(f), b"\xff\xfe")
        assert f.read_bytes() == b"\xff\xfe"

    def test_update_creates_parents(self, runtime: LocalRuntime, tmp_path) -> None:
        f = tmp_path / "sub" / "out.bin"
        runtime.update_file(str(f), b"data")
        assert f.read_bytes() == b"data"


class TestClose:
    def test_close_noop(self, runtime: LocalRuntime) -> None:
        runtime.close()
