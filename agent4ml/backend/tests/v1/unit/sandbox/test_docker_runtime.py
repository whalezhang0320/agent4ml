from __future__ import annotations

import base64
import sys
import threading
import time
import types
from unittest.mock import MagicMock

import pytest

# Inject mock agent_sandbox module so tests run without the real package installed.
# import firewall: only this test file injects mock.
if "agent_sandbox" not in sys.modules:
    _mock_mod = types.ModuleType("agent_sandbox")
    _mock_mod.Sandbox = MagicMock
    sys.modules["agent_sandbox"] = _mock_mod

from agent4ml.backend.agents.sandbox.exceptions import (
    SandboxCommandError,
    SandboxError,
    SandboxFileError,
    SandboxFileNotFoundError,
    SandboxPermissionError,
    SandboxRuntimeError,
)
from agent4ml.backend.agents.sandbox.runtimes.docker_runtime import (
    _ERROR_OBSERVATION_SIGNATURE,
    DockerRuntime,
)


def _make_result(output: str = "", content: str = "", files=None):
    """Build a mock SDK result with .data attribute."""
    data = MagicMock()
    data.output = output
    data.content = content
    data.files = files or []
    result = MagicMock()
    result.data = data
    return result


def _make_runtime(url: str = "http://localhost:8080") -> DockerRuntime:
    """Build a DockerRuntime with a mock SDK client."""
    return DockerRuntime(sandbox_url=url)


class TestInit:
    def test_holds_client_lock_closed(self) -> None:
        rt = _make_runtime()
        assert rt._url == "http://localhost:8080"
        assert rt._client is not None
        assert isinstance(rt._lock, type(threading.Lock()))
        assert rt._closed is False

    def test_url_strips_trailing_slash_not_required(self) -> None:
        rt = _make_runtime("http://localhost:8080/")
        assert rt._url == "http://localhost:8080/"


class TestExecCommand:
    def test_success(self) -> None:
        rt = _make_runtime()
        rt._client.shell.exec_command.return_value = _make_result(output="hello")
        assert rt.exec_command("echo hello") == "hello"

    def test_no_output_returns_placeholder(self) -> None:
        rt = _make_runtime()
        rt._client.shell.exec_command.return_value = _make_result(output="")
        assert rt.exec_command("true") == "(no output)"

    def test_none_data_returns_placeholder(self) -> None:
        rt = _make_runtime()
        result = MagicMock()
        result.data = None
        rt._client.shell.exec_command.return_value = result
        assert rt.exec_command("true") == "(no output)"

    def test_serial_lock(self) -> None:
        """threading.Lock serializes concurrent exec calls."""
        rt = _make_runtime()
        active = threading.Event()
        conflicts: list[bool] = []

        def slow_exec(command, **kwargs):
            if active.is_set():
                conflicts.append(True)
            active.set()
            time.sleep(0.03)
            active.clear()
            return _make_result(output="ok")

        rt._client.shell.exec_command.side_effect = slow_exec
        threads = [threading.Thread(target=rt.exec_command, args=("cmd",)) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert conflicts == []

    def test_error_observation_retry(self) -> None:
        rt = _make_runtime()
        call_count = [0]

        def exec_with_error(command, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return _make_result(output=_ERROR_OBSERVATION_SIGNATURE)
            return _make_result(output="real output")

        rt._client.shell.exec_command.side_effect = exec_with_error
        assert rt.exec_command("test") == "real output"
        assert call_count[0] == 2
        rt._client.shell.create_session.assert_called_once()
        rt._client.shell.cleanup_session.assert_called_once()

    def test_error_observation_retry_still_returns_placeholder_if_empty(self) -> None:
        rt = _make_runtime()
        call_count = [0]

        def exec_with_error(command, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return _make_result(output=_ERROR_OBSERVATION_SIGNATURE)
            return _make_result(output="")

        rt._client.shell.exec_command.side_effect = exec_with_error
        assert rt.exec_command("test") == "(no output)"

    def test_exception_wrapped_to_command_error(self) -> None:
        rt = _make_runtime()
        rt._client.shell.exec_command.side_effect = RuntimeError("connection lost")
        with pytest.raises(SandboxCommandError) as exc_info:
            rt.exec_command("cmd")
        assert exc_info.value.details["command"] == "cmd"
        assert exc_info.value.details["exit_code"] == -1

    def test_command_error_is_sandbox_error(self) -> None:
        rt = _make_runtime()
        rt._client.shell.exec_command.side_effect = RuntimeError("boom")
        with pytest.raises(SandboxError):
            rt.exec_command("cmd")

    def test_closed_runtime_raises(self) -> None:
        rt = _make_runtime()
        rt._closed = True
        with pytest.raises(SandboxRuntimeError):
            rt.exec_command("cmd")

    def test_cleanup_session_failure_does_not_break_retry(self) -> None:
        rt = _make_runtime()
        call_count = [0]

        def exec_with_error(command, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return _make_result(output=_ERROR_OBSERVATION_SIGNATURE)
            return _make_result(output="ok")

        rt._client.shell.exec_command.side_effect = exec_with_error
        rt._client.shell.cleanup_session.side_effect = RuntimeError("cleanup failed")
        assert rt.exec_command("test") == "ok"

    def test_error_observation_persists_after_retry_raises(self) -> None:
        rt = _make_runtime()
        rt._client.shell.exec_command.return_value = _make_result(
            output=_ERROR_OBSERVATION_SIGNATURE
        )
        with pytest.raises(SandboxCommandError) as exc_info:
            rt.exec_command("failing-cmd")
        assert "ErrorObservation" in str(exc_info.value)
        assert exc_info.value.details["command"] == "failing-cmd"


class TestReadFile:
    def test_success(self) -> None:
        rt = _make_runtime()
        rt._client.file.read_file.return_value = _make_result(content="file body")
        assert rt.read_file("/mnt/agent4ml/user-data/test.txt") == "file body"

    def test_empty_data(self) -> None:
        rt = _make_runtime()
        result = MagicMock()
        result.data = None
        rt._client.file.read_file.return_value = result
        assert rt.read_file("/path") == ""

    def test_not_found(self) -> None:
        rt = _make_runtime()
        rt._client.file.read_file.side_effect = Exception("File not found")
        with pytest.raises(SandboxFileNotFoundError) as exc_info:
            rt.read_file("/missing")
        assert exc_info.value.details["path"] == "/missing"
        assert exc_info.value.details["operation"] == "read"

    def test_no_such_file(self) -> None:
        rt = _make_runtime()
        rt._client.file.read_file.side_effect = Exception("No such file or directory")
        with pytest.raises(SandboxFileNotFoundError):
            rt.read_file("/missing")

    def test_permission_denied(self) -> None:
        rt = _make_runtime()
        rt._client.file.read_file.side_effect = Exception("Permission denied")
        with pytest.raises(SandboxPermissionError) as exc_info:
            rt.read_file("/secret")
        assert exc_info.value.details["operation"] == "read"

    def test_403_permission(self) -> None:
        rt = _make_runtime()
        rt._client.file.read_file.side_effect = Exception("HTTP 403 Forbidden")
        with pytest.raises(SandboxPermissionError):
            rt.read_file("/secret")

    def test_generic_error(self) -> None:
        rt = _make_runtime()
        rt._client.file.read_file.side_effect = RuntimeError("disk full")
        with pytest.raises(SandboxFileError) as exc_info:
            rt.read_file("/path")
        assert exc_info.value.details["operation"] == "read"

    def test_not_found_is_sandbox_error(self) -> None:
        rt = _make_runtime()
        rt._client.file.read_file.side_effect = Exception("not found")
        with pytest.raises(SandboxError):
            rt.read_file("/missing")


class TestWriteFile:
    def test_write(self) -> None:
        rt = _make_runtime()
        rt.write_file("/path", "content")
        rt._client.file.write_file.assert_called_once_with(file="/path", content="content", append=False)

    def test_append_concatenates(self) -> None:
        rt = _make_runtime()
        rt._client.file.read_file.return_value = _make_result(content="old")
        rt.write_file("/path", "new", append=True)
        rt._client.file.write_file.assert_called_once_with(file="/path", content="new", append=True)

    def test_append_not_found_writes_new(self) -> None:
        rt = _make_runtime()
        rt.write_file("/path", "new", append=True)
        rt._client.file.write_file.assert_called_once_with(file="/path", content="new", append=True)

    def test_append_empty_existing_writes_content(self) -> None:
        rt = _make_runtime()
        rt.write_file("/path", "new", append=True)
        rt._client.file.write_file.assert_called_once_with(file="/path", content="new", append=True)

    def test_permission_denied(self) -> None:
        rt = _make_runtime()
        rt._client.file.write_file.side_effect = Exception("Permission denied")
        with pytest.raises(SandboxPermissionError) as exc_info:
            rt.write_file("/readonly", "x")
        assert exc_info.value.details["operation"] == "write"

    def test_generic_error(self) -> None:
        rt = _make_runtime()
        rt._client.file.write_file.side_effect = RuntimeError("disk full")
        with pytest.raises(SandboxFileError) as exc_info:
            rt.write_file("/path", "x")
        assert exc_info.value.details["operation"] == "write"

    def test_closed_runtime_raises(self) -> None:
        rt = _make_runtime()
        rt._closed = True
        with pytest.raises(SandboxRuntimeError):
            rt.write_file("/path", "x")


class TestClose:
    def test_idempotent(self) -> None:
        rt = _make_runtime()
        real_httpx = MagicMock()
        rt._client._client_wrapper.httpx_client.httpx_client = real_httpx
        rt.close()
        rt.close()  # second call no-op
        real_httpx.close.assert_called_once()

    def test_attribute_chain_walk(self) -> None:
        rt = _make_runtime()
        real_httpx = MagicMock()
        fern_http = MagicMock()
        fern_http.httpx_client = real_httpx
        wrapper = MagicMock()
        wrapper.httpx_client = fern_http
        rt._client._client_wrapper = wrapper
        rt.close()
        real_httpx.close.assert_called_once()

    def test_fallback_to_fern_http(self) -> None:
        """real_httpx missing, fallback to fern_http.close()."""
        rt = _make_runtime()
        fern_http = MagicMock()
        # fern_http has no real httpx_client underneath (simulates Fern wrapper
        # whose inner httpx_client attr is missing). MagicMock auto-creates attrs,
        # so set spec=[] to make getattr return None instead of a child mock.
        fern_http.httpx_client = MagicMock(spec=[])
        fern_http.close = MagicMock()
        rt._client._client_wrapper.httpx_client = fern_http
        rt.close()
        fern_http.close.assert_called_once()

    def test_fail_safe(self) -> None:
        rt = _make_runtime()
        rt._client._client_wrapper.httpx_client.httpx_client.close.side_effect = RuntimeError("boom")
        rt.close()  # should not raise

    def test_drops_reference(self) -> None:
        rt = _make_runtime()
        rt.close()
        assert rt._client is None

    def test_sets_closed_flag(self) -> None:
        rt = _make_runtime()
        rt.close()
        assert rt._closed is True

    def test_exec_after_close_raises(self) -> None:
        rt = _make_runtime()
        rt.close()
        with pytest.raises(SandboxRuntimeError):
            rt.exec_command("cmd")

    def test_write_after_close_raises(self) -> None:
        rt = _make_runtime()
        rt.close()
        with pytest.raises(SandboxRuntimeError):
            rt.write_file("/path", "x")

    def test_close_with_none_data_wrapper(self) -> None:
        """_client_wrapper None, no crash (defensive)."""
        rt = _make_runtime()
        rt._client._client_wrapper = None
        rt.close()  # should not raise


def _make_entry(path: str, is_directory: bool = False):
    """Build a mock file entry with .path + .is_directory."""
    e = MagicMock()
    e.path = path
    e.is_directory = is_directory
    return e


def _make_search_result(line_numbers=None, matches=None):
    """Build a mock search_in_file result with .data.line_numbers + .data.matches."""
    data = MagicMock()
    data.line_numbers = line_numbers or []
    data.matches = matches or []
    result = MagicMock()
    result.data = data
    return result


class TestListDir:
    def test_success(self) -> None:
        rt = _make_runtime()
        rt._client.shell.exec_command.return_value = _make_result(output="a\nb\nc")
        assert rt.list_dir("/mnt/agent4ml/user-data") == ["a", "b", "c"]

    def test_empty_output(self) -> None:
        rt = _make_runtime()
        rt._client.shell.exec_command.return_value = _make_result(output="")
        assert rt.list_dir("/path") == []

    def test_strips_whitespace(self) -> None:
        rt = _make_runtime()
        rt._client.shell.exec_command.return_value = _make_result(output="  a  \n b \n")
        assert rt.list_dir("/path") == ["a", "b"]

    def test_find_command_uses_maxdepth(self) -> None:
        rt = _make_runtime()
        rt._client.shell.exec_command.return_value = _make_result(output="")
        rt.list_dir("/mnt/agent4ml/user-data", max_depth=3)
        call_args = rt._client.shell.exec_command.call_args
        assert "-maxdepth 3" in call_args.kwargs["command"]

    def test_closed_raises(self) -> None:
        rt = _make_runtime()
        rt._closed = True
        with pytest.raises(SandboxRuntimeError):
            rt.list_dir("/path")

    def test_exception_wrapped(self) -> None:
        rt = _make_runtime()
        rt._client.shell.exec_command.side_effect = RuntimeError("boom")
        with pytest.raises(SandboxFileError) as exc_info:
            rt.list_dir("/path")
        assert exc_info.value.details["operation"] == "list"


class TestGlob:
    def test_no_include_dirs(self) -> None:
        rt = _make_runtime()
        rt._client.file.find_files.return_value = _make_result(
            files=["/mnt/agent4ml/user-data/a.py", "/mnt/agent4ml/user-data/b.py"]
        )
        matches, truncated = rt.glob("/mnt/agent4ml/user-data", "*.py")
        assert matches == ["/mnt/agent4ml/user-data/a.py", "/mnt/agent4ml/user-data/b.py"]
        assert truncated is False

    def test_no_include_dirs_filters_ignored(self) -> None:
        rt = _make_runtime()
        rt._client.file.find_files.return_value = _make_result(
            files=["/mnt/agent4ml/user-data/a.py", "/mnt/agent4ml/user-data/.git/config"]
        )
        matches, _ = rt.glob("/mnt/agent4ml/user-data", "*")
        assert "/mnt/agent4ml/user-data/a.py" in matches
        assert all(".git" not in m for m in matches)

    def test_no_include_dirs_truncate(self) -> None:
        rt = _make_runtime()
        files = [f"/mnt/agent4ml/user-data/f{i}.py" for i in range(10)]
        rt._client.file.find_files.return_value = _make_result(files=files)
        matches, truncated = rt.glob("/mnt/agent4ml/user-data", "*.py", max_results=5)
        assert len(matches) == 5
        assert truncated is True

    def test_include_dirs_fnmatch(self) -> None:
        rt = _make_runtime()
        rt._client.file.list_path.return_value = _make_result(
            files=[
                _make_entry("/mnt/agent4ml/user-data/sub"),
                _make_entry("/mnt/agent4ml/user-data/a.py"),
                _make_entry("/mnt/agent4ml/user-data/b.txt"),
            ]
        )
        matches, truncated = rt.glob("/mnt/agent4ml/user-data", "*.py", include_dirs=True)
        assert matches == ["/mnt/agent4ml/user-data/a.py"]
        assert truncated is False

    def test_include_dirs_truncate(self) -> None:
        rt = _make_runtime()
        entries = [_make_entry(f"/mnt/agent4ml/user-data/f{i}.py") for i in range(10)]
        rt._client.file.list_path.return_value = _make_result(files=entries)
        matches, truncated = rt.glob("/mnt/agent4ml/user-data", "*.py", include_dirs=True, max_results=3)
        assert len(matches) == 3
        assert truncated is True

    def test_closed_raises(self) -> None:
        rt = _make_runtime()
        rt._closed = True
        with pytest.raises(SandboxRuntimeError):
            rt.glob("/path", "*")


class TestGrep:
    def test_literal_escape(self) -> None:
        rt = _make_runtime()
        rt._client.file.list_path.return_value = _make_result(files=[])
        rt.grep("/mnt/agent4ml/user-data", "a.b", literal=True)
        search_call = rt._client.file.search_in_file
        # No files to search, but verify regex was escaped
        assert search_call.call_count == 0  # no candidates

    def test_case_insensitive_prefix(self) -> None:
        rt = _make_runtime()
        entry = _make_entry("/mnt/agent4ml/user-data/a.py")
        rt._client.file.list_path.return_value = _make_result(files=[entry])
        rt._client.file.search_in_file.return_value = _make_search_result(
            line_numbers=[1], matches=["Hello"]
        )
        rt.grep("/mnt/agent4ml/user-data", "hello", case_sensitive=False)
        call_kwargs = rt._client.file.search_in_file.call_args.kwargs
        assert call_kwargs["regex"] == "(?i)hello"

    def test_case_sensitive_no_prefix(self) -> None:
        rt = _make_runtime()
        entry = _make_entry("/mnt/agent4ml/user-data/a.py")
        rt._client.file.list_path.return_value = _make_result(files=[entry])
        rt._client.file.search_in_file.return_value = _make_search_result(
            line_numbers=[1], matches=["hello"]
        )
        rt.grep("/mnt/agent4ml/user-data", "hello", case_sensitive=True)
        call_kwargs = rt._client.file.search_in_file.call_args.kwargs
        assert call_kwargs["regex"] == "hello"

    def test_glob_filter(self) -> None:
        rt = _make_runtime()
        # find_files with glob="*.py" returns only .py files (SDK filters)
        rt._client.file.find_files.return_value = _make_result(
            files=["/mnt/agent4ml/user-data/a.py"]
        )
        rt._client.file.search_in_file.return_value = _make_search_result(
            line_numbers=[1], matches=["match"]
        )
        matches, _ = rt.grep("/mnt/agent4ml/user-data", "match", glob="*.py")
        rt._client.file.find_files.assert_called_once_with(path="/mnt/agent4ml/user-data", glob="*.py")
        assert len(matches) == 1
        assert matches[0].path == "/mnt/agent4ml/user-data/a.py"

    def test_max_results_truncate(self) -> None:
        rt = _make_runtime()
        entries = [_make_entry(f"/mnt/agent4ml/user-data/f{i}.py") for i in range(5)]
        rt._client.file.list_path.return_value = _make_result(files=entries)
        rt._client.file.search_in_file.return_value = _make_search_result(
            line_numbers=[1], matches=["match"]
        )
        matches, truncated = rt.grep("/mnt/agent4ml/user-data", "match", max_results=3)
        assert len(matches) == 3
        assert truncated is True

    def test_ignores_git_paths(self) -> None:
        rt = _make_runtime()
        rt._client.file.list_path.return_value = _make_result(
            files=[_make_entry("/mnt/agent4ml/user-data/.git/config")]
        )
        matches, _ = rt.grep("/mnt/agent4ml/user-data", "x")
        assert matches == []

    def test_data_none_skipped(self) -> None:
        rt = _make_runtime()
        entry = _make_entry("/mnt/agent4ml/user-data/a.py")
        rt._client.file.list_path.return_value = _make_result(files=[entry])
        result_none = MagicMock()
        result_none.data = None
        rt._client.file.search_in_file.return_value = result_none
        matches, _ = rt.grep("/mnt/agent4ml/user-data", "x")
        assert matches == []

    def test_invalid_regex_raises(self) -> None:
        import re
        rt = _make_runtime()
        rt._client.file.list_path.return_value = _make_result(files=[])
        with pytest.raises(SandboxFileError):
            rt.grep("/mnt/agent4ml/user-data", "[invalid")

    def test_closed_raises(self) -> None:
        rt = _make_runtime()
        rt._closed = True
        with pytest.raises(SandboxRuntimeError):
            rt.grep("/path", "x")


class TestDownloadFile:
    def test_success(self) -> None:
        rt = _make_runtime()
        rt._client.file.download_file.return_value = [b"chunk1", b"chunk2"]
        assert rt.download_file("/mnt/agent4ml/user-data/file.bin") == b"chunk1chunk2"

    def test_path_traversal_rejected(self) -> None:
        rt = _make_runtime()
        with pytest.raises(SandboxPermissionError) as exc_info:
            rt.download_file("/mnt/agent4ml/user-data/../etc/passwd")
        assert exc_info.value.details["operation"] == "download"

    def test_outside_prefix_rejected(self) -> None:
        rt = _make_runtime()
        with pytest.raises(SandboxPermissionError):
            rt.download_file("/etc/passwd")

    def test_root_prefix_allowed(self) -> None:
        rt = _make_runtime()
        rt._client.file.download_file.return_value = [b"ok"]
        assert rt.download_file("/mnt/agent4ml/user-data") == b"ok"

    def test_100mb_limit(self) -> None:
        rt = _make_runtime()
        big_chunk = b"x" * (100 * 1024 * 1024 + 1)
        rt._client.file.download_file.return_value = [big_chunk]
        with pytest.raises(SandboxFileError) as exc_info:
            rt.download_file("/mnt/agent4ml/user-data/big.bin")
        assert "exceeds" in str(exc_info.value)

    def test_exception_wrapped(self) -> None:
        rt = _make_runtime()
        rt._client.file.download_file.side_effect = RuntimeError("conn lost")
        with pytest.raises(SandboxFileError) as exc_info:
            rt.download_file("/mnt/agent4ml/user-data/file.bin")
        assert exc_info.value.details["operation"] == "download"

    def test_closed_raises(self) -> None:
        rt = _make_runtime()
        rt._closed = True
        with pytest.raises(SandboxRuntimeError):
            rt.download_file("/mnt/agent4ml/user-data/file.bin")


class TestUpdateFile:
    def test_base64_encoding(self) -> None:
        rt = _make_runtime()
        rt.update_file("/mnt/agent4ml/user-data/file.bin", b"hello")
        expected_b64 = base64.b64encode(b"hello").decode("utf-8")
        rt._client.file.write_file.assert_called_once_with(
            file="/mnt/agent4ml/user-data/file.bin", content=expected_b64, encoding="base64",
        )

    def test_empty_content(self) -> None:
        rt = _make_runtime()
        rt.update_file("/mnt/agent4ml/user-data/empty.bin", b"")
        rt._client.file.write_file.assert_called_once()

    def test_exception_wrapped(self) -> None:
        rt = _make_runtime()
        rt._client.file.write_file.side_effect = RuntimeError("disk full")
        with pytest.raises(SandboxFileError) as exc_info:
            rt.update_file("/mnt/agent4ml/user-data/file.bin", b"x")
        assert exc_info.value.details["operation"] == "update"

    def test_closed_raises(self) -> None:
        rt = _make_runtime()
        rt._closed = True
        with pytest.raises(SandboxRuntimeError):
            rt.update_file("/mnt/agent4ml/user-data/file.bin", b"x")
