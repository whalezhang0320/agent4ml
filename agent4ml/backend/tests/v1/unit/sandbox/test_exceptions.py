from __future__ import annotations

import pytest

from agent4ml.backend.agents.sandbox.exceptions import (
    SandboxCommandError,
    SandboxError,
    SandboxFileError,
    SandboxFileNotFoundError,
    SandboxNotFoundError,
    SandboxPermissionError,
    SandboxRuntimeError,
)


class TestSandboxErrorStr:
    def test_no_details_returns_message(self) -> None:
        err = SandboxError("something failed")
        assert str(err) == "something failed"

    def test_empty_details_returns_message(self) -> None:
        err = SandboxError("failed", details={})
        assert str(err) == "failed"

    def test_with_details_formats_key_val(self) -> None:
        err = SandboxError("failed", details={"key": "val"})
        assert str(err) == "failed (key='val')"

    def test_with_multiple_details(self) -> None:
        err = SandboxError("failed", details={"a": 1, "b": "x"})
        result = str(err)
        assert "failed" in result
        assert "a=1" in result
        assert "b='x'" in result

    def test_message_attribute(self) -> None:
        err = SandboxError("msg", details={"k": "v"})
        assert err.message == "msg"

    def test_details_attribute(self) -> None:
        err = SandboxError("msg", details={"k": "v"})
        assert err.details == {"k": "v"}


class TestSandboxNotFoundError:
    def test_includes_sandbox_id(self) -> None:
        err = SandboxNotFoundError("abc123")
        assert "abc123" in str(err)
        assert err.details["sandbox_id"] == "abc123"

    def test_is_sandbox_error(self) -> None:
        err = SandboxNotFoundError("x")
        assert isinstance(err, SandboxError)


class TestSandboxCommandError:
    def test_short_command_not_truncated(self) -> None:
        err = SandboxCommandError("cmd failed", command="ls", exit_code=1)
        assert err.details["command"] == "ls"
        assert err.details["exit_code"] == 1

    def test_long_command_truncated(self) -> None:
        long_cmd = "x" * 150
        err = SandboxCommandError("failed", command=long_cmd)
        assert err.details["command"] == "x" * 100 + "..."

    def test_exit_code_none_default(self) -> None:
        err = SandboxCommandError("failed", command="ls")
        assert err.details["exit_code"] is None


class TestSandboxFileError:
    def test_includes_path_and_operation(self) -> None:
        err = SandboxFileError("denied", path="/mnt/agent4ml/user-data/x", operation="read")
        assert err.details["path"] == "/mnt/agent4ml/user-data/x"
        assert err.details["operation"] == "read"


class TestExceptionHierarchy:
    def test_permission_is_file_error(self) -> None:
        err = SandboxPermissionError("denied", path="/x", operation="read")
        assert isinstance(err, SandboxFileError)

    def test_file_not_found_is_file_error(self) -> None:
        err = SandboxFileNotFoundError("missing", path="/x", operation="read")
        assert isinstance(err, SandboxFileError)

    def test_file_error_is_sandbox_error(self) -> None:
        err = SandboxFileError("err", path="/x", operation="read")
        assert isinstance(err, SandboxError)

    def test_runtime_error_is_sandbox_error(self) -> None:
        err = SandboxRuntimeError("runtime down")
        assert isinstance(err, SandboxError)

    def test_command_error_is_sandbox_error(self) -> None:
        err = SandboxCommandError("failed", command="ls")
        assert isinstance(err, SandboxError)

    def test_not_found_is_sandbox_error(self) -> None:
        err = SandboxNotFoundError("x")
        assert isinstance(err, SandboxError)

    @pytest.mark.parametrize(
        "exc_cls",
        [
            SandboxNotFoundError,
            SandboxRuntimeError,
            SandboxCommandError,
            SandboxFileError,
            SandboxPermissionError,
            SandboxFileNotFoundError,
        ],
    )
    def test_all_subclass_sandbox_error(self, exc_cls: type) -> None:
        assert issubclass(exc_cls, SandboxError)
