from __future__ import annotations


class SandboxError(Exception):
    """Sandbox 错误基类。带 details dict，__str__ 自动展开。"""

    def __init__(self, message: str = "", *, details: dict | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __str__(self) -> str:
        if not self.details:
            return self.message
        parts = [f"{k}={v!r}" for k, v in self.details.items()]
        return f"{self.message} ({', '.join(parts)})"


class SandboxNotFoundError(SandboxError):
    """沙箱找不到。"""

    def __init__(self, sandbox_id: str) -> None:
        super().__init__(
            f"sandbox not found: {sandbox_id}",
            details={"sandbox_id": sandbox_id},
        )


class SandboxRuntimeError(SandboxError):
    """runtime 不可用 / 配置错。"""


class SandboxCommandError(SandboxError):
    """命令执行失败。command 自动截断 100 字符。"""

    def __init__(
        self,
        message: str,
        *,
        command: str,
        exit_code: int | None = None,
    ) -> None:
        truncated = command[:100] + "..." if len(command) > 100 else command
        super().__init__(
            message,
            details={"command": truncated, "exit_code": exit_code},
        )


class SandboxFileError(SandboxError):
    """文件操作失败。"""

    def __init__(self, message: str, *, path: str, operation: str) -> None:
        super().__init__(
            message,
            details={"path": path, "operation": operation},
        )


class SandboxPermissionError(SandboxFileError):
    """权限拒绝。"""


class SandboxFileNotFoundError(SandboxFileError):
    """文件不存在。"""
