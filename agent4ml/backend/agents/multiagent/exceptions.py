"""Multi-Agent 异常层次 — SpecialistError + SubagentError 独立层次。

设计（design.md §8 + spec.md SpecialistError Requirement）:
- SpecialistError 基类带 details dict + __str__ 自动展开（参考 sandbox.exceptions.SandboxError）
- 4 个 specialist 子类：Timeout / Crash（exit_code）/ Startup / Credential
- SubagentError 独立层次（不属 SpecialistError，subagent 是 Agent4ML 内部）
- SpecialistNotFoundError 用于 Registry.get 缺失
- pairing 完整性（INV#6）：specialist 失败抛 SpecialistError 子类，OrchestrationMiddleware 转 error ToolMessage
"""
from __future__ import annotations


class SpecialistError(Exception):
    """Specialist 错误基类。带 details dict，__str__ 自动展开。

    details 展开格式：``message (k='v', k2='v2')``（参考 sandbox.exceptions.SandboxError）。
    """

    def __init__(self, message: str = "", *, details: dict | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __str__(self) -> str:
        if not self.details:
            return self.message
        parts = [f"{k}={v!r}" for k, v in self.details.items()]
        return f"{self.message} ({', '.join(parts)})"


class SpecialistTimeoutError(SpecialistError):
    """specialist 超时（timeout_seconds 触发 kill）。"""

    def __init__(
        self,
        message: str = "specialist timeout",
        *,
        timeout_seconds: float | None = None,
    ) -> None:
        details = {"timeout_seconds": timeout_seconds} if timeout_seconds is not None else {}
        super().__init__(message, details=details)


class SpecialistCrashError(SpecialistError):
    """specialist 子进程异常退出（含 exit_code）。"""

    def __init__(
        self,
        message: str = "specialist crash",
        *,
        exit_code: int | None = None,
    ) -> None:
        details = {"exit_code": exit_code} if exit_code is not None else {}
        super().__init__(message, details=details)


class SpecialistStartupError(SpecialistError):
    """specialist 启动失败（ACP handshake 失败 / 子进程 spawn 失败）。"""


class SpecialistCredentialError(SpecialistError):
    """specialist 凭证缺失或失效（不进 LLM 主态，标 disabled）。"""


class SpecialistNotFoundError(SpecialistError):
    """specialist 未注册（SpecialistRegistry.get 缺失抛此）。"""

    def __init__(self, name: str) -> None:
        super().__init__(
            f"specialist not registered: {name}",
            details={"name": name},
        )


class SubagentError(Exception):
    """Subagent 错误基类（独立层次，不属 SpecialistError）。

    subagent 是 Agent4ML 内部 self-copy，不属 specialist 黑盒，独立异常层次。
    """

    def __init__(self, message: str = "", *, details: dict | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __str__(self) -> str:
        if not self.details:
            return self.message
        parts = [f"{k}={v!r}" for k, v in self.details.items()]
        return f"{self.message} ({', '.join(parts)})"


class SubagentTimeoutError(SubagentError):
    """subagent 超时（timeout_seconds 触发中断）。"""


class SubagentMaxStepsError(SubagentError):
    """subagent 超过 max_steps 限制（leaf role 递归控制，INV#4）。"""

    def __init__(
        self,
        message: str = "subagent exceeded max_steps",
        *,
        max_steps: int | None = None,
    ) -> None:
        details = {"max_steps": max_steps} if max_steps is not None else {}
        super().__init__(message, details=details)
