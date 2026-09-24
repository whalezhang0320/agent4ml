from __future__ import annotations

from contextvars import ContextVar

_sandbox_id: ContextVar[str | None] = ContextVar("sandbox_id", default=None)


def set_sandbox_id(sandbox_id: str | None) -> None:
    """设置当前 tool_call 的 sandbox_id。

    Stage 4 SandboxMiddleware 在 wrap_tool_call 前调用。
    """
    _sandbox_id.set(sandbox_id)


def get_sandbox_id() -> str | None:
    """获取当前 tool_call 的 sandbox_id。"""
    return _sandbox_id.get()
