"""Memory 模块异常层次。

基类 `MemoryError` 带 `details: dict` + `__str__` 自动展开（照抄 sandbox
`SandboxError` 模式）。5 子类按操作语义携带上下文（trace_id / operation+path /
query 截断 / trace_ids / old_id+new_id）。

INVARIANT: 异常是 fail-closed 信号——store/retrieve/consolidate 失败时抛
`MemoryError` 子类，由 middleware 统一捕获转优雅降级（不静默吞）。
"""

from __future__ import annotations


class MemoryError(Exception):
    """Memory 错误基类。带 details dict，__str__ 自动展开。"""

    def __init__(self, message: str = "", *, details: dict | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __str__(self) -> str:
        if not self.details:
            return self.message
        parts = [f"{k}={v!r}" for k, v in self.details.items()]
        return f"{self.message} ({', '.join(parts)})"


class MemoryNotFoundError(MemoryError):
    """记忆找不到。带 trace_id。"""

    def __init__(self, trace_id: str) -> None:
        super().__init__(
            f"memory trace not found: {trace_id}",
            details={"trace_id": trace_id},
        )


class MemoryStoreError(MemoryError):
    """存储失败。带 operation + path。"""

    def __init__(self, message: str, *, operation: str, path: str = "") -> None:
        super().__init__(message, details={"operation": operation, "path": path})


class MemoryRetrieveError(MemoryError):
    """检索失败。带 query（超 100 字符自动截断）。"""

    def __init__(self, message: str, *, query: str) -> None:
        truncated = query[:100] + "..." if len(query) > 100 else query
        super().__init__(message, details={"query": truncated})


class MemoryConsolidateError(MemoryError):
    """巩固失败。带 trace_ids。"""

    def __init__(self, message: str, *, trace_ids: list[str]) -> None:
        super().__init__(message, details={"trace_ids": trace_ids})


class MemoryConflictError(MemoryError):
    """矛盾冲突。带 old_id + new_id。"""

    def __init__(self, message: str, *, old_id: str, new_id: str) -> None:
        super().__init__(message, details={"old_id": old_id, "new_id": new_id})
