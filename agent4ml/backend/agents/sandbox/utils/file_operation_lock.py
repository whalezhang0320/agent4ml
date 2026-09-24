"""文件操作锁——进程内 WeakValueDictionary。

已知限制（S11 文档声明）：
- 锁仅进程内，不跨进程。Docker 模式下 read_file + write_file 是两次独立 HTTP 调用，
  别进程修改文件 → read 拿旧内容 → write 覆盖别进程的修改 → 静默丢数据（TOCTOU）。
- 当前 Agent4ML 单进程运行，此限制可接受。多 worker 部署需引入容器内 flock 或 SDK 原子接口。
- 长期方案见 design_docs/todo_docs/02-sandbox-security-vulnerabilities-fix.md §11 B6。
"""
from __future__ import annotations

import threading
import weakref

_LockKey = tuple[str, str]  # (sandbox_id, virtual_path)
_FILE_OPERATION_LOCKS: weakref.WeakValueDictionary[_LockKey, threading.Lock] = (
    weakref.WeakValueDictionary()
)
_FILE_OPERATION_LOCKS_GUARD = threading.Lock()


def get_file_operation_lock_key(sandbox_id: str, virtual_path: str) -> _LockKey:
    """构造锁 key。用虚拟路径（Grill #6：工具层拿到的是虚拟路径）。"""
    return (sandbox_id, virtual_path)


def get_file_operation_lock(sandbox_id: str, virtual_path: str) -> threading.Lock:
    """获取 (sandbox_id, virtual_path) 对应的锁。不存在则创建。

    WeakValueDictionary: 锁无引用时自动 GC，防长跑进程内存泄漏。
    key 用虚拟路径（Grill #6）。
    """
    lock_key = get_file_operation_lock_key(sandbox_id, virtual_path)
    with _FILE_OPERATION_LOCKS_GUARD:
        lock = _FILE_OPERATION_LOCKS.get(lock_key)
        if lock is None:
            lock = threading.Lock()
            _FILE_OPERATION_LOCKS[lock_key] = lock
        return lock
