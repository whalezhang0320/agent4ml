"""Cross-process file lock — fcntl (Unix) / msvcrt (Windows).

仅导出 3 函数（无 context manager）。sync/async 统一用显式 open/lock/unlock/close，
同一 file 对象贯穿。仿 deer-flow。
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    import fcntl
except ImportError:  # Windows
    fcntl = None  # type: ignore[assignment]
    import msvcrt


def open_lock_file(lock_path: Path):
    """打开锁文件。自动创建父目录。

    Returns:
        已打开的文件对象（append 模式）。
    """
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    return open(lock_path, "a", encoding="utf-8")


def lock_file_exclusive(lock_file) -> None:
    """获取排他锁。阻塞直到获取。

    Unix: fcntl.flock(LOCK_EX)
    Windows: msvcrt.locking(LK_LOCK)
    """
    if fcntl is not None:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        return
    lock_file.seek(0)
    msvcrt.locking(lock_file.fileno(), msvcrt.LK_LOCK, 1)


def unlock_file(lock_file) -> None:
    """释放锁。

    Unix: fcntl.flock(LOCK_UN)
    Windows: msvcrt.locking(LK_UNLCK)
    """
    if fcntl is not None:
        fcntl.flock(lock_file, fcntl.LOCK_UN)
        return
    lock_file.seek(0)
    msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
