from __future__ import annotations

import gc
import weakref

from agent4ml.backend.agents.sandbox.utils.file_operation_lock import (
    get_file_operation_lock,
    get_file_operation_lock_key,
)


class TestGetFileOperationLockKey:
    def test_construct_key(self) -> None:
        key = get_file_operation_lock_key("sb-1", "/mnt/x")
        assert key == ("sb-1", "/mnt/x")


class TestGetFileOperationLock:
    def test_same_key_same_lock(self) -> None:
        lock1 = get_file_operation_lock("sb-1", "/mnt/x")
        lock2 = get_file_operation_lock("sb-1", "/mnt/x")
        assert lock1 is lock2

    def test_different_sandbox_different_lock(self) -> None:
        lock1 = get_file_operation_lock("sb-1", "/mnt/x")
        lock2 = get_file_operation_lock("sb-2", "/mnt/x")
        assert lock1 is not lock2

    def test_different_path_different_lock(self) -> None:
        lock1 = get_file_operation_lock("sb-1", "/mnt/x")
        lock2 = get_file_operation_lock("sb-1", "/mnt/y")
        assert lock1 is not lock2

    def test_lock_is_threading_lock(self) -> None:
        import threading

        lock = get_file_operation_lock("sb-test", "/mnt/test")
        assert type(lock) is type(threading.Lock())

    def test_weakvalue_gc(self) -> None:
        """WeakValueDictionary: 锁无引用后自动 GC。"""
        lock = get_file_operation_lock("sb-gc", "/mnt/gc")
        ref = weakref.ref(lock)
        assert ref() is not None
        del lock
        gc.collect()
        assert ref() is None
