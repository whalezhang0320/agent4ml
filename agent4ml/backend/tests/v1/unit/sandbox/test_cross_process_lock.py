from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

try:
    import anyio  # noqa: F401
    HAS_ANYIO = True
except ImportError:
    HAS_ANYIO = False

from agent4ml.backend.agents.sandbox.docker.cross_process_lock import (
    lock_file_exclusive,
    open_lock_file,
    unlock_file,
)


class TestOpenLockFile:
    def test_creates_parent_dirs(self, tmp_path) -> None:
        lock_path = tmp_path / "aio_docker" / "abc.lock"
        f = open_lock_file(lock_path)
        try:
            assert lock_path.parent.exists()
            assert f is not None
        finally:
            f.close()

    def test_returns_open_file(self, tmp_path) -> None:
        lock_path = tmp_path / "test.lock"
        f = open_lock_file(lock_path)
        try:
            assert not f.closed
        finally:
            f.close()

    def test_existing_file_no_error(self, tmp_path) -> None:
        lock_path = tmp_path / "test.lock"
        lock_path.write_text("")
        f = open_lock_file(lock_path)
        try:
            assert not f.closed
        finally:
            f.close()


class TestLockFileExclusive:
    def test_unix_fcntl(self) -> None:
        mock_fcntl = MagicMock()
        mock_file = MagicMock()
        with patch.dict(sys.modules, {"fcntl": mock_fcntl}):
            # Simulate Unix: fcntl is available
            with patch("agent4ml.backend.agents.sandbox.docker.cross_process_lock.fcntl", mock_fcntl):
                lock_file_exclusive(mock_file)
        mock_fcntl.flock.assert_called_once_with(mock_file, mock_fcntl.LOCK_EX)

    def test_windows_msvcrt(self) -> None:
        mock_msvcrt = MagicMock()
        mock_file = MagicMock()
        mock_file.fileno.return_value = 42
        # Simulate Windows: fcntl is None
        with patch("agent4ml.backend.agents.sandbox.docker.cross_process_lock.fcntl", None):
            with patch.dict(sys.modules, {"msvcrt": mock_msvcrt}):
                with patch("agent4ml.backend.agents.sandbox.docker.cross_process_lock.msvcrt", mock_msvcrt):
                    lock_file_exclusive(mock_file)
        mock_file.seek.assert_called_once_with(0)
        mock_msvcrt.locking.assert_called_once_with(42, mock_msvcrt.LK_LOCK, 1)


class TestUnlockFile:
    def test_unix_fcntl(self) -> None:
        mock_fcntl = MagicMock()
        mock_file = MagicMock()
        with patch("agent4ml.backend.agents.sandbox.docker.cross_process_lock.fcntl", mock_fcntl):
            unlock_file(mock_file)
        mock_fcntl.flock.assert_called_once_with(mock_file, mock_fcntl.LOCK_UN)

    def test_windows_msvcrt(self) -> None:
        mock_msvcrt = MagicMock()
        mock_file = MagicMock()
        mock_file.fileno.return_value = 42
        with patch("agent4ml.backend.agents.sandbox.docker.cross_process_lock.fcntl", None):
            with patch.dict(sys.modules, {"msvcrt": mock_msvcrt}):
                with patch("agent4ml.backend.agents.sandbox.docker.cross_process_lock.msvcrt", mock_msvcrt):
                    unlock_file(mock_file)
        mock_file.seek.assert_called_once_with(0)
        mock_msvcrt.locking.assert_called_once_with(42, mock_msvcrt.LK_UNLCK, 1)


class TestIntegration:
    """sync 显式 open/lock/unlock/close 模式验证。"""

    def test_explicit_pattern(self, tmp_path) -> None:
        """验证 sync 路径：open → lock → unlock → close，同一 file 对象。"""
        lock_path = tmp_path / "test.lock"
        lock_file = open_lock_file(lock_path)
        try:
            lock_file_exclusive(lock_file)
            # ... do work ...
            unlock_file(lock_file)
        finally:
            lock_file.close()
        assert lock_file.closed

    @pytest.mark.skipif(not HAS_ANYIO, reason="anyio not installed")
    @pytest.mark.anyio
    async def test_async_pattern(self, tmp_path) -> None:
        """验证 async 路径：asyncio.to_thread 包每步，同一 file 对象。"""
        import asyncio

        lock_path = tmp_path / "async_test.lock"
        lock_file = await asyncio.to_thread(open_lock_file, lock_path)
        try:
            await asyncio.to_thread(lock_file_exclusive, lock_file)
            # ... do work ...
            await asyncio.to_thread(unlock_file, lock_file)
        finally:
            await asyncio.to_thread(lock_file.close)
        assert lock_file.closed
