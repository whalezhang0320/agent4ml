from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from agent4ml.backend.agents.memory.bootstrap import (
    get_memory_worker,
    shutdown_memory_worker,
    start_memory_worker,
)


@pytest.fixture(autouse=True)
def _reset_worker():
    shutdown_memory_worker(timeout=1.0)
    yield
    shutdown_memory_worker(timeout=1.0)


class TestStartMemoryWorker:
    def test_starts_and_returns_worker(self) -> None:
        from agent4ml.backend.agents.memory.worker import MemoryWorker

        manager = MagicMock()
        llm = MagicMock()
        worker = start_memory_worker(manager, llm)

        assert isinstance(worker, MemoryWorker)
        assert worker._thread is not None
        assert worker._thread.is_alive()

    def test_idempotent_returns_same_instance(self) -> None:
        manager = MagicMock()
        llm = MagicMock()
        first = start_memory_worker(manager, llm)
        second = start_memory_worker(manager, llm)

        assert first is second


class TestShutdownMemoryWorker:
    def test_stops_worker(self) -> None:
        manager = MagicMock()
        llm = MagicMock()
        start_memory_worker(manager, llm)

        shutdown_memory_worker(timeout=2.0)

        assert get_memory_worker() is None

    def test_idempotent_no_error(self) -> None:
        shutdown_memory_worker(timeout=1.0)
        shutdown_memory_worker(timeout=1.0)

    def test_no_started_worker_no_error(self) -> None:
        shutdown_memory_worker(timeout=1.0)
        assert get_memory_worker() is None


class TestGetMemoryWorker:
    def test_returns_none_when_not_started(self) -> None:
        assert get_memory_worker() is None

    def test_returns_instance_when_started(self) -> None:
        manager = MagicMock()
        llm = MagicMock()
        worker = start_memory_worker(manager, llm)

        assert get_memory_worker() is worker
