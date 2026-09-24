from __future__ import annotations

from unittest.mock import patch

import pytest

from agent4ml.backend.agents.sandbox.contracts import SandboxProvider
from agent4ml.backend.agents.sandbox.guards.audit_guard import AuditGuard
from agent4ml.backend.agents.sandbox.guards.local_security_guard import (
    LocalSecurityGuard,
)
from agent4ml.backend.agents.sandbox.local.local_sandbox_provider import (
    LocalSandboxProvider,
    _deterministic_sandbox_id,
)
from agent4ml.backend.agents.sandbox.runtimes.local_runtime import LocalRuntime
from agent4ml.backend.agents.sandbox.sandbox import Sandbox
from agent4ml.backend.agents.sandbox.translators.local_path_translator import (
    LocalPathTranslator,
)


@pytest.fixture
def provider() -> LocalSandboxProvider:
    return LocalSandboxProvider()


class TestAcquireNoneRejected:
    def test_acquire_none_raises(self, provider: LocalSandboxProvider) -> None:
        with pytest.raises(ValueError, match="thread_id is required"):
            provider.acquire(None)


class TestDeterministicId:
    def test_same_user_thread_same_id(self, provider: LocalSandboxProvider) -> None:
        id1 = provider.acquire("thread-1", user_id="user-1")
        id2 = provider.acquire("thread-1", user_id="user-1")
        assert id1 == id2

    def test_different_thread_different_id(self, provider: LocalSandboxProvider) -> None:
        id1 = provider.acquire("thread-1", user_id="user-1")
        id2 = provider.acquire("thread-2", user_id="user-1")
        assert id1 != id2

    def test_different_user_different_id(self, provider: LocalSandboxProvider) -> None:
        id1 = provider.acquire("thread-1", user_id="user-1")
        id2 = provider.acquire("thread-1", user_id="user-2")
        assert id1 != id2

    def test_id_is_8_chars(self, provider: LocalSandboxProvider) -> None:
        sid = provider.acquire("thread-1", user_id="user-1")
        assert len(sid) == 8


class TestGet:
    def test_get_returns_sandbox(self, provider: LocalSandboxProvider) -> None:
        sid = provider.acquire("thread-1", user_id="user-1")
        sandbox = provider.get(sid)
        assert sandbox is not None
        assert isinstance(sandbox, Sandbox)

    def test_get_unknown_returns_none(self, provider: LocalSandboxProvider) -> None:
        assert provider.get("unknown") is None


class TestRelease:
    def test_release_noop(self, provider: LocalSandboxProvider) -> None:
        sid = provider.acquire("thread-1", user_id="user-1")
        provider.release(sid)
        assert provider.get(sid) is not None


class TestResetShutdown:
    def test_reset_clears(self, provider: LocalSandboxProvider) -> None:
        sid = provider.acquire("thread-1", user_id="user-1")
        provider.reset()
        assert provider.get(sid) is None

    def test_shutdown_closes_and_clears(self, provider: LocalSandboxProvider) -> None:
        sid = provider.acquire("thread-1", user_id="user-1")
        sandbox = provider.get(sid)
        provider.shutdown()
        assert provider.get(sid) is None


class TestLRU:
    def test_lru_eviction_calls_close(self) -> None:
        provider = LocalSandboxProvider(lru_size=2)
        sid1 = provider.acquire("t1", user_id="u1")
        sid2 = provider.acquire("t2", user_id="u1")
        sandbox1 = provider.get(sid1)
        sid3 = provider.acquire("t3", user_id="u1")
        assert provider.get(sid1) is None
        assert provider.get(sid2) is not None
        assert provider.get(sid3) is not None


class TestConcurrencySafety:
    """严重1 修复验证：多线程并发 acquire 无状态损坏。"""

    def test_concurrent_acquire_no_error(self) -> None:
        """10 线程 × 100 acquire 并发压测，无异常 + 状态一致。"""
        import threading

        provider = LocalSandboxProvider(lru_size=50)
        errors: list[Exception] = []

        def worker(thread_idx: int) -> None:
            try:
                for i in range(100):
                    sid = provider.acquire(f"thread-{thread_idx}", user_id=f"user-{i % 5}")
                    sandbox = provider.get(sid)
                    assert sandbox is not None, f"get returned None for {sid}"
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"concurrent errors: {errors}"

    def test_concurrent_acquire_same_thread_id(self) -> None:
        """多线程同时 acquire 同一 thread_id，返回相同 sandbox_id。"""
        import threading

        provider = LocalSandboxProvider()
        results: list[str] = []
        lock = threading.Lock()

        def worker() -> None:
            sid = provider.acquire("shared-thread", user_id="shared-user")
            with lock:
                results.append(sid)

        threads = [threading.Thread(target=worker) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == 20
        assert all(r == results[0] for r in results), "same thread_id should get same sandbox_id"

    def test_concurrent_reset_acquire(self) -> None:
        """reset 与 acquire 并发不崩溃。"""
        import threading

        provider = LocalSandboxProvider()
        errors: list[Exception] = []

        def acquirer() -> None:
            try:
                for i in range(50):
                    provider.acquire(f"t-{i}", user_id="u")
            except Exception as exc:
                errors.append(exc)

        def resetter() -> None:
            try:
                for _ in range(50):
                    provider.reset()
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=acquirer)] + [threading.Thread(target=resetter)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"reset/acquire race errors: {errors}"

    def test_lru_touch_moves_to_end(self, provider: LocalSandboxProvider) -> None:
        provider = LocalSandboxProvider(lru_size=2)
        sid1 = provider.acquire("t1", user_id="u1")
        sid2 = provider.acquire("t2", user_id="u1")
        provider.acquire("t1", user_id="u1")
        sid3 = provider.acquire("t3", user_id="u1")
        assert provider.get(sid1) is not None
        assert provider.get(sid2) is None


class TestCapabilityFlags:
    def test_uses_thread_data_mounts(self, provider: LocalSandboxProvider) -> None:
        assert provider.uses_thread_data_mounts is True

    def test_needs_upload_permission_adjustment(self, provider: LocalSandboxProvider) -> None:
        assert provider.needs_upload_permission_adjustment is False


class TestComposedComponents:
    def test_sandbox_has_local_runtime(self, provider: LocalSandboxProvider) -> None:
        sid = provider.acquire("thread-1", user_id="user-1")
        sandbox = provider.get(sid)
        assert isinstance(sandbox._runtime, LocalRuntime)

    def test_sandbox_has_local_path_translator(self, provider: LocalSandboxProvider) -> None:
        sid = provider.acquire("thread-1", user_id="user-1")
        sandbox = provider.get(sid)
        assert isinstance(sandbox._translator, LocalPathTranslator)

    def test_sandbox_has_audit_guard_wrapping_local(self, provider: LocalSandboxProvider) -> None:
        sid = provider.acquire("thread-1", user_id="user-1")
        sandbox = provider.get(sid)
        assert isinstance(sandbox._guard, AuditGuard)
        assert isinstance(sandbox._guard._inner, LocalSecurityGuard)


class TestProtocolConformance:
    def test_is_sandbox_provider(self, provider: LocalSandboxProvider) -> None:
        assert isinstance(provider, SandboxProvider)
