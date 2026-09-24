from __future__ import annotations

import sys
import time
import types
from unittest.mock import MagicMock, patch

import pytest

try:
    import anyio  # noqa: F401
    HAS_ANYIO = True
except ImportError:
    HAS_ANYIO = False

# Inject mock agent_sandbox (for DockerRuntime lazy import in _make_sandbox)
if "agent_sandbox" not in sys.modules:
    _mock_mod = types.ModuleType("agent_sandbox")
    _mock_mod.Sandbox = MagicMock
    sys.modules["agent_sandbox"] = _mock_mod

from agent4ml.backend.agents.sandbox.docker.docker_sandbox_provider import (  # noqa: E402
    DockerSandboxProvider,
    _deterministic_sandbox_id,
)
from agent4ml.backend.agents.sandbox.sandbox import Sandbox  # noqa: E402
from agent4ml.backend.agents.sandbox.types import SandboxInfo  # noqa: E402


def _make_config(**kwargs):
    """Build a mock SandboxConfig."""
    config = MagicMock()
    config.image = kwargs.get("image", "")
    config.port = kwargs.get("port", 0)
    config.container_prefix = kwargs.get("container_prefix", "")
    config.environment = kwargs.get("environment", {})
    config.idle_timeout = kwargs.get("idle_timeout", 0)
    config.replicas = kwargs.get("replicas", 0)
    config.mounts = kwargs.get("mounts", [])
    return config


def _make_provider(**kwargs) -> DockerSandboxProvider:
    """Build a DockerSandboxProvider with mocked provisioner."""
    config = kwargs.pop("sandbox_config", None)
    with patch(
        "agent4ml.backend.agents.sandbox.docker.docker_sandbox_provider."
        "LocalContainerBackend._detect_runtime",
        return_value="docker",
    ):
        return DockerSandboxProvider(sandbox_config=config, **kwargs)


def _make_info(sandbox_id="abc", url="http://localhost:9090") -> SandboxInfo:
    return SandboxInfo(sandbox_id=sandbox_id, sandbox_url=url, container_name=f"agent4ml-sandbox-{sandbox_id}")


class TestDeterministicSandboxId:
    def test_format(self) -> None:
        sid = _deterministic_sandbox_id("user1", "thread1")
        assert len(sid) == 8
        assert all(c in "0123456789abcdef" for c in sid)

    def test_deterministic(self) -> None:
        assert _deterministic_sandbox_id("u", "t") == _deterministic_sandbox_id("u", "t")

    def test_different_inputs_different_ids(self) -> None:
        assert _deterministic_sandbox_id("u1", "t") != _deterministic_sandbox_id("u2", "t")

    def test_none_defaults(self) -> None:
        sid = _deterministic_sandbox_id(None, None)
        assert len(sid) == 8


class TestInit:
    def test_config_extraction(self) -> None:
        config = _make_config(
            image="img:latest", port=9000, container_prefix="test-sb",
            environment={"NODE_ENV": "prod"}, idle_timeout=300, replicas=5,
        )
        p = _make_provider(sandbox_config=config)
        assert p._idle_timeout == 300
        assert p._replicas == 5

    def test_sandbox_root_type_layer(self) -> None:
        p = _make_provider()
        assert "aio_docker" in str(p._sandbox_root)

    def test_extra_mounts_from_config(self) -> None:
        mount = MagicMock()
        mount.host_path = "/skills"
        mount.container_path = "/mnt/agent4ml/skills"
        mount.read_only = True
        config = _make_config(mounts=[mount])
        p = _make_provider(sandbox_config=config)
        assert len(p._extra_mounts) == 1
        assert p._extra_mounts[0].container_path == "/mnt/agent4ml/skills"
        assert p._extra_mounts[0].local_path == "/skills"
        assert p._extra_mounts[0].read_only is True

    def test_extra_mounts_empty_without_config(self) -> None:
        p = _make_provider()
        assert p._extra_mounts == []

    def test_path_mappings_not_in_extra_mounts(self) -> None:
        from agent4ml.backend.agents.sandbox.types import PathMapping
        p = _make_provider(
            path_mappings=[PathMapping("/mnt/agent4ml/user-data/workspace", "/host/ws")],
        )
        # path_mappings stored but NOT in extra_mounts
        assert len(p._path_mappings) == 1
        assert p._extra_mounts == []

    def test_state_dicts_initialized(self) -> None:
        p = _make_provider()
        assert p._sandboxes == {}
        assert p._warm_pool == {}
        assert p._thread_sandboxes == {}
        assert p._shutdown_called is False


class TestAcquireThreadIdRequired:
    def test_none_raises(self) -> None:
        p = _make_provider()
        with pytest.raises(ValueError, match="thread_id"):
            p.acquire(None)

    @pytest.mark.skipif(not HAS_ANYIO, reason="anyio not installed")
    @pytest.mark.anyio
    async def test_async_none_raises(self) -> None:
        p = _make_provider()
        with pytest.raises(ValueError, match="thread_id"):
            await p.acquire_async(None)


class TestAcquireLayer1:
    def test_cache_hit(self) -> None:
        p = _make_provider()
        info = _make_info("abc")
        sandbox = MagicMock()
        p._sandboxes["abc"] = sandbox
        p._sandbox_infos["abc"] = info
        p._thread_sandboxes[("default", "t1")] = "abc"
        p._backend.is_alive = MagicMock(return_value=True)

        result = p.acquire("t1")
        assert result == "abc"
        p._backend.is_alive.assert_called_once_with(info)

    def test_cache_miss_returns_none(self) -> None:
        p = _make_provider()
        sid = _deterministic_sandbox_id("default", "t1")
        info = _make_info(sid)
        p._backend.discover = MagicMock(return_value=info)
        p._backend.is_alive = MagicMock(return_value=True)
        result = p.acquire("t1")
        assert result == sid

    def test_health_check_fail_drops(self) -> None:
        p = _make_provider()
        sid_cached = _deterministic_sandbox_id("default", "t1")
        info_cached = _make_info(sid_cached)
        sandbox = MagicMock()
        p._sandboxes[sid_cached] = sandbox
        p._sandbox_infos[sid_cached] = info_cached
        p._thread_sandboxes[("default", "t1")] = sid_cached
        p._backend.is_alive = MagicMock(return_value=False)
        p._backend.destroy = MagicMock()
        p._backend.discover = MagicMock(return_value=None)
        p._backend.create = MagicMock(return_value=info_cached)
        with patch(
            "agent4ml.backend.agents.sandbox.docker.docker_sandbox_provider."
            "wait_for_sandbox_ready", return_value=True,
        ):
            result = p.acquire("t1")
        assert result == sid_cached
        sandbox.close.assert_called_once()
        assert p._backend.destroy.call_count >= 1

    def test_health_check_none_continues(self) -> None:
        """is_alive=None (unknown) should NOT drop 鈥?continue with cached sandbox."""
        p = _make_provider()
        info = _make_info("abc")
        p._sandboxes["abc"] = MagicMock()
        p._sandbox_infos["abc"] = info
        p._thread_sandboxes[("default", "t1")] = "abc"
        p._backend.is_alive = MagicMock(return_value=None)

        result = p.acquire("t1")
        assert result == "abc"  # None 鈫?don't drop, reuse cached


class TestAcquireLayer15:
    def test_warm_pool_reclaim(self) -> None:
        p = _make_provider()
        sid = _deterministic_sandbox_id("default", "t1")
        info = _make_info(sid)
        p._warm_pool[sid] = (info, 0.0)
        p._backend.is_alive = MagicMock(return_value=True)

        result = p.acquire("t1")
        assert result == sid
        assert sid not in p._warm_pool
        assert sid in p._sandboxes

    def test_warm_pool_health_fail_drops(self) -> None:
        p = _make_provider()
        sid = _deterministic_sandbox_id("default", "t1")
        info = _make_info(sid)
        p._warm_pool[sid] = (info, 0.0)
        p._backend.is_alive = MagicMock(return_value=False)
        p._backend.destroy = MagicMock()
        p._backend.discover = MagicMock(return_value=None)
        p._backend.create = MagicMock(return_value=info)
        with patch(
            "agent4ml.backend.agents.sandbox.docker.docker_sandbox_provider."
            "wait_for_sandbox_ready", return_value=True,
        ):
            p.acquire("t1")
        assert sid not in p._warm_pool
        assert p._backend.destroy.call_count >= 1

    def test_warm_pool_empty_continues_to_layer2(self) -> None:
        p = _make_provider()
        sid = _deterministic_sandbox_id("default", "t1")
        info = _make_info(sid)
        p._backend.is_alive = MagicMock(return_value=True)
        p._backend.discover = MagicMock(return_value=info)
        result = p.acquire("t1")
        assert result == sid


class TestGet:
    def test_found(self) -> None:
        p = _make_provider()
        sandbox = MagicMock()
        p._sandboxes["abc"] = sandbox
        assert p.get("abc") is sandbox

    def test_not_found(self) -> None:
        p = _make_provider()
        assert p.get("xyz") is None

    def test_updates_last_activity(self) -> None:
        p = _make_provider()
        p._sandboxes["abc"] = MagicMock()
        assert "abc" not in p._last_activity
        p.get("abc")
        assert "abc" in p._last_activity


class TestMakeSandbox:
    def test_constructs_components(self) -> None:
        p = _make_provider()
        info = _make_info("abc")
        sandbox = p._make_sandbox("abc", info)
        assert isinstance(sandbox, Sandbox)
        assert sandbox.id == "abc"


class TestGetSandboxInfo:
    def test_returns_info_when_exists(self) -> None:
        p = _make_provider()
        info = _make_info("abc")
        p._sandbox_infos["abc"] = info

        result = p.get_sandbox_info("abc")

        assert result is info

    def test_returns_none_when_not_exists(self) -> None:
        p = _make_provider()

        result = p.get_sandbox_info("nonexistent")

        assert result is None

    def test_returns_info_with_sandbox_url(self) -> None:
        p = _make_provider()
        info = _make_info("abc", url="http://localhost:8080")
        p._sandbox_infos["abc"] = info

        result = p.get_sandbox_info("abc")

        assert result is not None
        assert result.sandbox_url == "http://localhost:8080"


class TestLocalSandboxProviderGetSandboxInfo:
    def test_returns_none(self) -> None:
        from agent4ml.backend.agents.sandbox.local.local_sandbox_provider import (
            LocalSandboxProvider,
        )

        p = LocalSandboxProvider()
        assert p.get_sandbox_info("any-id") is None


class TestDropUnhealthy:
    def test_removes_from_all_dicts(self) -> None:
        p = _make_provider()
        info = _make_info("abc")
        sandbox = MagicMock()
        p._sandboxes["abc"] = sandbox
        p._sandbox_infos["abc"] = info
        p._thread_sandboxes[("default", "t1")] = "abc"
        p._last_activity["abc"] = 0.0
        p._warm_pool["abc"] = (info, 0.0)
        p._backend.destroy = MagicMock()

        p._drop_unhealthy("abc", "test reason")

        assert "abc" not in p._sandboxes
        assert "abc" not in p._sandbox_infos
        assert ("default", "t1") not in p._thread_sandboxes
        assert "abc" not in p._last_activity
        assert "abc" not in p._warm_pool
        sandbox.close.assert_called_once()
        p._backend.destroy.assert_called_once_with(info)

    def test_destroy_failure_does_not_raise(self) -> None:
        p = _make_provider()
        info = _make_info("abc")
        p._sandbox_infos["abc"] = info
        p._backend.destroy = MagicMock(side_effect=RuntimeError("boom"))
        p._drop_unhealthy("abc", "test")  # should not raise


class TestDropUnhealthyExpectedInfo:
    """S7: destroy 前二次 discover 校验，避免误杀新容器。"""

    def test_skip_destroy_when_container_changed(self) -> None:
        """expected_info != current discover → 跳过 destroy。"""
        p = _make_provider()
        old_info = _make_info("abc", "http://localhost:9000")
        new_info = _make_info("abc", "http://localhost:9001")  # 同 ID 不同 URL
        p._sandbox_infos["abc"] = old_info
        p._backend.discover = MagicMock(return_value=new_info)
        p._backend.destroy = MagicMock()

        p._drop_unhealthy("abc", "test", expected_info=old_info)

        p._backend.destroy.assert_not_called()  # 跳过 destroy

    def test_destroy_when_container_unchanged(self) -> None:
        """expected_info == current discover → 正常 destroy。"""
        p = _make_provider()
        info = _make_info("abc", "http://localhost:9000")
        p._sandbox_infos["abc"] = info
        p._backend.discover = MagicMock(return_value=info)  # 同 info
        p._backend.destroy = MagicMock()

        p._drop_unhealthy("abc", "test", expected_info=info)

        p._backend.destroy.assert_called_once_with(info)

    def test_destroy_when_container_gone(self) -> None:
        """discover 返回 None（容器已不在）→ 正常 destroy expected_info。"""
        p = _make_provider()
        info = _make_info("abc")
        p._sandbox_infos["abc"] = info
        p._backend.discover = MagicMock(return_value=None)
        p._backend.destroy = MagicMock()

        p._drop_unhealthy("abc", "test", expected_info=info)

        p._backend.destroy.assert_called_once_with(info)

    def test_no_expected_info_always_destroys(self) -> None:
        """不传 expected_info（向后兼容）→ 始终 destroy。"""
        p = _make_provider()
        info = _make_info("abc")
        p._sandbox_infos["abc"] = info
        p._backend.discover = MagicMock(return_value=_make_info("abc", "http://other:9999"))
        p._backend.destroy = MagicMock()

        p._drop_unhealthy("abc", "test")  # 无 expected_info

        p._backend.destroy.assert_called_once_with(info)


class TestGetThreadLock:
    def test_same_thread_same_lock(self) -> None:
        p = _make_provider()
        lock1 = p._get_thread_lock("t1", "u1")
        lock2 = p._get_thread_lock("t1", "u1")
        assert lock1 is lock2

    def test_different_thread_different_lock(self) -> None:
        p = _make_provider()
        lock1 = p._get_thread_lock("t1", "u1")
        lock2 = p._get_thread_lock("t2", "u1")
        assert lock1 is not lock2


class TestAcquireAsyncCancelSafety:
    """S5: acquire_async cancel 后同 thread_id 可重新 acquire（无死锁）。"""

    @pytest.mark.skipif(not HAS_ANYIO, reason="anyio not installed")
    @pytest.mark.anyio
    async def test_cancel_during_acquire_no_deadlock(self) -> None:
        """cancel 在 _acquire_internal_async 等待期间 → finally release → 无死锁。"""
        import asyncio

        p = _make_provider()
        sid = _deterministic_sandbox_id("default", "t1")
        info = _make_info(sid)
        p._backend.discover = MagicMock(return_value=info)
        p._backend.create = MagicMock()
        p._backend.is_alive = MagicMock(return_value=True)

        # 用 event 让 _acquire_internal_async 阻塞，给 cancel 窗口
        block_event = asyncio.Event()

        original_internal = p._acquire_internal_async

        async def blocking_internal(thread_id, *, user_id):
            await block_event.wait()  # 阻塞直到 set
            return await original_internal(thread_id, user_id=user_id)

        p._acquire_internal_async = blocking_internal

        # 启动 acquire_async task
        task = asyncio.create_task(p.acquire_async("t1"))
        await asyncio.sleep(0.1)  # 等 task 进入 blocking_internal

        # cancel task
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass

        # 验证锁已释放：同 thread_id 可重新 acquire
        block_event.set()  # 解除阻塞（虽然 task 已 cancel）
        # 重新 mock 为正常路径
        p._acquire_internal_async = original_internal
        result = p.acquire("t1")  # 同步 acquire 验证锁不阻塞
        assert result == sid

    @pytest.mark.skipif(not HAS_ANYIO, reason="anyio not installed")
    @pytest.mark.anyio
    async def test_cancel_before_acquire_no_release_error(self) -> None:
        """cancel 在 acquire 等待期间（未持有锁）→ finally 不 release → 无 RuntimeError。"""
        import asyncio

        p = _make_provider()
        # 让 thread_lock.acquire 阻塞（被另一 holder 持有）
        lock = p._get_thread_lock("t1", "default")
        lock.acquire()  # 模拟另一线程持有

        # 启动 acquire_async task（会阻塞在 acquire）
        task = asyncio.create_task(p.acquire_async("t1"))
        await asyncio.sleep(0.1)

        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass

        # 释放锁，验证后续 acquire 正常
        lock.release()
        # 如果 cancel 路径误 release 了未持有的锁，这里 lock 状态会错乱
        # 验证 lock 仍可正常 acquire/release
        assert lock.acquire(blocking=False) is True
        lock.release()


class TestDiscoverOrCreate:
    def test_discover_hit(self) -> None:
        p = _make_provider()
        sid = _deterministic_sandbox_id("default", "t1")
        info = _make_info(sid)
        p._backend.discover = MagicMock(return_value=info)
        p._backend.create = MagicMock()
        p._backend.is_alive = MagicMock(return_value=True)

        result = p.acquire("t1")
        assert result == sid
        p._backend.discover.assert_called_once_with(sid)
        p._backend.create.assert_not_called()

    def test_discover_miss_create_success(self) -> None:
        p = _make_provider()
        sid = _deterministic_sandbox_id("default", "t1")
        info = _make_info(sid)
        p._backend.discover = MagicMock(return_value=None)
        p._backend.create = MagicMock(return_value=info)
        p._backend.is_alive = MagicMock(return_value=True)

        with patch(
            "agent4ml.backend.agents.sandbox.docker.docker_sandbox_provider."
            "wait_for_sandbox_ready", return_value=True,
        ):
            result = p.acquire("t1")
        assert result == sid
        p._backend.create.assert_called_once()

    def test_readiness_fail_destroy_and_raise(self) -> None:
        p = _make_provider()
        sid = _deterministic_sandbox_id("default", "t1")
        info = _make_info(sid)
        p._backend.discover = MagicMock(return_value=None)
        p._backend.create = MagicMock(return_value=info)
        p._backend.destroy = MagicMock()
        p._backend.is_alive = MagicMock(return_value=True)

        with patch(
            "agent4ml.backend.agents.sandbox.docker.docker_sandbox_provider."
            "wait_for_sandbox_ready", return_value=False,
        ):
            with pytest.raises(RuntimeError, match="failed readiness"):
                p.acquire("t1")
        p._backend.destroy.assert_called_once_with(info)

    def test_extra_mounts_passed_to_create(self) -> None:
        from agent4ml.backend.agents.sandbox.types import PathMapping
        mount = MagicMock()
        mount.host_path = "/skills"
        mount.container_path = "/mnt/agent4ml/skills"
        mount.read_only = True
        config = _make_config(mounts=[mount])
        p = _make_provider(sandbox_config=config)

        sid = _deterministic_sandbox_id("default", "t1")
        info = _make_info(sid)
        p._backend.discover = MagicMock(return_value=None)
        p._backend.create = MagicMock(return_value=info)
        p._backend.is_alive = MagicMock(return_value=True)

        with patch(
            "agent4ml.backend.agents.sandbox.docker.docker_sandbox_provider."
            "wait_for_sandbox_ready", return_value=True,
        ):
            p.acquire("t1")

        call_kwargs = p._backend.create.call_args
        extra = call_kwargs.kwargs.get("extra_mounts")
        assert extra is not None
        assert any(m.container_path == "/mnt/agent4ml/skills" for m in extra)


class TestEnforceReplicas:
    def test_evict_when_over_limit(self) -> None:
        p = _make_provider(replicas=2)
        info1 = _make_info("w1")
        info2 = _make_info("w2")
        p._warm_pool["w1"] = (info1, 100.0)
        p._warm_pool["w2"] = (info2, 200.0)
        p._sandboxes["active1"] = MagicMock()
        p._sandbox_infos["active1"] = _make_info("active1")
        p._backend.destroy = MagicMock()

        p._enforce_replicas()
        # total = 1 active + 2 warm = 3 >= replicas=2 鈫?evict oldest warm (w1, ts=100)
        p._backend.destroy.assert_called_once_with(info1)
        assert "w1" not in p._warm_pool
        assert "w2" in p._warm_pool

    def test_no_evict_when_under_limit(self) -> None:
        p = _make_provider(replicas=5)
        p._warm_pool["w1"] = (_make_info("w1"), 100.0)
        p._backend.destroy = MagicMock()

        p._enforce_replicas()
        p._backend.destroy.assert_not_called()

    def test_no_evict_when_warm_pool_empty(self) -> None:
        p = _make_provider(replicas=1)
        p._sandboxes["a1"] = MagicMock()
        p._backend.destroy = MagicMock()

        p._enforce_replicas()
        p._backend.destroy.assert_not_called()


class TestEvictOldestWarm:
    def test_evict_oldest_by_release_ts(self) -> None:
        p = _make_provider()
        info_old = _make_info("old")
        info_new = _make_info("new")
        p._warm_pool["old"] = (info_old, 100.0)
        p._warm_pool["new"] = (info_new, 200.0)
        p._backend.destroy = MagicMock()

        p._evict_oldest_warm()
        p._backend.destroy.assert_called_once_with(info_old)
        assert "old" not in p._warm_pool
        assert "new" in p._warm_pool

    def test_noop_when_empty(self) -> None:
        p = _make_provider()
        p._backend.destroy = MagicMock()
        p._evict_oldest_warm()
        p._backend.destroy.assert_not_called()

    def test_destroy_failure_does_not_raise(self) -> None:
        p = _make_provider()
        p._warm_pool["w1"] = (_make_info("w1"), 100.0)
        p._backend.destroy = MagicMock(side_effect=RuntimeError("boom"))
        p._evict_oldest_warm()  # should not raise


class TestRelease:
    def test_moves_to_warm_pool(self) -> None:
        p = _make_provider()
        sid = "abc"
        sandbox = MagicMock()
        info = _make_info(sid)
        p._sandboxes[sid] = sandbox
        p._sandbox_infos[sid] = info
        p._thread_sandboxes[("default", "t1")] = sid
        p._last_activity[sid] = 0.0
        p._backend.destroy = MagicMock()

        p.release(sid)

        assert sid not in p._sandboxes
        assert sid in p._warm_pool
        sandbox.close.assert_called_once()
        p._backend.destroy.assert_not_called()

    def test_release_unknown_sandbox_noop(self) -> None:
        p = _make_provider()
        p.release("nonexistent")
        assert p._warm_pool == {}


class TestShutdown:
    def test_idempotent(self) -> None:
        p = _make_provider()
        p._backend.destroy = MagicMock()
        p.shutdown()
        p.shutdown()
        assert p._shutdown_called is True

    def test_destroys_active_and_warm(self) -> None:
        p = _make_provider()
        active_info = _make_info("a1")
        warm_info = _make_info("w1")
        p._sandboxes["a1"] = MagicMock()
        p._sandbox_infos["a1"] = active_info
        p._warm_pool["w1"] = (warm_info, 0.0)
        p._backend.destroy = MagicMock()

        p.shutdown()

        assert p._warm_pool == {}
        p._backend.destroy.assert_any_call(active_info)
        p._backend.destroy.assert_any_call(warm_info)

    def test_destroy_failure_does_not_raise(self) -> None:
        p = _make_provider()
        p._sandboxes["a1"] = MagicMock()
        p._sandbox_infos["a1"] = _make_info("a1")
        p._backend.destroy = MagicMock(side_effect=RuntimeError("boom"))
        p.shutdown()


class TestReconcileOrphans:
    def test_adopt_running(self) -> None:
        p = _make_provider()
        info1 = _make_info("orphan1", "http://localhost:9090")
        info2 = _make_info("orphan2", "http://localhost:9091")
        p._backend.list_running = MagicMock(return_value=[info1, info2])

        p._reconcile_orphans()

        assert "orphan1" in p._warm_pool
        assert "orphan2" in p._warm_pool

    def test_skip_empty_url(self) -> None:
        p = _make_provider()
        info_no_url = SandboxInfo(sandbox_id="no_url", sandbox_url="", container_name="test")
        info_with_url = _make_info("has_url")
        p._backend.list_running = MagicMock(return_value=[info_no_url, info_with_url])

        p._reconcile_orphans()

        assert "no_url" not in p._warm_pool
        assert "has_url" in p._warm_pool

    def test_skip_already_tracked(self) -> None:
        p = _make_provider()
        info = _make_info("tracked")
        p._sandboxes["tracked"] = MagicMock()
        p._backend.list_running = MagicMock(return_value=[info])

        p._reconcile_orphans()

        assert "tracked" not in p._warm_pool

    def test_list_running_failure_does_not_raise(self) -> None:
        p = _make_provider()
        p._backend.list_running = MagicMock(side_effect=RuntimeError("docker error"))
        p._reconcile_orphans()


class TestIdleChecker:
    def test_cleanup_idle_warm(self) -> None:
        p = _make_provider(idle_timeout=0)  # disable thread
        info = _make_info("w1")
        p._warm_pool["w1"] = (info, 0.0)
        p._backend.destroy = MagicMock()

        p._cleanup_idle()

        p._backend.destroy.assert_called_once_with(info)

    def test_cleanup_idle_active(self) -> None:
        p = _make_provider(idle_timeout=0)
        info = _make_info("a1")
        p._sandboxes["a1"] = MagicMock()
        p._sandbox_infos["a1"] = info
        p._thread_sandboxes[("default", "t1")] = "a1"
        p._last_activity["a1"] = 0.0
        p._backend.destroy = MagicMock()

        p._cleanup_idle()

        p._backend.destroy.assert_called_once_with(info)

    def test_active_re_acquired_not_destroyed(self) -> None:
        p = _make_provider(idle_timeout=0)
        p._last_activity["a1"] = time.time()
        p._backend.destroy = MagicMock()

        p._cleanup_idle()

        p._backend.destroy.assert_not_called()
