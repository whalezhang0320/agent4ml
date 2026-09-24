"""DockerSandboxProvider — Docker 容器生命周期管理 + warm_pool + idle_checker。

实现 SandboxProvider ABC。编排 LocalContainerBackend（基础设施 CRUD）
+ DockerRuntime（HTTP 调容器）+ DockerPathTranslator + DockerPathGuard。
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import threading
import time
from pathlib import Path

from agent4ml.backend.agents.sandbox.contracts import SandboxProvider
from agent4ml.backend.agents.sandbox.docker.cross_process_lock import (
    lock_file_exclusive,
    open_lock_file,
    unlock_file,
)
from agent4ml.backend.agents.sandbox.docker.executor import DockerExecutor
from agent4ml.backend.agents.sandbox.docker.local_container_backend import (
    LocalContainerBackend,
)
from agent4ml.backend.agents.sandbox.docker.readiness import (
    wait_for_sandbox_ready,
    wait_for_sandbox_ready_async,
)
from agent4ml.backend.agents.sandbox.guards.audit_guard import AuditGuard
from agent4ml.backend.agents.sandbox.sandbox import Sandbox
from agent4ml.backend.agents.sandbox.types import PathMapping, SandboxInfo
from agent4ml.backend.agents.sandbox.utils.sandbox_id import validate_sandbox_id

logger = logging.getLogger(__name__)

_DEFAULT_IDLE_TIMEOUT = 600
_DEFAULT_REPLICAS = 3
_IDLE_CHECK_INTERVAL = 60


def _deterministic_sandbox_id(user_id: str | None, thread_id: str | None) -> str:
    raw = f"{user_id or 'default'}:{thread_id or 'default'}"
    return hashlib.sha256(raw.encode()).hexdigest()[:8]


class DockerSandboxProvider(SandboxProvider):
    """DockerSandboxProvider — 三层缓存 + warm_pool + idle_checker。

    INVARIANT:
    - 三层缓存：in-process → warm_pool → cross-process lock + discover/create
    - release 移入 warm_pool，容器保持运行（不 destroy）
    - idle_checker 60s 间隔，超 idle_timeout 销毁 active + warm
    - 孤儿对账：startup list_running → adopt，跳过空 sandbox_url
    - replicas 软上限：仅 warm_pool 计入驱逐预算，active 不强制停
    - 跨进程锁：3 函数（open/lock/unlock），无 context manager
    - readiness 60s 超时，失败 destroy + raise
    - 信号处理：atexit（Stage 4）+ SIGTERM/SIGINT/SIGHUP（Docker 专用）
    - 构造 Sandbox(DockerRuntime + DockerPathTranslator + DockerPathGuard)
    - path_mappings 不作为 extra_mounts（父 bind mount 覆盖），仅传 config.sandbox.mounts
    - sandbox_root = .agent4ml/sandbox/aio_docker/（类型分层）
    - thread_id 必传
    """

    uses_thread_data_mounts = True

    def __init__(
        self,
        path_mappings: list[PathMapping] | None = None,
        *,
        sandbox_config=None,
        image: str = "all-in-one-sandbox:latest",
        base_port: int = 8080,
        container_prefix: str = "agent4ml-sandbox",
        sandbox_root: str | None = None,
        environment: dict[str, str] | None = None,
        idle_timeout: int = _DEFAULT_IDLE_TIMEOUT,
        replicas: int = _DEFAULT_REPLICAS,
    ) -> None:
        if sandbox_config is not None:
            image = sandbox_config.image or image
            base_port = sandbox_config.port or base_port
            container_prefix = sandbox_config.container_prefix or container_prefix
            environment = dict(sandbox_config.environment) if sandbox_config.environment else environment
            idle_timeout = sandbox_config.idle_timeout or idle_timeout
            replicas = sandbox_config.replicas or replicas

        self._path_mappings = path_mappings or []
        self._idle_timeout = idle_timeout
        self._replicas = replicas
        self._sandbox_root = (
            Path(sandbox_root) if sandbox_root
            else Path.cwd() / ".agent4ml" / "sandbox" / "aio_docker"
        )

        self._extra_mounts: list[PathMapping] = []
        if sandbox_config and sandbox_config.mounts:
            for mount in sandbox_config.mounts:
                self._extra_mounts.append(
                    PathMapping(mount.container_path, mount.host_path, mount.read_only)
                )

        executor = self._make_executor(sandbox_config)

        self._backend = LocalContainerBackend(
            image=image,
            base_port=base_port,
            container_prefix=container_prefix,
            sandbox_root=str(self._sandbox_root),
            environment=environment,
            executor=executor,
        )

        self._lock = threading.Lock()
        self._sandboxes: dict[str, Sandbox] = {}
        self._sandbox_infos: dict[str, SandboxInfo] = {}
        self._thread_sandboxes: dict[tuple[str, str], str] = {}
        self._warm_pool: dict[str, tuple[SandboxInfo, float]] = {}
        self._last_activity: dict[str, float] = {}
        self._thread_locks: dict[tuple[str, str], threading.Lock] = {}
        self._shutdown_called = False
        self._idle_checker_stop = threading.Event()
        self._idle_checker_thread: threading.Thread | None = None

        self._reconcile_orphans()
        if idle_timeout > 0:
            self._start_idle_checker()
        self._register_signal_handlers()

    # ── acquire ──────────────────────────────────────────────────

    def acquire(
        self, thread_id: str | None = None, *, user_id: str | None = None,
    ) -> str:
        if thread_id is None:
            raise ValueError("thread_id is required")
        effective_user_id = user_id or "default"
        thread_lock = self._get_thread_lock(thread_id, effective_user_id)
        with thread_lock:
            return self._acquire_internal(thread_id, user_id=effective_user_id)

    async def acquire_async(
        self, thread_id: str | None = None, *, user_id: str | None = None,
    ) -> str:
        if thread_id is None:
            raise ValueError("thread_id is required")
        effective_user_id = user_id or "default"
        thread_lock = self._get_thread_lock(thread_id, effective_user_id)
        # S5: acquire 必须在 try 块内——cancel 在 acquire 等待期间抛出时，
        # finally 能正确判断是否已持有锁。acquired flag 避免 cancel-before-acquire
        # 路径误 release（未持有的锁 release 会 RuntimeError）。
        acquired = False
        try:
            await asyncio.to_thread(thread_lock.acquire)
            acquired = True
            return await self._acquire_internal_async(thread_id, user_id=effective_user_id)
        finally:
            if acquired:
                thread_lock.release()

    def _acquire_internal(self, thread_id: str, *, user_id: str) -> str:
        cached = self._reuse_in_process(thread_id, user_id)
        if cached is not None:
            return cached
        sandbox_id = _deterministic_sandbox_id(user_id, thread_id)
        reclaimed = self._reclaim_warm(thread_id, sandbox_id, user_id)
        if reclaimed is not None:
            return reclaimed
        return self._discover_or_create(thread_id, sandbox_id, user_id)

    async def _acquire_internal_async(self, thread_id: str, *, user_id: str) -> str:
        cached = await asyncio.to_thread(self._reuse_in_process, thread_id, user_id)
        if cached is not None:
            return cached
        sandbox_id = _deterministic_sandbox_id(user_id, thread_id)
        reclaimed = await asyncio.to_thread(self._reclaim_warm, thread_id, sandbox_id, user_id)
        if reclaimed is not None:
            return reclaimed
        return await self._discover_or_create_async(thread_id, sandbox_id, user_id)

    # ── Layer 1: in-process cache ────────────────────────────────

    def _reuse_in_process(self, thread_id: str, user_id: str) -> str | None:
        key = (user_id, thread_id)
        with self._lock:
            sid = self._thread_sandboxes.get(key)
            if sid is None or sid not in self._sandboxes:
                return None
            info = self._sandbox_infos.get(sid)
        if info is not None:
            alive = self._backend.is_alive(info)
            if alive is False:
                self._drop_unhealthy(sid, "in-process cache health check failed", expected_info=info)
                return None
        with self._lock:
            self._last_activity[sid] = time.time()
            return sid

    # ── Layer 1.5: warm pool ─────────────────────────────────────

    def _reclaim_warm(self, thread_id: str, sandbox_id: str, user_id: str) -> str | None:
        with self._lock:
            if sandbox_id not in self._warm_pool:
                return None
            info, _ = self._warm_pool[sandbox_id]
        alive = self._backend.is_alive(info)
        if alive is False:
            self._drop_unhealthy(sandbox_id, "warm pool health check failed", expected_info=info)
            return None
        sandbox = self._make_sandbox(sandbox_id, info)
        with self._lock:
            self._warm_pool.pop(sandbox_id, None)
            self._sandboxes[sandbox_id] = sandbox
            self._sandbox_infos[sandbox_id] = info
            self._last_activity[sandbox_id] = time.time()
            self._thread_sandboxes[(user_id, thread_id)] = sandbox_id
        logger.info(f"Reclaimed warm-pool sandbox {sandbox_id}")
        return sandbox_id

    # ── get ──────────────────────────────────────────────────────

    def get(self, sandbox_id: str) -> Sandbox | None:
        with self._lock:
            sandbox = self._sandboxes.get(sandbox_id)
            if sandbox is not None:
                self._last_activity[sandbox_id] = time.time()
            return sandbox

    def get_sandbox_info(self, sandbox_id: str) -> SandboxInfo | None:
        """返 SandboxInfo（含 sandbox_url），供 specialist 透传连接容器。"""
        with self._lock:
            return self._sandbox_infos.get(sandbox_id)

    # ── helpers ──────────────────────────────────────────────────

    def _make_executor(self, sandbox_config) -> DockerExecutor:
        from agent4ml.backend.agents.sandbox.docker.executor import (
            LocalDockerExecutor,
            WslDockerExecutor,
        )
        if sandbox_config and getattr(sandbox_config, "executor", "local") == "wsl":
            distro = getattr(sandbox_config, "wsl_distro", None) or "Ubuntu"
            user = getattr(sandbox_config, "wsl_user", None)
            return WslDockerExecutor(distro=distro, user=user)
        return LocalDockerExecutor()

    def _make_sandbox(self, sandbox_id: str, info: SandboxInfo) -> Sandbox:
        from agent4ml.backend.agents.sandbox.runtimes.docker_runtime import DockerRuntime
        from agent4ml.backend.agents.sandbox.translators.docker_path_translator import (
            DockerPathTranslator,
        )
        from agent4ml.backend.agents.sandbox.guards.docker_path_guard import (
            DockerPathGuard,
        )
        runtime = DockerRuntime(info.sandbox_url)
        translator = DockerPathTranslator(self._sandbox_root, sandbox_id)
        guard = AuditGuard(DockerPathGuard())
        return Sandbox(sandbox_id, runtime, translator, guard)

    def _register(
        self, thread_id: str, sandbox_id: str, info: SandboxInfo, user_id: str,
    ) -> str:
        sandbox = self._make_sandbox(sandbox_id, info)
        with self._lock:
            self._sandboxes[sandbox_id] = sandbox
            self._sandbox_infos[sandbox_id] = info
            self._last_activity[sandbox_id] = time.time()
            self._thread_sandboxes[(user_id, thread_id)] = sandbox_id
        return sandbox_id

    def _get_thread_lock(self, thread_id: str, user_id: str) -> threading.Lock:
        key = (user_id, thread_id)
        with self._lock:
            if key not in self._thread_locks:
                self._thread_locks[key] = threading.Lock()
            return self._thread_locks[key]

    def _drop_unhealthy(
        self, sandbox_id: str, reason: str, expected_info: SandboxInfo | None = None,
    ) -> None:
        """移除不健康 sandbox + 销毁容器。

        S7: destroy 前二次 discover 校验——若别线程已用同 ID 建新容器
        （new_info != expected_info），跳过 destroy 避免误杀。
        """
        with self._lock:
            sandbox = self._sandboxes.pop(sandbox_id, None)
            info = self._sandbox_infos.pop(sandbox_id, None)
            keys = [k for k, sid in self._thread_sandboxes.items() if sid == sandbox_id]
            for k in keys:
                del self._thread_sandboxes[k]
            self._last_activity.pop(sandbox_id, None)
            warm_item = self._warm_pool.pop(sandbox_id, None)
            if info is None and warm_item is not None:
                info = warm_item[0]
        if sandbox:
            try:
                sandbox.close()
            except Exception:
                pass
        if info:
            # S7: destroy 前二次探活——别线程可能已用同 ID 建新容器
            try:
                current_info = self._backend.discover(sandbox_id)
            except Exception:
                current_info = None  # discover 失败→按"容器状态未知"处理，继续 destroy
            if current_info is not None and expected_info is not None and current_info != expected_info:
                logger.warning(
                    f"Skip destroy for {sandbox_id}: container changed "
                    f"(expected {expected_info.container_name}, "
                    f"current {current_info.container_name})"
                )
            else:
                try:
                    self._backend.destroy(info)
                except Exception:
                    pass
        logger.warning(f"Dropped unhealthy sandbox {sandbox_id}: {reason}")

    # ── Layer 2: cross-process lock + discover/create ────────────

    def _discover_or_create(self, thread_id: str, sandbox_id: str, user_id: str) -> str:
        validate_sandbox_id(sandbox_id)
        lock_path = self._sandbox_root / f"{sandbox_id}.lock"
        lock_file = open_lock_file(lock_path)
        try:
            lock_file_exclusive(lock_file)
            cached = self._reuse_in_process(thread_id, user_id)
            if cached is not None:
                return cached
            discovered = self._backend.discover(sandbox_id)
            if discovered is not None:
                return self._register(thread_id, sandbox_id, discovered, user_id)
            return self._create_sandbox(thread_id, sandbox_id, user_id)
        finally:
            unlock_file(lock_file)
            lock_file.close()

    async def _discover_or_create_async(
        self, thread_id: str, sandbox_id: str, user_id: str,
    ) -> str:
        validate_sandbox_id(sandbox_id)
        lock_path = self._sandbox_root / f"{sandbox_id}.lock"
        lock_file = await asyncio.to_thread(open_lock_file, lock_path)
        try:
            await asyncio.to_thread(lock_file_exclusive, lock_file)
            cached = await asyncio.to_thread(self._reuse_in_process, thread_id, user_id)
            if cached is not None:
                return cached
            discovered = await asyncio.to_thread(self._backend.discover, sandbox_id)
            if discovered is not None:
                return await asyncio.to_thread(
                    self._register, thread_id, sandbox_id, discovered, user_id,
                )
            return await self._create_sandbox_async(thread_id, sandbox_id, user_id)
        finally:
            await asyncio.to_thread(unlock_file, lock_file)
            await asyncio.to_thread(lock_file.close)

    def _create_sandbox(self, thread_id: str, sandbox_id: str, user_id: str) -> str:
        self._enforce_replicas()
        info = self._backend.create(
            thread_id, sandbox_id, extra_mounts=self._extra_mounts or None,
        )
        if not wait_for_sandbox_ready(info.sandbox_url, timeout=60):
            self._backend.destroy(info)
            raise RuntimeError(
                f"Sandbox {sandbox_id} failed readiness at {info.sandbox_url}"
            )
        logger.info(f"Created sandbox {sandbox_id} at {info.sandbox_url}")
        return self._register(thread_id, sandbox_id, info, user_id)

    async def _create_sandbox_async(
        self, thread_id: str, sandbox_id: str, user_id: str,
    ) -> str:
        await asyncio.to_thread(self._enforce_replicas)
        info = await asyncio.to_thread(
            self._backend.create, thread_id, sandbox_id,
            extra_mounts=self._extra_mounts or None,
        )
        if not await wait_for_sandbox_ready_async(info.sandbox_url, timeout=60):
            await asyncio.to_thread(self._backend.destroy, info)
            raise RuntimeError(
                f"Sandbox {sandbox_id} failed readiness at {info.sandbox_url}"
            )
        logger.info(f"Created sandbox {sandbox_id} at {info.sandbox_url}")
        return await asyncio.to_thread(self._register, thread_id, sandbox_id, info, user_id)

    def _enforce_replicas(self) -> None:
        with self._lock:
            total = len(self._sandboxes) + len(self._warm_pool)
        if total >= self._replicas:
            self._evict_oldest_warm()

    def _evict_oldest_warm(self) -> None:
        with self._lock:
            if not self._warm_pool:
                return
            oldest = min(self._warm_pool, key=lambda sid: self._warm_pool[sid][1])
            info, _ = self._warm_pool.pop(oldest)
        try:
            self._backend.destroy(info)
            logger.info(f"Evicted warm-pool sandbox {oldest}")
        except Exception as exc:
            logger.error(f"Failed to evict warm {oldest}: {exc}")

    # ── release / shutdown / idle / orphan / signal ──────────────

    def release(self, sandbox_id: str) -> None:
        with self._lock:
            sandbox = self._sandboxes.pop(sandbox_id, None)
            info = self._sandbox_infos.pop(sandbox_id, None)
            keys = [k for k, sid in self._thread_sandboxes.items() if sid == sandbox_id]
            for k in keys:
                del self._thread_sandboxes[k]
            self._last_activity.pop(sandbox_id, None)
            if info and sandbox_id not in self._warm_pool:
                self._warm_pool[sandbox_id] = (info, time.time())
        if sandbox is not None:
            try:
                sandbox.close()
            except Exception as exc:
                logger.warning(f"Error closing sandbox {sandbox_id} during release: {exc}")
        logger.info(f"Released sandbox {sandbox_id} to warm pool")

    def shutdown(self) -> None:
        with self._lock:
            if self._shutdown_called:
                return
            self._shutdown_called = True
            active_ids = list(self._sandboxes.keys())
            warm_items = list(self._warm_pool.items())
            self._warm_pool.clear()
        self._idle_checker_stop.set()
        if self._idle_checker_thread and self._idle_checker_thread.is_alive():
            self._idle_checker_thread.join(timeout=5)
        for sid in active_ids:
            try:
                self._destroy_active(sid)
            except Exception as exc:
                logger.error(f"Failed to destroy {sid} during shutdown: {exc}")
        for sid, (info, _) in warm_items:
            try:
                self._backend.destroy(info)
            except Exception as exc:
                logger.error(f"Failed to destroy warm {sid}: {exc}")

    def _destroy_active(self, sandbox_id: str) -> None:
        with self._lock:
            sandbox = self._sandboxes.pop(sandbox_id, None)
            info = self._sandbox_infos.pop(sandbox_id, None)
            keys = [k for k, sid in self._thread_sandboxes.items() if sid == sandbox_id]
            for k in keys:
                del self._thread_sandboxes[k]
            self._last_activity.pop(sandbox_id, None)
        if sandbox:
            try:
                sandbox.close()
            except Exception:
                pass
        if info:
            # S7: destroy 前二次探活——避免误杀别线程新建的同 ID 容器
            try:
                current_info = self._backend.discover(sandbox_id)
            except Exception:
                current_info = None
            if current_info is not None and current_info != info:
                logger.warning(
                    f"Skip destroy for {sandbox_id}: container changed "
                    f"(expected {info.container_name}, "
                    f"current {current_info.container_name})"
                )
            else:
                try:
                    self._backend.destroy(info)
                except Exception:
                    pass

    def _start_idle_checker(self) -> None:
        self._idle_checker_thread = threading.Thread(
            target=self._idle_checker_loop, name="sandbox-idle-checker", daemon=True,
        )
        self._idle_checker_thread.start()

    def _idle_checker_loop(self) -> None:
        while not self._idle_checker_stop.wait(timeout=_IDLE_CHECK_INTERVAL):
            try:
                self._cleanup_idle()
            except Exception as exc:
                logger.error(f"Idle checker error: {exc}")

    def _cleanup_idle(self) -> None:
        now = time.time()
        active_to_destroy = []
        warm_to_destroy = []
        with self._lock:
            for sid, ts in self._last_activity.items():
                if now - ts > self._idle_timeout:
                    active_to_destroy.append(sid)
            for sid, (info, release_ts) in list(self._warm_pool.items()):
                if now - release_ts > self._idle_timeout:
                    warm_to_destroy.append((sid, info))
                    del self._warm_pool[sid]
        for sid in active_to_destroy:
            with self._lock:
                ts = self._last_activity.get(sid)
                if ts is None or (time.time() - ts) < self._idle_timeout:
                    continue
            self._destroy_active(sid)
        for sid, info in warm_to_destroy:
            try:
                self._backend.destroy(info)
            except Exception as exc:
                logger.error(f"Failed to destroy idle warm {sid}: {exc}")

    def _reconcile_orphans(self) -> None:
        try:
            running = self._backend.list_running()
        except Exception as exc:
            logger.warning(f"Orphan reconciliation failed: {exc}")
            return
        now = time.time()
        adopted = 0
        for info in running:
            if not info.sandbox_url:
                continue
            with self._lock:
                if info.sandbox_id in self._sandboxes or info.sandbox_id in self._warm_pool:
                    continue
                self._warm_pool[info.sandbox_id] = (info, now)
            adopted += 1
        if adopted:
            logger.info(f"Orphan reconciliation: adopted {adopted} container(s)")

    def _register_signal_handlers(self) -> None:
        import signal as _signal
        original_sigterm = _signal.getsignal(_signal.SIGTERM)
        original_sigint = _signal.getsignal(_signal.SIGINT)

        def handler(signum, frame):
            self.shutdown()
            original = original_sigterm if signum == _signal.SIGTERM else original_sigint
            if callable(original):
                original(signum, frame)

        try:
            _signal.signal(_signal.SIGTERM, handler)
            _signal.signal(_signal.SIGINT, handler)
            if hasattr(_signal, "SIGHUP"):
                original_sighup = _signal.getsignal(_signal.SIGHUP)
                def sighup_handler(signum, frame):
                    self.shutdown()
                    if callable(original_sighup):
                        original_sighup(signum, frame)
                _signal.signal(_signal.SIGHUP, sighup_handler)
        except (ValueError, OSError):
            pass
