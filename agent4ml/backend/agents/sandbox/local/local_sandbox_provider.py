from __future__ import annotations

import hashlib
import threading
from collections import OrderedDict

from agent4ml.backend.agents.sandbox.contracts import SandboxProvider
from agent4ml.backend.agents.sandbox.guards.audit_guard import AuditGuard
from agent4ml.backend.agents.sandbox.guards.local_security_guard import (
    LocalSecurityGuard,
)
from agent4ml.backend.agents.sandbox.runtimes.local_runtime import LocalRuntime
from agent4ml.backend.agents.sandbox.sandbox import Sandbox
from agent4ml.backend.agents.sandbox.translators.local_path_translator import (
    LocalPathTranslator,
)
from agent4ml.backend.agents.sandbox.types import PathMapping

_DEFAULT_LRU_SIZE = 256


def _deterministic_sandbox_id(user_id: str | None, thread_id: str | None) -> str:
    """确定性 sandbox_id = sha256(user:thread)[:8]。跨进程可推导。"""
    raw = f"{user_id or 'default'}:{thread_id or 'default'}"
    return hashlib.sha256(raw.encode()).hexdigest()[:8]


class LocalSandboxProvider(SandboxProvider):
    """LocalSandboxProvider — LRU 缓存 + 确定性 ID。

    INVARIANT:
    - acquire(thread_id) → per-thread sandbox，确定性 ID = sha256(user:thread)[:8]
    - thread_id 必传（Grill #6：去掉 legacy 单例，None 抛 ValueError）
    - LRU 缓存默认 256 条，超出按 LRU 驱逐（驱逐前调 sandbox.close()）
    - release no-op（保留缓存，下次复用）
    - get 纯内存查找，事件循环安全
    - 构造 Sandbox 时组合 LocalRuntime + LocalPathTranslator + LocalSecurityGuard
    - _sandboxes OrderedDict 全部读写路径用 _lock 保护（严重1 修复）
    - LRU 驱逐在锁内 pop 元组，锁外调 sandbox.close()（避免持锁阻塞 I/O）
    """

    uses_thread_data_mounts = True
    needs_upload_permission_adjustment = False

    def __init__(
        self,
        path_mappings: list[PathMapping] | None = None,
        lru_size: int = _DEFAULT_LRU_SIZE,
        sandbox_config=None,
    ) -> None:
        self._path_mappings = path_mappings or []
        self._lru_size = lru_size
        self._sandboxes: OrderedDict[str, Sandbox] = OrderedDict()
        self._lock = threading.Lock()
        self._allow_host_bash = True
        if sandbox_config is not None:
            self._allow_host_bash = getattr(sandbox_config, "allow_host_bash", True)

    def acquire(
        self, thread_id: str | None = None, *, user_id: str | None = None
    ) -> str:
        if thread_id is None:
            raise ValueError(
                "thread_id is required (legacy singletons removed, Grill #6)"
            )
        sandbox_id = _deterministic_sandbox_id(user_id, thread_id)

        evicted: Sandbox | None = None
        with self._lock:
            if sandbox_id in self._sandboxes:
                self._sandboxes.move_to_end(sandbox_id)
                return sandbox_id

            runtime = LocalRuntime(allow_host_bash=self._allow_host_bash)
            translator = LocalPathTranslator(self._path_mappings)
            guard = AuditGuard(LocalSecurityGuard(self._path_mappings))
            sandbox = Sandbox(sandbox_id, runtime, translator, guard)
            self._sandboxes[sandbox_id] = sandbox

            if len(self._sandboxes) > self._lru_size:
                _evicted_id, evicted = self._sandboxes.popitem(last=False)

        # 锁外 close 避免持锁阻塞 I/O（被驱逐 sandbox 可能正被别线程用，
        # 但 close 是 LocalRuntime 的 no-op，不会炸）
        if evicted is not None:
            evicted.close()
        return sandbox_id

    def get(self, sandbox_id: str) -> Sandbox | None:
        with self._lock:
            return self._sandboxes.get(sandbox_id)

    def release(self, sandbox_id: str) -> None:
        pass

    def reset(self) -> None:
        with self._lock:
            self._sandboxes.clear()

    def shutdown(self) -> None:
        with self._lock:
            sandboxes = list(self._sandboxes.values())
            self._sandboxes.clear()
        # 锁外逐个 close，避免持锁时阻塞
        for sandbox in sandboxes:
            sandbox.close()
