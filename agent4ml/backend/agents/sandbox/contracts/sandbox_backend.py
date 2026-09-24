from __future__ import annotations

from abc import ABC, abstractmethod

from agent4ml.backend.agents.sandbox.types import PathMapping, SandboxInfo


class SandboxBackend(ABC):
    """沙箱运行平台契约（基础设施操作层）。

    管理沙箱实例在基础设施上的 CRUD（create / destroy / is_alive / discover）。
    不限定容器类型——Docker / K8s / Firecracker microVM / WASM 都是"沙箱运行平台"。

    与 SandboxProvider 区分：
    - SandboxProvider：进程内对象生命周期（acquire / get / release / 缓存 / warm_pool）
    - SandboxBackend：基础设施操作（create / destroy / is_alive / discover / 跨进程）

    子类：LocalPlatform（本地宿主机）/ DockerPlatform（Docker/Apple）/ K8sPlatform（预留空壳）。

    INVARIANT:
    - create 幂等：同 sandbox_id 返回已存在的
    - is_alive 轻量（docker inspect），不调 HTTP
    - is_alive 返 None 表未知（不因健康检查失败误杀容器）
    - discover 用于跨进程恢复：按确定性 ID 查已有实例
    """

    @abstractmethod
    def create(
        self,
        thread_id: str,
        sandbox_id: str,
        extra_mounts: list[PathMapping] | None = None,
        *,
        user_id: str | None = None,
    ) -> SandboxInfo:
        """创建沙箱实例。幂等：同 sandbox_id 返回已存在的。"""
        ...

    @abstractmethod
    def destroy(self, info: SandboxInfo) -> None:
        """销毁沙箱实例。"""
        ...

    @abstractmethod
    def is_alive(self, info: SandboxInfo) -> bool | None:
        """查实例存活。轻量（docker inspect），不调 HTTP。

        返 True=running / False=stopped / None=未知（不误杀）。
        """
        ...

    @abstractmethod
    def discover(self, sandbox_id: str) -> SandboxInfo | None:
        """按确定性 ID 查已有实例。跨进程恢复用。"""
        ...

    def list_running(self) -> list[SandboxInfo]:
        """列所有运行中实例。孤儿对账用。默认空。"""
        return []
