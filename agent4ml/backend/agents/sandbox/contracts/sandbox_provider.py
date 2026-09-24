from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent4ml.backend.agents.sandbox.sandbox import Sandbox
    from agent4ml.backend.agents.sandbox.types import SandboxInfo


class SandboxProvider(ABC):
    """沙箱生命周期管理契约。

    方案 C 生命周期层。Local / Docker 各写子类。

    INVARIANT:
    - get(sandbox_id) 是纯内存查找，事件循环安全
    - acquire 可能阻塞（Docker 操作），async 路径用 acquire_async
    - release 不一定销毁（LocalSandboxProvider no-op，DockerSandboxProvider 移入 warm_pool）
    - reset 清缓存不 shutdown；shutdown 销毁所有沙箱
    - get_sandbox_info 返 SandboxInfo（含 sandbox_url）供 specialist 透传；Local 返 None
    """

    uses_thread_data_mounts: bool = False
    needs_upload_permission_adjustment: bool = True

    @abstractmethod
    def acquire(
        self, thread_id: str | None = None, *, user_id: str | None = None
    ) -> str:
        """获取（或复用）沙箱，返 sandbox_id。可能阻塞。"""
        ...

    async def acquire_async(
        self, thread_id: str | None = None, *, user_id: str | None = None
    ) -> str:
        """异步版本。默认 asyncio.to_thread 包同步实现。

        子类可重写以更精细控制（如 DockerSandboxProvider 重写避免阻塞事件循环）。
        """
        return await asyncio.to_thread(self.acquire, thread_id, user_id=user_id)

    @abstractmethod
    def get(self, sandbox_id: str) -> Sandbox | None:
        """纯内存查找。返 None 表示未找到。事件循环安全。"""
        ...

    @abstractmethod
    def release(self, sandbox_id: str) -> None:
        """释放沙箱（不一定销毁）。"""
        ...

    def get_sandbox_info(self, sandbox_id: str) -> SandboxInfo | None:
        """返沙箱元信息（含 sandbox_url），供 specialist 透传连接 Docker 容器。

        DockerSandboxProvider 返 SandboxInfo；LocalSandboxProvider 返 None（无 URL 概念）。
        默认返 None，Docker 子类重写。
        """
        return None

    def reset(self) -> None:
        """清缓存，不 shutdown。默认空实现。"""
        pass

    def shutdown(self) -> None:
        """销毁所有沙箱 + 清缓存。默认空实现。"""
        pass
