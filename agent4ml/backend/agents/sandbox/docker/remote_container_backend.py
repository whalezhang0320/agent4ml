from __future__ import annotations

from agent4ml.backend.agents.sandbox.contracts import SandboxBackend
from agent4ml.backend.agents.sandbox.types import PathMapping, SandboxInfo


class RemoteContainerBackend(SandboxBackend):
    """RemoteContainerBackend — K8s Pod + NodePort 空壳。

    预留接口，不实现（Grill #9：K8s 保留接口空壳）。
    create/destroy/discover/list_running 抛 NotImplementedError。
    is_alive 返 None（符合 ABC 契约 bool|None，允许上层 graceful 降级，
    如 fallback 到 LocalContainerBackend）。

    未来接 K8s 时实现全部方法。
    """

    def create(
        self,
        thread_id: str,
        sandbox_id: str,
        extra_mounts: list[PathMapping] | None = None,
        *,
        user_id: str | None = None,
    ) -> SandboxInfo:
        raise NotImplementedError("K8s provisioner not implemented (Stage 5 空壳)")

    def destroy(self, info: SandboxInfo) -> None:
        raise NotImplementedError("K8s provisioner not implemented (Stage 5 空壳)")

    def is_alive(self, info: SandboxInfo) -> bool | None:
        return None

    def discover(self, sandbox_id: str) -> SandboxInfo | None:
        raise NotImplementedError("K8s provisioner not implemented (Stage 5 空壳)")

    def list_running(self) -> list[SandboxInfo]:
        raise NotImplementedError("K8s provisioner not implemented (Stage 5 空壳)")
