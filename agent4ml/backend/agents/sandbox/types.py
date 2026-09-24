from __future__ import annotations

from dataclasses import dataclass, field
from typing import NamedTuple

from agent4ml.backend.agents.journal.events import utc_now_iso


@dataclass(frozen=True)
class PathMapping:
    """虚拟路径到宿主路径映射。LocalSandbox 专属。

    frozen = 不可变，线程安全。按 container_path 长度降序匹配（最长前缀优先）。
    """

    container_path: str
    local_path: str
    read_only: bool = False


class ResolvedPath(NamedTuple):
    """_resolve_path_with_mapping 返回值。同时返回路径和匹配的映射。"""

    path: str
    mapping: PathMapping | None


@dataclass
class GrepMatch:
    """grep 工具的单条匹配结果。跨实现统一。"""

    path: str
    line_number: int
    line: str


@dataclass
class SandboxInfo:
    """跨进程可恢复的沙箱元数据。

    container_name / container_id 仅 LocalContainerBackend 有。
    K8s 模式（RemoteContainerBackend）只有 sandbox_url。
    created_at 用 Agent4ML 统一 ISO 格式（utc_now_iso），与 journal 一致。
    """

    sandbox_id: str
    sandbox_url: str
    container_name: str | None = None
    container_id: str | None = None
    created_at: str = field(default_factory=utc_now_iso)

    def to_dict(self) -> dict:
        """序列化（持久化 / 日志）。"""
        return {
            "sandbox_id": self.sandbox_id,
            "sandbox_url": self.sandbox_url,
            "container_name": self.container_name,
            "container_id": self.container_id,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> SandboxInfo:
        """反序列化。兼容旧字段名 base_url → sandbox_url。"""
        sandbox_url = data.get("sandbox_url") or data.get("base_url", "")
        created_at = data.get("created_at")
        if not created_at:
            created_at = utc_now_iso()
        return cls(
            sandbox_id=data["sandbox_id"],
            sandbox_url=sandbox_url,
            container_name=data.get("container_name"),
            container_id=data.get("container_id"),
            created_at=created_at,
        )
