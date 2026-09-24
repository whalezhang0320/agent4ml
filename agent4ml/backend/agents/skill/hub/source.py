"""SkillSource Protocol + SkillMeta — skill 发现抽象。

设计（design_docs/46 §2.3）:
- SkillSource Protocol：每种 source（builtin/github/well-known/claude-marketplace）写一个 adapter
- 新增 source = 新增 adapter + 注册，不动核心 search/install 逻辑
- SkillMeta：source 返回的 skill 元数据（name/description/category/source/identifier/...）
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class SkillMeta:
    """source 返回的 skill 元数据。

    与 SkillRecord 区分：SkillMeta 是 source 搜索结果的轻量元数据，
    不含 metrics/version lineage（这些在 install 后由 SkillStore 管理）。
    """

    name: str
    description: str
    category: str
    source: str                              # "builtin" / "github" / "well-known" / "claude-marketplace"
    identifier: str                          # 安装时传给 installer 的唯一标识
    install_path: str | None = None          # 已安装时的路径（未安装 None）
    preview_url: str | None = None           # SKILL.md 预览 URL（远程 source）
    is_installed: bool = False


class SkillSource(Protocol):
    """Skill registry source adapter。

    每种 source（builtin / github / well-known / claude-marketplace）写一个 adapter。
    新增 source = 新增 adapter + 注册，不动核心 search/install 逻辑。

    实现示例：BuiltinSource / GitHubSource / WellKnownSource / ClaudeMarketplaceSource。
    """

    name: str

    def search(self, query: str, limit: int = 10) -> list[SkillMeta]:
        """搜索 skill，返匹配 SkillMeta 列表。

        query: 关键词（匹配 name 或 description）
        limit: 最多返多少条
        """
        ...

    def fetch(self, identifier: str, dest_dir: Path) -> Path:
        """下载/克隆 skill 到 dest_dir，返 skill 目录路径。

        identifier: source 特定的唯一标识（如 github:owner/repo@skill-name）
        dest_dir: 目标目录（installer 创建）
        """
        ...

    def preview(self, identifier: str) -> str | None:
        """预览 SKILL.md 内容（不安装）。

        返 SKILL.md 文本内容，或 None（source 不支持预览）。
        """
        ...
