"""ClaudeMarketplaceSource — 拉 Claude Marketplace registry 安装 skill。

设计（design_docs/46 §2.4）:
- 拉 Claude Marketplace registry（Anthropic 官方 skill 生态）
- 转 SkillMeta
- identifier 格式：claude-marketplace:<skill-name>
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from agent4ml.backend.agents.skill.hub.source import SkillMeta


# Claude Marketplace registry URL（Anthropic 官方，MVP 用 placeholder）
_MARKETPLACE_REGISTRY_URL = "https://registry.claude.com/skills/index.json"


class ClaudeMarketplaceSource:
    """从 Claude Marketplace 发现 skill 的 source。

    拉 Claude Marketplace registry，转 SkillMeta。
    registry 不可达时降级为返空（不抛异常）。
    """

    name = "claude-marketplace"

    def __init__(self, registry_url: str | None = None) -> None:
        self._registry_url = registry_url or _MARKETPLACE_REGISTRY_URL

    def search(self, query: str, limit: int = 10) -> list[SkillMeta]:
        """搜索 Claude Marketplace，返匹配 SkillMeta 列表。

        registry 不可达时降级为返空（不抛异常）。
        """
        try:
            registry = self._fetch_registry()
            if not registry:
                return []
        except Exception:
            return []

        query_lower = query.lower()
        results: list[SkillMeta] = []
        for skill in registry:
            name = skill.get("name", "")
            desc = skill.get("description", "")
            if query_lower in name.lower() or query_lower in desc.lower():
                results.append(SkillMeta(
                    name=name,
                    description=desc,
                    category=skill.get("category", "claude"),
                    source="claude-marketplace",
                    identifier=f"claude-marketplace:{name}",
                    preview_url=skill.get("preview_url"),
                    is_installed=False,
                ))
                if len(results) >= limit:
                    return results
        return results

    def fetch(self, identifier: str, dest_dir: Path) -> Path:
        """下载 Claude Marketplace skill（MVP：返 dest_dir，需 HTTP 下载实现）。"""
        return dest_dir

    def preview(self, identifier: str) -> str | None:
        """预览 Claude Marketplace SKILL.md（MVP：返 None）。"""
        return None

    def _fetch_registry(self) -> list[dict[str, Any]] | None:
        """HTTP GET Claude Marketplace registry，返 skill 列表。

        registry 不可达时返 None（不抛异常）。
        """
        try:
            import httpx

            response = httpx.get(self._registry_url, timeout=10, follow_redirects=True)
            if response.status_code != 200:
                return None
            data = response.json()
            if isinstance(data, list):
                return data
            if isinstance(data, dict) and "skills" in data:
                return data["skills"]
            return None
        except Exception:
            return None
