"""WellKnownSource — HTTP GET /.well-known/skills/index.json 安装 skill。

设计（design_docs/46 §2.4）:
- 任何网站可暴露 skill 索引（/.well-known/skills/index.json）
- HTTP GET index.json，关键词匹配 name/description
- endpoint 不可达时降级为返空（不抛异常）
- identifier 格式：well-known:https://example.com 或 well-known:example.com
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agent4ml.backend.agents.skill.hub.source import SkillMeta


class WellKnownSource:
    """从 /.well-known/skills/index.json 发现 skill 的 source。

    任何网站可暴露 skill 索引。HTTP GET index.json，关键词匹配。
    endpoint 不可达时降级为返空（不抛异常）。
    """

    name = "well-known"

    def __init__(self, endpoints: list[str] | None = None) -> None:
        """endpoints: well-known URL 列表（如 ["https://example.com"]）。

        None 时用默认 endpoints（空列表，用户配置补充）。
        """
        self._endpoints = endpoints or []

    def search(self, query: str, limit: int = 10) -> list[SkillMeta]:
        """搜索 well-known endpoints，返匹配 SkillMeta 列表。

        endpoint 不可达时降级为返空（不抛异常）。
        """
        results: list[SkillMeta] = []
        query_lower = query.lower()

        for endpoint in self._endpoints:
            try:
                index = self._fetch_index(endpoint)
                if not index:
                    continue
                for skill in index:
                    name = skill.get("name", "")
                    desc = skill.get("description", "")
                    if query_lower in name.lower() or query_lower in desc.lower():
                        results.append(SkillMeta(
                            name=name,
                            description=desc,
                            category=skill.get("category", "unknown"),
                            source="well-known",
                            identifier=f"well-known:{endpoint}@{name}",
                            preview_url=skill.get("preview_url"),
                            is_installed=False,
                        ))
                        if len(results) >= limit:
                            return results
            except Exception:
                # endpoint 不可达，降级为跳过
                continue

        return results

    def fetch(self, identifier: str, dest_dir: Path) -> Path:
        """下载 skill 到 dest_dir（MVP：返 dest_dir，需 HTTP 下载实现）。

        identifier 格式：well-known:<endpoint>@<skill-name>
        """
        # MVP：不实现 HTTP 下载（需 httpx GET SKILL.md content）
        # 进阶：HTTP GET endpoint + skill_name → 下载 SKILL.md + 相关文件
        return dest_dir

    def preview(self, identifier: str) -> str | None:
        """预览 well-known SKILL.md（MVP：返 None，需 HTTP GET）。"""
        return None

    def _fetch_index(self, endpoint: str) -> list[dict[str, Any]] | None:
        """HTTP GET endpoint/.well-known/skills/index.json，返 skill 列表。

        endpoint 不可达时返 None（不抛异常）。
        """
        url = f"{endpoint.rstrip('/')}/.well-known/skills/index.json"
        try:
            import httpx

            response = httpx.get(url, timeout=10, follow_redirects=True)
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
