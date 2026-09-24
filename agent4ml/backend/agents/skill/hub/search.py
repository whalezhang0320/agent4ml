"""unified_search — 跨 source 聚合搜索。

设计（design_docs/46 §2.6）:
- 跨 source 聚合搜索（builtin + hub sources 并行）
- 去重（按 name）+ 排序（is_installed 优先）+ limit 截断
- hub 不可用时降级为只搜 builtin
"""
from __future__ import annotations

import logging
from typing import Any

from agent4ml.backend.agents.skill.hub.source import SkillMeta, SkillSource

logger = logging.getLogger(__name__)


def unified_search(
    query: str,
    sources: list[SkillSource] | None = None,
    limit: int = 10,
) -> list[SkillMeta]:
    """跨 source 聚合搜索。

    去重（按 name）+ 排序（is_installed 优先）+ limit 截断。
    hub 不可用时降级为只搜 builtin。
    """
    if not sources:
        # 降级：只搜 builtin
        try:
            from agent4ml.backend.agents.skill.hub.sources.builtin_source import (
                BuiltinSource,
            )

            sources = [BuiltinSource()]
        except Exception as e:
            logger.warning("unified_search: builtin source unavailable: %s", e)
            return []

    all_results: list[SkillMeta] = []
    seen_names: set[str] = set()

    for source in sources:
        try:
            results = source.search(query, limit=limit)
            for meta in results:
                if meta.name and meta.name not in seen_names:
                    seen_names.add(meta.name)
                    all_results.append(meta)
        except Exception as e:
            logger.warning(
                "unified_search: source %s failed: %s",
                getattr(source, "name", "unknown"), e,
            )
            continue

    # 排序：is_installed 优先
    all_results.sort(key=lambda m: (not m.is_installed, m.name))

    # limit 截断
    return all_results[:limit]


def unified_search_as_dicts(
    query: str,
    sources: list[SkillSource] | None = None,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """unified_search 的 dict 版本（供 JSON 返回）。

    返 [{"name", "description", "category", "source", "identifier", "is_installed", "install_path", "preview_url"}]
    """
    metas = unified_search(query, sources, limit)
    return [
        {
            "name": m.name,
            "description": m.description,
            "category": m.category,
            "source": m.source,
            "identifier": m.identifier,
            "is_installed": m.is_installed,
            "install_path": m.install_path,
            "preview_url": m.preview_url,
        }
        for m in metas
    ]
