"""BuiltinSource — 扫 builtin_skills/ 树（零网络）。

设计（design_docs/46 §2.4）:
- 调既有 SkillManager.search_builtin_skills，转 SkillMeta
- 已激活的 skill 标 is_installed=True
- source="builtin"
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from agent4ml.backend.agents.skill.hub.source import SkillMeta


class BuiltinSource:
    """扫 builtin_skills/ 树的 source（零网络）。

    调既有 SkillManager.search_builtin_skills，转 SkillMeta。
    """

    name = "builtin"

    def __init__(self, skill_manager: Any | None = None) -> None:
        self._skill_manager = skill_manager

    def search(self, query: str, limit: int = 10) -> list[SkillMeta]:
        """搜 builtin_skills/ 树，返 SkillMeta 列表。"""
        mgr = self._skill_manager
        if mgr is None:
            # 延迟加载，避免循环 import
            from agent4ml.backend.agents.skill import build_skill_manager

            mgr = build_skill_manager()
        if mgr is None:
            return []

        builtin_results = mgr.search_builtin_skills(query)
        metas: list[SkillMeta] = []
        for r in builtin_results[:limit]:
            metas.append(SkillMeta(
                name=r.get("name", ""),
                description=r.get("description", ""),
                category=r.get("category", "core"),
                source="builtin",
                identifier=f"builtin:{r.get('name', '')}",
                install_path=r.get("path"),
                is_installed=r.get("is_active", False),
            ))
        return metas

    def fetch(self, identifier: str, dest_dir: Path) -> Path:
        """builtin skill 已在本地，直接返 path。

        identifier 格式：builtin:<name>
        """
        # builtin skill 不需要下载，install_path 已在 search 时返回
        # caller (Installer) 应直接用 install_path 而非 fetch
        # 这里返 identifier 对应的 builtin path（从 skill_manager 查）
        mgr = self._skill_manager
        if mgr is None:
            from agent4ml.backend.agents.skill import build_skill_manager

            mgr = build_skill_manager()
        if mgr is None:
            return dest_dir

        # 解析 identifier: builtin:<name>
        name = identifier.split(":", 1)[1] if ":" in identifier else identifier
        # 搜 builtin 找 path
        results = mgr.search_builtin_skills(name)
        for r in results:
            if r.get("name") == name:
                return Path(r.get("path", "."))
        return dest_dir

    def preview(self, identifier: str) -> str | None:
        """预览 builtin SKILL.md 内容。"""
        path = self.fetch(identifier, Path("."))
        if path.exists():
            return path.read_text(encoding="utf-8")
        return None
