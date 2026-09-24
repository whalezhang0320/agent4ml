"""Prompt 管理系统 — 统一加载/渲染/切换 prompt。

三层架构：
- PromptManager（本模块）：low-level，唯一碰文件的层
- 模块 accessor（leader/prompts.py 等）：mid-level，知道 category/name + vars
- 外部调用方：high-level，只调 accessor

prompt 持久化为 .md 文件，${variable} regex 渲染，user/ 覆盖 system/。
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any


class PromptManager:
    """统一 prompt 加载/渲染/切换管理。"""

    def __init__(self, base_dir: Path) -> None:
        self._base = base_dir
        self._cache: dict[str, str] = {}

    def load(self, category: str, name: str, **vars: Any) -> str:
        """加载 prompt + ${var} 渲染。user/ 优先 system/。"""
        template, _source = self._read(category, name)
        return self._render(template, vars)

    def load_raw(self, category: str, name: str) -> tuple[str, str]:
        """加载原始文本（不渲染）。返回 (text, source)。source="user"|"system"。"""
        return self._read(category, name)

    def list_prompts(self, category: str | None = None) -> list[str]:
        """列出可用 prompt（category/name 格式）。"""
        result: set[str] = set()
        for layer in ("system", "user"):
            base = self._base / layer
            if not base.exists():
                continue
            for cat_dir in base.iterdir():
                if not cat_dir.is_dir():
                    continue
                if category and cat_dir.name != category:
                    continue
                for md in cat_dir.glob("*.md"):
                    result.add(f"{cat_dir.name}/{md.stem}")
        return sorted(result)

    def clear_cache(self) -> None:
        self._cache.clear()

    def _read(self, category: str, name: str) -> tuple[str, str]:
        """读 .md 文件。user/ 优先。返回 (text, source)。"""
        user_path = self._base / "user" / category / f"{name}.md"
        sys_path = self._base / "system" / category / f"{name}.md"
        if user_path.exists():
            return self._cached_read(user_path), "user"
        if sys_path.exists():
            return self._cached_read(sys_path), "system"
        raise FileNotFoundError(f"prompt not found: {category}/{name}")

    def _cached_read(self, path: Path) -> str:
        key = str(path)
        if key not in self._cache:
            self._cache[key] = path.read_text(encoding="utf-8")
        return self._cache[key]

    @staticmethod
    def _render(template: str, vars: dict[str, Any]) -> str:
        """${variable} regex 替换。未匹配保留原样 + warning。"""
        def replacer(m: re.Match) -> str:
            key = m.group(1)
            if key in vars:
                return str(vars[key])
            print(f"[PromptManager] warning: unbound ${{{key}}}", file=sys.stderr)
            return m.group(0)
        return re.sub(r"\$\{(\w+)\}", replacer, template)


_pm: PromptManager | None = None


def get_prompt_manager() -> PromptManager:
    global _pm
    if _pm is None:
        _pm = PromptManager(Path(__file__).parent)
    return _pm
