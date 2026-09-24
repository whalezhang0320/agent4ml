"""CodexContextSummarizer — rule-based extraction (code snippets + file paths, zero LLM).

设计（design.md §4）:
- per-specialist 输入端转换器
- 从 ThreadState 选取代码片段 + 文件路径（不传全量 context）
- 零 LLM：纯规则提取
"""
from __future__ import annotations

import re
from typing import Any

from agent4ml.backend.agents.state.types import ThreadState

_CODE_BLOCK_RE = re.compile(r"```[\s\S]*?```")
_MAX_CODE_BLOCKS = 3
_MAX_PATHS = 10
_MAX_SUMMARY_CHARS = 3000


def _field(item: Any, name: str) -> Any:
    if isinstance(item, dict):
        return item.get(name)
    return getattr(item, name, None)


class CodexContextSummarizer:
    """Codex specialist 输入端转换器（规则提取：代码 + 文件路径）。"""

    def summarize(
        self,
        state: ThreadState,
        goal: str,
        success_criteria: str,
        template: Any | None = None,
    ) -> str:
        parts: list[str] = [f"Goal: {goal}", f"Success criteria: {success_criteria}"]

        code_blocks = self._extract_code_blocks(state)
        if code_blocks:
            parts.append("Relevant code:")
            parts.extend(code_blocks)

        paths = self._extract_file_paths(state)
        if paths:
            parts.append(f"Files: {', '.join(paths)}")

        summary = "\n".join(parts)
        if len(summary) > _MAX_SUMMARY_CHARS:
            summary = summary[:_MAX_SUMMARY_CHARS] + "\n...(truncated)"
        return summary

    def _extract_code_blocks(self, state: ThreadState) -> list[str]:
        blocks: list[str] = []
        for msg in state.get("messages", []):
            content = _field(msg, "content")
            if not isinstance(content, str):
                continue
            found = _CODE_BLOCK_RE.findall(content)
            blocks.extend(found)
            if len(blocks) >= _MAX_CODE_BLOCKS:
                break
        return blocks[:_MAX_CODE_BLOCKS]

    def _extract_file_paths(self, state: ThreadState) -> list[str]:
        paths: list[str] = []
        for artifact in state.get("artifacts", []):
            path = _field(artifact, "path")
            if path and path not in paths:
                paths.append(str(path))
            if len(paths) >= _MAX_PATHS:
                break
        return paths
