"""PiContextSummarizer — rule-based extraction (code snippets + file paths, zero LLM).

设计（design_docs/46 §10.3.6）:
- per-specialist 输入端转换器（与 CodexContextSummarizer 同构）
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


class PiContextSummarizer:
    """Pi specialist 输入端转换器（规则提取：代码 + 文件路径）。

    与 CodexContextSummarizer 同构——pi 是 coding specialist，
    需要代码片段 + 文件路径作为 context。
    """

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
            parts.append("Relevant files:")
            parts.extend(paths)

        summary = "\n".join(parts)
        if len(summary) > _MAX_SUMMARY_CHARS:
            summary = summary[:_MAX_SUMMARY_CHARS] + "\n... (truncated)"
        return summary

    def _extract_code_blocks(self, state: ThreadState) -> list[str]:
        """从 messages 提取代码块（```...```）。"""
        messages = state.get("messages") or []
        blocks: list[str] = []
        for msg in messages:
            content = _field(msg, "content")
            if not isinstance(content, str):
                continue
            for match in _CODE_BLOCK_RE.finditer(content):
                if len(blocks) >= _MAX_CODE_BLOCKS:
                    break
                blocks.append(match.group(0))
            if len(blocks) >= _MAX_CODE_BLOCKS:
                break
        return blocks

    def _extract_file_paths(self, state: ThreadState) -> list[str]:
        """从 observations + messages 提取文件路径（简单正则）。"""
        paths: list[str] = []
        path_re = re.compile(r"[\w/\-\.]+\.\w+")

        observations = state.get("observations") or []
        for obs in observations:
            content = _field(obs, "content")
            if not isinstance(content, str):
                continue
            for match in path_re.finditer(content):
                p = match.group(0)
                if p not in paths:
                    paths.append(p)
                if len(paths) >= _MAX_PATHS:
                    return paths

        messages = state.get("messages") or []
        for msg in messages:
            content = _field(msg, "content")
            if not isinstance(content, str):
                continue
            for match in path_re.finditer(content):
                p = match.group(0)
                if p not in paths:
                    paths.append(p)
                if len(paths) >= _MAX_PATHS:
                    return paths

        return paths
