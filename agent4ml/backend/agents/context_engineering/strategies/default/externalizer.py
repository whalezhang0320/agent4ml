from __future__ import annotations

import json
import logging
import os
import uuid
from datetime import datetime
from typing import Any

from langchain_core.messages import HumanMessage, ToolMessage

from agent4ml.backend.agents.context_engineering.strategies.default._constants import CST
from agent4ml.backend.agents.middlewares.tagged_context_middleware import (
    AGENT4ML_EXTERNALIZED,
    AGENT4ML_EXTERNALIZED_META,
    AGENT4ML_EXTERNALIZED_PATH,
)

logger = logging.getLogger(__name__)


class ExternalizerExecutor:
    """外化超阈 tool result 到磁盘 + preview。

    P1 externalize_history：FIFO 时序 + 近 N 轮豁免 + 每轮保 1 + 幂等 + 同步写盘。
    """

    def __init__(
        self,
        externalize_dir: str = ".agent4ml/externalized",
        min_chars: int = 500,
        preview_chars: int = 500,
        exempt_rounds: int = 2,
        tool_metadata: dict[str, dict] | None = None,
    ) -> None:
        self._dir = externalize_dir
        self._min_chars = min_chars
        self._preview_chars = preview_chars
        self._exempt_rounds = exempt_rounds
        self._tool_metadata = tool_metadata or {}

    def _get_threshold(self, tool_name: str | None) -> int:
        """按工具元数据调阈值。有元数据取 max(min_chars, typical_tokens * 4)，无走 min_chars。"""
        if tool_name and tool_name in self._tool_metadata:
            typical_tokens = self._tool_metadata[tool_name].get("typical_output_tokens", 0)
            if typical_tokens > 0:
                return max(self._min_chars, typical_tokens * 4)
        return self._min_chars

    def externalize_if_needed(self, tool_result: ToolMessage) -> ToolMessage | None:
        """外化单个 ToolMessage：写盘成功 → 返替换版（preview + path）；失败 → 返 None。

        支持 str 和 list[dict] content 格式（MCP 工具常返回 list）。
        """
        if tool_result.additional_kwargs.get(AGENT4ML_EXTERNALIZED):
            return None
        text = self._extract_text(tool_result.content)
        threshold = self._get_threshold(tool_result.name)
        if len(text) <= threshold:
            return None
        path = self._write_to_disk(text, tool_result.tool_call_id, tool_result.name)
        if path is None:
            return None
        preview = text[: self._preview_chars] + f"\n\n[externalized path={path} tokens~{len(text) // 4}]"
        meta = {"tool_name": tool_result.name or "", "tokens_saved": len(text) // 4, "created_at": datetime.now(CST).isoformat()}
        new_kwargs = {**tool_result.additional_kwargs, AGENT4ML_EXTERNALIZED: True, AGENT4ML_EXTERNALIZED_PATH: path, AGENT4ML_EXTERNALIZED_META: meta}
        return tool_result.model_copy(update={"content": preview, "additional_kwargs": new_kwargs})

    def externalize_history(self, messages: list) -> list[ToolMessage] | None:
        """FIFO 外化：近 exempt_rounds 轮豁免 + 每轮保 1 + 幂等跳过。

        返回替换后的 ToolMessage 列表（同 id → add_messages 替换），不含未改消息。
        """
        turns = self._partition_turns(messages)
        # 诊断日志
        total_tools = sum(1 for t in turns for _, m in t if isinstance(m, ToolMessage))
        logger.info(
            "externalize_history: msgs=%d turns=%d total_tools=%d exempt_rounds=%d min_chars=%d dir=%s",
            len(messages), len(turns), total_tools, self._exempt_rounds, self._min_chars, self._dir,
        )
        if len(turns) <= self._exempt_rounds:
            logger.info("externalize_history: SKIP (turns=%d <= exempt_rounds=%d)", len(turns), self._exempt_rounds)
            return None

        exempt_start = len(turns) - self._exempt_rounds
        candidates: list[tuple[int, ToolMessage]] = []

        for turn_idx in range(exempt_start):
            tool_msgs = [(mi, m) for mi, m in turns[turn_idx] if isinstance(m, ToolMessage)]
            if len(tool_msgs) <= 1:
                logger.info("externalize_history: turn %d has %d tools (<=1, skip)", turn_idx, len(tool_msgs))
                continue
            # 保留最新 1 个（列表最后），其余可外化
            for mi, m in tool_msgs[:-1]:
                if m.additional_kwargs.get(AGENT4ML_EXTERNALIZED):
                    logger.info("externalize_history: turn %d tc=%s already externalized, skip", turn_idx, m.tool_call_id)
                    continue
                text = self._extract_text(m.content)
                if len(text) <= self._min_chars:
                    logger.info("externalize_history: turn %d tc=%s text_len=%d <= min_chars=%d, skip", turn_idx, m.tool_call_id, len(text), self._min_chars)
                    continue
                candidates.append((mi, m))
                logger.info("externalize_history: turn %d tc=%s CANDIDATE (text_len=%d)", turn_idx, m.tool_call_id, len(text))

        if not candidates:
            logger.info("externalize_history: NO candidates (all filtered), returning None")
            return None

        # FIFO——按 message_index 排序（最早优先）
        candidates.sort(key=lambda x: x[0])

        rewritten_list: list[ToolMessage] = []
        for _, msg in candidates:
            rewritten = self.externalize_if_needed(msg)
            if rewritten is not None:
                rewritten_list.append(rewritten)
                logger.info("externalize_history: WROTE file for tc=%s", msg.tool_call_id)
            else:
                logger.warning("externalize failed for tool_call_id=%s, keeping original", msg.tool_call_id)

        if rewritten_list:
            logger.info("externalize_history: DONE, %d/%d candidates externalized", len(rewritten_list), len(candidates))
        else:
            logger.info("externalize_history: all writes failed, returning None")
        return rewritten_list if rewritten_list else None

    @staticmethod
    def _partition_turns(messages: list) -> list[list[tuple[int, Any]]]:
        """按 HumanMessage 分 turn，返回 [(message_index, message)] 列表的列表。"""
        turns: list[list[tuple[int, Any]]] = []
        current: list[tuple[int, Any]] = []
        for i, msg in enumerate(messages):
            if isinstance(msg, HumanMessage):
                if current:
                    turns.append(current)
                current = [(i, msg)]
            else:
                current.append((i, msg))
        if current:
            turns.append(current)
        return turns

    def _is_externalizable(self, msg: ToolMessage) -> bool:
        text = self._extract_text(msg.content)
        return len(text) > self._min_chars

    @staticmethod
    def _extract_text(content: Any) -> str:
        """从 ToolMessage content（str | list[dict] | list[str]）提取纯文本。

        MCP 工具常返回 list[{"type": "text", "text": "..."}] 格式。
        """
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict) and "text" in item:
                    parts.append(str(item["text"]))
                elif isinstance(item, str):
                    parts.append(item)
            return "".join(parts)
        return str(content) if content is not None else ""

    def _write_to_disk(self, content: str, tool_call_id: str | None, tool_name: str | None) -> str | None:
        try:
            os.makedirs(self._dir, exist_ok=True)
            safe_name = (tool_name or "unknown").replace("/", "_").replace("\\", "_")
            short_id = (tool_call_id or uuid.uuid4().hex)[:12]

            # 尝试解析为 JSON → 格式化缩进存 .json；否则原样存 .txt
            try:
                parsed = json.loads(content)
                ext = ".json"
                write_content = json.dumps(parsed, indent=2, ensure_ascii=False)
            except (json.JSONDecodeError, TypeError):
                ext = ".txt"
                write_content = content

            filename = f"{safe_name}-{short_id}{ext}"
            filepath = os.path.join(self._dir, filename)
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(write_content)
            return filepath
        except (OSError, TypeError) as exc:
            logger.error("externalize write failed: %s (tool_call_id=%s)", exc, tool_call_id)
            return None
