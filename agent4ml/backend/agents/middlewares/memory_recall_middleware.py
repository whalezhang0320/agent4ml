"""MemoryMiddleware — before_model 召回注入 + after_model 清除（Phase 1）。

承接 `design_docs/53-memory-l4-middleware-bootstrap.md` §4 Step 1 + 00 §9.1。

挂载位置：Sandbox 后，HelpRequest/ToolCall 前（记忆可引用 sandbox 结果，注入为
user message 不进 tool pairing）。
注入方式：per-call HumanMessage（保护 prompt caching，不进 system_prompt cache prefix）。
set_turn_id：before_model 注入，after_model 清除（traceability C）。

块 B 迁移：从 agents/memory/middleware.py 迁到 agents/middlewares/memory_recall_middleware.py，
与既有 18 个 middleware 同层治理。内容 1:1 搬迁，类名 MemoryMiddleware 保留。

INVARIANT：
- 记忆是 middleware，不进 leader agent 主体（00 D9）
- 不进 system prompt cache：per-call HumanMessage(hide_from_ui=True)（00 D10）
- set_turn_id 注入/清除：before_model 注入，after_model 清除
- recalled_memories 只存索引(id+score+strength)，不存全量内容
"""

from __future__ import annotations

import logging
from typing import Any

from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.messages import HumanMessage
from langgraph.runtime import Runtime

from agent4ml.backend.agents.memory.schema import MemoryTrace
from agent4ml.backend.agents.memory.strategies.default.manager import set_turn_id
from agent4ml.backend.agents.memory.types import MemoryQuery, RetrievalResult
from agent4ml.backend.agents.state.types import ThreadState

logger = logging.getLogger(__name__)

# token 估算：1 token ≈ 4 字符（简化，精确需 tiktoken）
_CHARS_PER_TOKEN = 4


class MemoryMiddleware(AgentMiddleware):
    """记忆召回中间件（before_model recall + after_model 清除）。

    挂载位置：Sandbox 后，HelpRequest/ToolCall 前。
    注入方式：per-call HumanMessage（保护 prompt caching）。
    """

    state_schema = ThreadState  # type: ignore[assignment]

    def __init__(
        self,
        memory_provider: Any,
        *,
        enable_recall: bool = True,
        enable_extract: bool = False,
        token_budget: int = 2000,
    ) -> None:
        """初始化。

        Args:
            memory_provider: MemoryProvider（L3 DefaultMemoryProvider）
            enable_recall: before_model 召回开关（default 模式可选关）
            enable_extract: after_model 实时抽取开关（默认关，走 Phase 2 cron L5）
            token_budget: 召回注入 token 预算上限
        """
        self._provider = memory_provider
        self._enable_recall = enable_recall
        self._enable_extract = enable_extract
        self._token_budget = token_budget

    async def abefore_model(
        self, state: ThreadState, runtime: Runtime
    ) -> dict[str, Any] | None:
        """before_model：召回 + 注入 per-call HumanMessage + set_turn_id 注入。

        1A 强化写回在 HybridRetriever.retrieve 内部完成（caller 不负责）。
        """
        if not self._enable_recall:
            return None

        query = self._extract_query(state)
        if not query:
            return None

        # set_turn_id 注入（traceability C，Layer 2 manager operation_log.actor 取）
        turn_id = self._build_turn_id(state, runtime)
        set_turn_id(turn_id)

        # retrieve（L3 HybridRetriever，内部强化写回 1A + forgotten 过滤 3B）
        results = self._provider.retriever().retrieve(MemoryQuery(text=query))
        if not results:
            set_turn_id(None)  # 无召回，清除 turn_id
            return None

        # token budget 裁剪 + 格式化
        memories_text = self._format_recall(results, self._token_budget)

        # 注入 per-call HumanMessage（保护 prompt caching，不进 system prompt）
        return {
            "messages": [
                HumanMessage(
                    content=memories_text,
                    name="memory_recall",
                    additional_kwargs={"hide_from_ui": True},
                )
            ],
            # recalled_memories 只 + score + strength），不存全量内容
            "recalled_memories": [
                {"id": r.trace.id, "score": r.score, "strength": r.strength}
                for r in results
            ],
        }

    async def aafter_model(
        self, state: ThreadState, runtime: Runtime
    ) -> dict[str, Any] | None:
        """after_model：清除 turn_id + 可选抽取（默认关，走 Phase 2 cron L5）。"""
        # 清除 turn_id（traceability C，无论 before_model 是否注入都清除）
        set_turn_id(None)

        if not self._enable_extract:
            return None

        # 可选轻量抽取（默认关，走 Phase 2 cron L5）
        # Layer 4 仅保留 hook，不实现抽取逻辑
        return None

    def _extract_query(self, state: ThreadState) -> str:
        """从最后一条 user message 提取 query。"""
        messages = state.get("messages", []) or []
        # 从后往前找最后一条 HumanMessage
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                content = msg.content
                if isinstance(content, str):
                    return content
                # content 可能是 list（multimodal），取第一段文本
                if isinstance(content, list) and content:
                    first = content[0]
                    if isinstance(first, dict) and "text" in first:
                        return str(first["text"])
                return str(content)
        return ""

    def _format_recall(self, results: list[RetrievalResult], token_budget: int) -> str:
        """格式化召回结果 + token budget 裁剪。

       strength] content"，超 token_budget×4 字符截断。
        """
        max_chars = token_budget * _CHARS_PER_TOKEN
        lines: list[str] = ["[Recalled Memories]"]
        current_len = len(lines[0])
        for r in results:
            line = f"[score={r.score:.2f} strength={r.strength:.2f}] {r.trace.content}"
            if current_len + len(line) + 1 > max_chars:
                break  # 超预算截断
            lines.append(line)
            current_len += len(line) + 1
        return "\n".join(lines)

    def _build_turn_id(self, state: ThreadState, runtime: Runtime) -> str:
        """构造 turn_id（traceability C，关联记忆操作到对话轮次）。

        从 runtime config 取 thread_id + message 数作 turn 标识。
        """
        messages = state.get("messages", []) or []
        # thread_id 优先从 runtime config 取
        thread_id = "unknown"
        try:
            config = getattr(runtime, "config", None) or {}
            configurable = config.get("configurable", {}) if isinstance(config, dict) else {}
            thread_id = configurable.get("thread_id") or state.get("thread_id") or "unknown"
        except Exception:
            thread_id = state.get("thread_id", "unknown")
        return f"{thread_id}:turn:{len(messages)}"
