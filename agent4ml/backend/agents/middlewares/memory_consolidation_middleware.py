"""MemoryConsolidationMiddleware — aafter_model 每 N 轮非阻塞触发沉淀（L5）。

承接 `design_docs/54` §4 Step C2。

挂载位置：MemoryMiddleware 后（召回在前，沉淀在后）。
不阻塞：submit 后立即 return None。
不抽取全量历史：取最近 N*2 条 messages（N 轮 ≈ 2N messages）。

INVARIANT：
- 仅 aafter_model 触发（abefore_model no-op）
- turn_count % N == 0 才触发（非 N 不 submit）
- submit 后立即 return None（worker 异步处理）
- messages 截断最近 N*2 条（避免 token 爆炸）
"""
from __future__ import annotations

import logging
from typing import Any

from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.messages import HumanMessage
from langgraph.runtime import Runtime

from agent4ml.backend.agents.memory.worker import MemoryTask, MemoryWorker
from agent4ml.backend.agents.state.types import ThreadState

logger = logging.getLogger(__name__)


class MemoryConsolidationMiddleware(AgentMiddleware):
    """自动沉淀中间件：aafter_model 每 N 轮丢任务到 worker。

    挂载位置：MemoryMiddleware 后（召回在前，沉淀在后）。
    不阻塞：submit 后立即 return None。
    """

    state_schema = ThreadState  # type: ignore[assignment]

    def __init__(
        self,
        worker: MemoryWorker,
        *,
        trigger_every_n_turns: int = 10,
    ) -> None:
        self._worker = worker
        self._n = max(1, trigger_every_n_turns)

    async def abefore_model(
        self, state: ThreadState, runtime: Runtime
    ) -> dict[str, Any] | None:
        return None  # 沉淀只在 after_model

    async def aafter_model(
        self, state: ThreadState, runtime: Runtime
    ) -> dict[str, Any] | None:
        messages = state.get("messages", []) or []
        # turn_count 只算 HumanMessage(用户轮次),不算 tool/AI/memory_recall
        user_turn_count = sum(1 for m in messages if isinstance(m, HumanMessage))
        if user_turn_count == 0 or user_turn_count % self._n != 0:
            return None

        # 取最近 N*2 条（N 轮 ≈ 2N messages，避免 token 爆炸）
        recent = messages[-(self._n * 2):]
        # thread_id 优先从 runtime config 取(fallback state → "unknown")
        thread_id = "unknown"
        try:
            config = getattr(runtime, "config", None) or {}
            configurable = config.get("configurable", {}) if isinstance(config, dict) else {}
            thread_id = configurable.get("thread_id") or state.get("thread_id") or "unknown"
        except Exception:
            thread_id = state.get("thread_id", "unknown")
        task = MemoryTask(
            thread_id=thread_id, messages=recent, turn_count=user_turn_count,
        )
        self._worker.submit(task)
        logger.debug(
            f"MemoryConsolidationMiddleware submitted task: "
            f"thread={thread_id} turn={user_turn_count}"
        )
        return None
