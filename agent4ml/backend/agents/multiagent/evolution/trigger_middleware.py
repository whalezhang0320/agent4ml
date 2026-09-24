"""L2TriggerMiddleware — L1 graph after_model 末尾轻量检查（D-6.3=C3）。

设计（42 文档 §7.3 + spec.md L2TriggerMiddleware Requirement）:
- after_model 纯数值判断（检查 metrics 阈值 + 1h 冷却窗口），不调 LLM
- 命中阈值时 enqueue EvolutionTask 到 queue（daemon thread 消费）
- 不修改 ThreadState（返 None，INV-4）
- < 1ms 延迟，不影响 L1 turn
- 触发判定委托 TriggerManager（四源 + 节流）
"""
from __future__ import annotations

import time
from typing import Any, override

from langchain.agents.middleware.types import AgentMiddleware

from agent4ml.backend.agents.multiagent.evolution.metrics_view import MetricsView
from agent4ml.backend.agents.multiagent.evolution.trigger_manager import TriggerManager
from agent4ml.backend.agents.multiagent.evolution.types import EvolutionTask, TriggerSource
from agent4ml.backend.agents.state.types import ThreadState


class L2TriggerMiddleware(AgentMiddleware):
    """L1 graph after_model 末尾轻量检查（D-6.3=C3）。

    after_model 不修改 ThreadState（返 None，INV-4）。
    _should_trigger 纯数值判断（冷却窗口 + 阈值），不调 LLM（INV-4）。
    命中时 enqueue EvolutionTask 到 queue（daemon thread 消费）。
    < 1ms 延迟，不影响 L1 turn。
    """

    state_schema = ThreadState  # type: ignore[assignment]

    def __init__(
        self,
        trigger_manager: TriggerManager,
        metrics_view: MetricsView,
        profile: str = "default",
    ) -> None:
        self._trigger_manager = trigger_manager
        self._metrics_view = metrics_view
        self._profile = profile

    @override
    def after_model(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        """L1 graph after_model 末尾轻量检查。

        不修改 ThreadState（返 None，INV-4）。
        命中阈值时 enqueue EvolutionTask 到 queue。
        """
        if not self._should_trigger(state):
            return None
        self._enqueue_evolution_task(self._extract_snapshot_id(state))
        return None  # 不修改 state

    @override
    async def aafter_model(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        """异步版委托同步（L2 触发是纯数值，无阻塞 IO）。"""
        return self.after_model(state, runtime)

    def _should_trigger(self, state: Any) -> bool:
        """纯数值判断：检查 metrics 阈值 + 1h 冷却窗口（不调 LLM，INV-4）。

        委托 TriggerManager.should_trigger（四源 + 节流）。
        """
        return self._trigger_manager.should_trigger(
            self._metrics_view, self._profile
        )

    def _extract_snapshot_id(self, state: Any) -> str:
        """从 ThreadState 提取 snapshot_id（用于 EvolutionTask.task_id）。"""
        if isinstance(state, dict):
            metadata = state.get("metadata") or {}
            if isinstance(metadata, dict):
                sid = metadata.get("snapshot_id") or metadata.get("thread_id")
                if isinstance(sid, str):
                    return sid
        return ""

    def _enqueue_evolution_task(self, snapshot_id: str) -> None:
        """enqueue EvolutionTask 到 queue（daemon thread 消费）。 决定 per-profile 串行。
        trigger_source 由 TriggerManager.should_trigger 返回的 last_trigger 决定。
        """
        trigger_source = self._trigger_manager.last_trigger_source or TriggerSource.PERIODIC
        task = EvolutionTask(
            task_id=snapshot_id or f"l2_{int(time.time())}",
            profile=self._profile,
            trigger_source=trigger_source,
            trigger_detail=self._trigger_manager.last_trigger_detail,
            timestamp=str(int(time.time())),
        )
        self._trigger_manager.enqueue(task)
