"""L3 eval bootstrap - assemble L3 components + start health check daemon thread.

Design (43 doc §4 + §11.8 L3-8.4 + spec.md setup_l2 Requirement):
- setup_l3(config, metrics_store, task_queue) -> L3Setup | None
- config.l3.enabled=true: construct SpecialistEvalRegistry + 3 adapter +
  OrchestrationBridge + SpecialistRuntimeTracker + DecisionLogWriter/Reader +
  MultiagentProgrammaticFacade + start daemon thread (6h health check)
- config.l3.enabled=false: return None (L2 behavior unchanged)
- daemon thread 复用 L2 pattern（threading.Thread daemon=True + 6h cron fallback）
- trigger_l2_evolution_if_degraded enqueue L2 cron queue（不直接调 L2 TriggerManager）
"""
from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from typing import Any

from agent4ml.backend.agents.multiagent.config import MultiAgentConfig
from agent4ml.backend.agents.multiagent.eval.adapters.llm_judge import LLMJudgeAdapter
from agent4ml.backend.agents.multiagent.eval.adapters.longitudinal_pairs import (
    LongitudinalPairsAdapter,
)
from agent4ml.backend.agents.multiagent.eval.adapters.programmatic import (
    ProgrammaticAdapter,
)
from agent4ml.backend.agents.multiagent.eval.bridge import OrchestrationBridge
from agent4ml.backend.agents.multiagent.eval.decision_log import (
    DecisionLogReader,
    DecisionLogWriter,
)
from agent4ml.backend.agents.multiagent.eval.facade import MultiagentProgrammaticFacade
from agent4ml.backend.agents.multiagent.eval.registry import SpecialistEvalRegistry
from agent4ml.backend.agents.multiagent.eval.runtime_tracker import (
    SpecialistRuntimeTracker,
    trigger_l2_evolution_if_degraded,
)


@dataclass
class L3Setup:
    """L3 eval setup result - injected into L2Setup for L1 bootstrap."""

    bridge: OrchestrationBridge
    registry: SpecialistEvalRegistry
    runtime_tracker: SpecialistRuntimeTracker
    decision_log_writer: DecisionLogWriter
    decision_log_reader: DecisionLogReader
    facade: MultiagentProgrammaticFacade
    health_thread: threading.Thread


def setup_l3(
    config: MultiAgentConfig,
    metrics_store: Any,
    task_queue: queue.Queue,
) -> L3Setup | None:
    """Assemble L3 eval layer.

    config.l3.enabled=false -> return None (L2 behavior unchanged).
    config.l3.enabled=true -> construct all L3 components + start daemon thread.
    """
    if not config.l3.enabled:
        return None

    # SpecialistEvalRegistry + 3 adapter (MVP evaluator=None, data-driven trigger)
    registry = SpecialistEvalRegistry()
    registry.register("programmatic", ProgrammaticAdapter(evaluator=None))
    registry.register("llm_judge", LLMJudgeAdapter(judge_fn=None))
    registry.register("longitudinal_pairs", LongitudinalPairsAdapter(evaluator=None))

    # OrchestrationBridge (L2 PromotionGate.bridge 注入)
    bridge = OrchestrationBridge(adapter_registry=registry)

    # SpecialistRuntimeTracker (健康监控 + degraded 检测)
    runtime_tracker = SpecialistRuntimeTracker(
        metrics_view=metrics_store,
        degradation_delta=config.l3.degradation_delta,
    )

    # DecisionLog Writer/Reader (跨 run lessons 累积)
    decision_log_writer = DecisionLogWriter(store=metrics_store)
    decision_log_reader = DecisionLogReader(store=metrics_store)

    # MultiagentProgrammaticFacade (L3 未启用时备用，向后兼容)
    facade = MultiagentProgrammaticFacade(evaluator=None)

    # 启动 L3 健康监控 daemon thread (6h 周期, 复用 L2 daemon thread pattern)
    health_thread = _start_health_check_thread(
        runtime_tracker=runtime_tracker,
        task_queue=task_queue,
        threshold=config.l3.degradation_threshold,
        interval_seconds=6 * 3600.0,
    )

    return L3Setup(
        bridge=bridge,
        registry=registry,
        runtime_tracker=runtime_tracker,
        decision_log_writer=decision_log_writer,
        decision_log_reader=decision_log_reader,
        facade=facade,
        health_thread=health_thread,
    )


def _start_health_check_thread(
    runtime_tracker: SpecialistRuntimeTracker,
    task_queue: queue.Queue,
    threshold: float,
    interval_seconds: float,
) -> threading.Thread:
    """启动 L3 健康监控 daemon thread (6h 周期, 复用 L2 daemon thread pattern).

    trigger_l2_evolution_if_degraded enqueue L2 cron queue（不直接调 L2 TriggerManager）.
    """
    stop_event = threading.Event()

    def _health_loop() -> None:
        while not stop_event.is_set():
            try:
                trigger_l2_evolution_if_degraded(
                    tracker=runtime_tracker,
                    cron_queue=task_queue,
                    threshold=threshold,
                )
            except Exception:
                pass  # daemon thread 不崩溃
            stop_event.wait(timeout=interval_seconds)

    thread = threading.Thread(target=_health_loop, daemon=True, name="l3-health-check")
    thread.start()
    return thread
