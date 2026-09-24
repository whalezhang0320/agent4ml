"""L2 evolution bootstrap - assemble L2 components + start daemon thread.

Design (spec.md multiagent-core setup_multiagent Requirement + 42 doc S13):
- setup_l2(config, metrics_store) -> L2Setup | None
- config.l2.enabled=true: construct VersionDAG/PromotionGate/EvolutionMutator/
  TriggerManager/L2TriggerMiddleware/BudgetGuard/OrchestrationMetricsL2/L2EvolutionWorker
  + start daemon thread
- config.l2.enabled=false: return None (L1 behavior unchanged)
- Inject L2 components into L1 make_specialist_tool + OrchestrationMiddleware
"""
from __future__ import annotations

import queue
from dataclasses import dataclass
from typing import Any

from agent4ml.backend.agents.multiagent.config import BudgetConfig, MultiAgentConfig
from agent4ml.backend.agents.multiagent.evolution.budget_guard import (
    BudgetGuard,
    BudgetLimit,
)
from agent4ml.backend.agents.multiagent.evolution.evolution_mutator import EvolutionMutator
from agent4ml.backend.agents.multiagent.evolution.failure_focuser import FailureFocuser
from agent4ml.backend.agents.multiagent.evolution.metrics_l2 import OrchestrationMetricsL2
from agent4ml.backend.agents.multiagent.evolution.metrics_view import MetricsView
from agent4ml.backend.agents.multiagent.evolution.promotion_gate import PromotionGate
from agent4ml.backend.agents.multiagent.evolution.trigger_manager import (
    TriggerManager,
    TriggerThresholds,
)
from agent4ml.backend.agents.multiagent.evolution.trigger_middleware import L2TriggerMiddleware
from agent4ml.backend.agents.multiagent.evolution.version_dag import VersionDAG
from agent4ml.backend.agents.multiagent.evolution.worker import L2EvolutionWorker


@dataclass
class L2Setup:
    """L2 evolution setup result - injected into L1 MultiAgentSetup."""

    version_dag: VersionDAG
    promotion_gate: PromotionGate
    evolution_mutator: EvolutionMutator
    trigger_manager: TriggerManager
    l2_trigger_middleware: L2TriggerMiddleware
    budget_guard: BudgetGuard
    metrics_l2: OrchestrationMetricsL2
    worker: L2EvolutionWorker
    task_queue: "queue.Queue"
    # L3 eval setup (None when config.l3.enabled=false, L2 behavior unchanged)
    l3_setup: Any = None


def setup_l2(
    config: MultiAgentConfig,
    metrics_store: Any,
) -> L2Setup | None:
    """Assemble L2 evolution layer.

    config.l2.enabled=false -> return None (L1 behavior unchanged).
    config.l2.enabled=true -> construct all L2 components + start daemon thread.
    """
    if not config.l2.enabled:
        return None

    # L2 schema already initialized by MultiAgentMetricsStore (v1->v2 migration).
    # VersionDAG uses same db path (Z3 mode).
    version_dag = VersionDAG(db_path=config.metrics_db_path)

    # MetricsView: L1 MultiAgentMetricsStore implements MetricsView Protocol
    metrics_view = metrics_store  # isinstance check done at runtime

    # FailureFocuser (24h window default)
    focuser = FailureFocuser(
        window_seconds=config.l2.failure_window_hours * 3600.0
    )

    # EvolutionMutator (LLM caller None for MVP - real LLM injected in Batch 13 follow-up)
    mutator = EvolutionMutator(
        llm_caller=None,  # MVP: no LLM caller, data-driven trigger later
        evolution_model=config.l2.evolution_model,
        max_retries=2,
    )

    # PromotionGate (no evaluator for MVP floor eval)
    # L3 启用时注入 bridge（lazy import 避免 L2→L3 依赖）
    bridge: Any = None
    if config.l3.enabled:
        from agent4ml.backend.agents.multiagent.eval.bridge import OrchestrationBridge
        from agent4ml.backend.agents.multiagent.eval.registry import SpecialistEvalRegistry
        from agent4ml.backend.agents.multiagent.eval.adapters.programmatic import ProgrammaticAdapter
        from agent4ml.backend.agents.multiagent.eval.adapters.llm_judge import LLMJudgeAdapter
        from agent4ml.backend.agents.multiagent.eval.adapters.longitudinal_pairs import LongitudinalPairsAdapter
        registry = SpecialistEvalRegistry()
        registry.register("programmatic", ProgrammaticAdapter(evaluator=None))
        registry.register("llm_judge", LLMJudgeAdapter(judge_fn=None))
        registry.register("longitudinal_pairs", LongitudinalPairsAdapter(evaluator=None))
        bridge = OrchestrationBridge(adapter_registry=registry)

    gate = PromotionGate(
        evaluator=None,
        version_dag=version_dag,
        eval_timeout_seconds=config.l2.eval_timeout_seconds,
        eval_sample_min=config.l2.eval_sample_min,
        eval_sample_max=config.l2.eval_sample_max,
        eval_task_max_reuse=config.l2.eval_task_max_reuse,
        bridge=bridge,
    )

    # TriggerManager (4 sources + 1h cooldown + per-profile serial)
    task_queue: queue.Queue = queue.Queue()
    thresholds = TriggerThresholds(
        failure_window_seconds=config.l2.failure_window_hours * 3600.0,
        failure_threshold=config.l2.failure_threshold,
        degradation_min_invoked=config.l2.degradation_min_invoked,
        degradation_threshold=config.l2.degradation_threshold,
        cost_alert_usd=config.l2.cost_alert_usd,
        latency_alert_seconds=config.l2.latency_alert_seconds,
    )
    trigger_manager = TriggerManager(
        task_queue=task_queue,
        cooldown_seconds=config.l2.cooldown_seconds,
        cron_interval_seconds=config.l2.cron_interval_hours * 3600.0,
        thresholds=thresholds,
    )

    # L2TriggerMiddleware (L1 graph after_model, no LLM, no state modify)
    l2_trigger_mw = L2TriggerMiddleware(
        trigger_manager=trigger_manager,
        metrics_view=metrics_view,
    )

    # BudgetGuard (3 dimensions, per-day UTC 0 reset)
    budget_limits = _build_budget_limits(config.budget)
    budget_guard = BudgetGuard(
        db_path=config.metrics_db_path,
        limits=budget_limits,
        warning_threshold=config.budget.warning_threshold,
    )

    # OrchestrationMetricsL2 (11 event types)
    metrics_l2 = OrchestrationMetricsL2(db_path=config.metrics_db_path)

    # L2EvolutionWorker (daemon thread + queue + 6h cron fallback)
    worker = L2EvolutionWorker(
        task_queue=task_queue,
        metrics_view=metrics_view,
        failure_focuser=focuser,
        evolution_mutator=mutator,
        promotion_gate=gate,
        version_dag=version_dag,
        metrics_l2=metrics_l2,
        cron_interval_seconds=config.l2.cron_interval_hours * 3600.0,
    )

    # L3 eval setup (config.l3.enabled=false -> None, L2 behavior unchanged)
    l3_setup: Any = None
    if config.l3.enabled:
        from agent4ml.backend.agents.multiagent.eval.bootstrap import setup_l3
        l3_setup = setup_l3(config, metrics_store, task_queue)

    return L2Setup(
        version_dag=version_dag,
        promotion_gate=gate,
        evolution_mutator=mutator,
        trigger_manager=trigger_manager,
        l2_trigger_middleware=l2_trigger_mw,
        budget_guard=budget_guard,
        metrics_l2=metrics_l2,
        worker=worker,
        task_queue=task_queue,
        l3_setup=l3_setup,
    )


def _build_budget_limits(budget_config: BudgetConfig) -> dict[str, BudgetLimit]:
    """Build per-specialist BudgetLimit dict from BudgetConfig."""
    limits: dict[str, BudgetLimit] = {}
    for name in ("codex", "claude", "subagent", "pi"):
        attr = getattr(budget_config, name, None)
        if attr is not None:
            limits[name] = BudgetLimit(
                per_day_tokens=attr.per_day_tokens,
                per_day_cost_usd=attr.per_day_cost_usd,
                per_day_calls=attr.per_day_calls,
            )
    return limits
