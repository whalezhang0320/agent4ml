"""L2EvolutionWorker unit tests - orchestration loop + failure handling + blocked + daemon thread.

Design (spec.md L2EvolutionWorker Requirement + 42 doc S7.4):
- run(task): focus -> mutate -> evaluate -> gate -> commit
- failure no block (catch + log + continue)
- 3 consecutive failures -> evolution_blocked
- daemon thread start/stop
"""
from __future__ import annotations

import queue
import threading
import time
from unittest.mock import MagicMock

import pytest

from agent4ml.backend.agents.multiagent.evolution.evolution_mutator import (
    EvolutionMutator,
    EvolutionResult as MutatorResult,
    LLMCaller,
)
from agent4ml.backend.agents.multiagent.evolution.failure_focuser import FailureFocuser
from agent4ml.backend.agents.multiagent.evolution.metrics_l2 import OrchestrationMetricsL2
from agent4ml.backend.agents.multiagent.evolution.metrics_view import (
    GlobalMetricsSnapshot,
    SpecialistMetricsSnapshot,
)
from agent4ml.backend.agents.multiagent.evolution.promotion_gate import (
    EvalResult,
    PromotionGate,
)
from agent4ml.backend.agents.multiagent.evolution.types import (
    ContextSummaryTemplate,
    EvolutionTask,
    FailureCategory,
    FailureStats,
    PromotionDecision,
    TriggerSource,
)
from agent4ml.backend.agents.multiagent.evolution.version_dag import VersionDAG
from agent4ml.backend.agents.multiagent.evolution.worker import L2EvolutionWorker


class _MockMetricsView:
    def __init__(self, failure_cats=None, specialists=None):
        self._failure_cats = failure_cats or {}
        self._specialists = specialists or {}

    def get_specialist_metrics(self, name, *, since=None):
        return self._specialists.get(name)

    def get_global_metrics(self, *, since=None):
        return GlobalMetricsSnapshot(
            total_calls=0, total_cost_usd=0.0, avg_latency_seconds=0.0,
            total_selections=0, total_completions=0, total_fallbacks=0,
        )

    def get_failure_categories(self, *, since=None):
        return self._failure_cats

    def get_recent_failures(self, *, category, limit=10):
        return []

    def list_specialists(self):
        return list(self._specialists.keys())


class _FakeLLMCaller:
    def __init__(self, output_json):
        self._output = output_json
        self.calls = []

    def call(self, prompt):
        self.calls.append(prompt)
        return self._output


def _ctx_json(skeleton="s2"):
    import json
    return json.dumps({
        "version": "v2", "template_id": "default", "rationale": "evolved",
        "extractors": ["LastNMessages"], "filters": ["TruncateAtTokens"],
        "max_tokens": 3000, "prompt_skeleton": skeleton,
    })


@pytest.fixture
def worker_setup(tmp_path):
    db_path = str(tmp_path / "test_worker.db")
    mv = _MockMetricsView(
        failure_cats={FailureCategory.CONTEXT_INSUFFICIENT: 5}
    )
    focuser = FailureFocuser()
    # Pre-commit v1 to VersionDAG
    version_dag = VersionDAG(db_path=db_path)
    v1 = ContextSummaryTemplate(
        version="v1", template_id="default",
        extractors=(), filters=(), max_tokens=2000, prompt_skeleton="s1",
    )
    version_dag.commit(
        v1,
        MagicMock(candidate_score=1.0, baseline_score=0.0, ci_low=1.0, ci_high=1.0, sample_size=1, success=True),
        decision="accept",
    )
    # Mutator with fake LLM
    caller = _FakeLLMCaller(_ctx_json())
    mutator = EvolutionMutator(llm_caller=caller, max_retries=2)
    # Gate with no evaluator (floor eval)
    gate = PromotionGate(evaluator=None, version_dag=version_dag)
    metrics_l2 = OrchestrationMetricsL2(db_path=db_path)
    q: queue.Queue = queue.Queue()
    worker = L2EvolutionWorker(
        task_queue=q,
        metrics_view=mv,
        failure_focuser=focuser,
        evolution_mutator=mutator,
        promotion_gate=gate,
        version_dag=version_dag,
        metrics_l2=metrics_l2,
    )
    return worker, version_dag, metrics_l2, q


def _make_task(task_id="t1", profile="default"):
    return EvolutionTask(
        task_id=task_id, profile=profile,
        trigger_source=TriggerSource.FAILURE_FOCUSED,
        trigger_detail="context_insufficient=5",
        artifact_type="context_summary",
    )


# -- orchestration loop --------------------------------------------------------


def test_run_evolution_success(worker_setup):
    """run(task): focus -> mutate -> gate -> commit (success path)."""
    worker, version_dag, metrics_l2, q = worker_setup
    task = _make_task()
    result = worker.run(task)
    # No evaluator -> gate.evaluate returns failure -> decision=FAILED or REJECT
    # (floor eval no evaluator -> EvalResult success=False -> decision=FAILED)
    assert result.task_id == "t1"
    # Without real evaluator, gate.decide returns FAILED (eval failed)
    assert result.decision in (PromotionDecision.FAILED, PromotionDecision.REJECT)


def test_run_no_dominant_category(worker_setup):
    """GOAL_UNCLEAR dominant -> no evolution (reject)."""
    worker, version_dag, metrics_l2, q = worker_setup
    # Override metrics_view to return non-evolvable dominant
    worker._metrics_view = _MockMetricsView(
        failure_cats={FailureCategory.GOAL_UNCLEAR: 10}
    )
    task = _make_task()
    result = worker.run(task)
    assert result.decision == PromotionDecision.REJECT


def test_run_mutator_failure_no_block(worker_setup):
    """Mutator failure -> catch + continue (no block, INV-15)."""
    worker, version_dag, metrics_l2, q = worker_setup
    # Override mutator to fail
    worker._mutator = MagicMock()
    worker._mutator.evolve_context_summary.return_value = MutatorResult(
        success=False, candidate=None, failure_type="json_parse", rationale="bad json"
    )
    task = _make_task()
    result = worker.run(task)
    assert result.decision == PromotionDecision.FAILED
    assert result.error == "json_parse"


def test_run_exception_caught(worker_setup):
    """Exception in run -> caught + FAILED returned."""
    worker, version_dag, metrics_l2, q = worker_setup
    worker._focuser = MagicMock()
    worker._focuser.analyze.side_effect = RuntimeError("focus failed")
    task = _make_task()
    result = worker.run(task)
    assert result.decision == PromotionDecision.FAILED
    assert "focus failed" in (result.error or "")


# -- 3 consecutive failures -> blocked ----------------------------------------


def test_consecutive_failures_blocked(worker_setup):
    """3 consecutive same-pattern failures -> evolution_blocked (INV-16)."""
    worker, version_dag, metrics_l2, q = worker_setup
    worker._mutator = MagicMock()
    worker._mutator.evolve_context_summary.return_value = MutatorResult(
        success=False, candidate=None, failure_type="json_parse", rationale="bad"
    )
    task = _make_task()
    # Run 3 times (same failure pattern)
    for _ in range(3):
        worker.run(task)
    # After 3 failures, should record blocked_marked
    from agent4ml.backend.agents.multiagent.evolution.metrics_l2 import EVENT_BLOCKED_MARKED
    blocked = metrics_l2.query_by_event_type(EVENT_BLOCKED_MARKED)
    assert len(blocked) >= 1


def test_success_resets_consecutive_failures(worker_setup):
    """Success resets consecutive failure count."""
    worker, version_dag, metrics_l2, q = worker_setup
    worker._mutator = MagicMock()
    worker._mutator.evolve_context_summary.return_value = MutatorResult(
        success=False, candidate=None, failure_type="json_parse", rationale="bad"
    )
    task = _make_task()
    # 2 failures
    worker.run(task)
    worker.run(task)
    # Reset to success (but gate has no evaluator -> decision=FAILED, no commit)
    worker._mutator.evolve_context_summary.return_value = MutatorResult(
        success=True, candidate=v2_template(), rationale="ok"
    )
    worker.run(task)
    # failure count reset (mutator success resets)
    assert "default:json_parse" not in worker._failure_counts


def v2_template():
    return ContextSummaryTemplate(
        version="v2", template_id="default",
        extractors=(), filters=(), max_tokens=3000, prompt_skeleton="s2",
    )


def v1_template():
    return ContextSummaryTemplate(
        version="v1", template_id="default",
        extractors=(), filters=(), max_tokens=2000, prompt_skeleton="s1",
    )


# -- daemon thread start/stop -------------------------------------------------


def test_start_stop_daemon_thread(worker_setup):
    """start() launches daemon thread, stop() joins."""
    worker, _, _, _ = worker_setup
    assert not worker.is_running
    worker.start()
    assert worker.is_running
    worker.stop()
    assert not worker.is_running


def test_daemon_thread_consumes_queue(worker_setup):
    """Daemon thread consumes queue tasks."""
    worker, version_dag, metrics_l2, q = worker_setup
    # Put a task in queue
    q.put(_make_task())
    worker.start()
    # Wait for consumption (short timeout)
    time.sleep(0.5)
    worker.stop()
    # Queue should be empty (task consumed)
    assert q.empty()


def test_daemon_thread_no_task_sleep(worker_setup):
    """No task in queue -> worker sleeps (no busy loop)."""
    worker, _, _, q = worker_setup
    worker._cron_interval = 0.1  # short for test
    worker.start()
    time.sleep(0.3)
    # Still running, no crash
    assert worker.is_running
    worker.stop()


# -- per-profile serial -------------------------------------------------------


def test_per_profile_serial_by_daemon_thread(worker_setup):
    """Per-profile serial (single daemon thread consume queue, INV-28)."""
    worker, _, _, q = worker_setup
    # Put 3 tasks
    for i in range(3):
        q.put(_make_task(task_id=f"t{i}"))
    worker.start()
    time.sleep(1.0)
    worker.stop()
    assert q.empty()  # all consumed (serially by single thread)
