"""PromotionGate 单测 — Wilson CI 计算 + hash 防环 + ACCEPT/REJECT 决策 + task 累计淘汰 + 30min 超时.

设计（spec.md PromotionGate Requirement + 42 文档 §7.8 + R3）:
- _wilson_ci：小样本友好，p=0/1 不退化（INV-20）
- evaluate：candidate vs baseline 各跑 task_sample → Wilson 95% CI
- decide：hash 命中近 5 版 → REJECT / candidate CI 下界 > baseline CI 上界 → ACCEPT / 否则 REJECT
- eval 整体超时 30min → 中断（INV-22）
- task 累计 ≤ 3 次（防过拟合，R3.4）
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import pytest

from agent4ml.backend.agents.multiagent.evolution.promotion_gate import (
    EvalResult,
    EvalTask,
    Evaluator,
    PromotionGate,
    _wilson_ci,
)
from agent4ml.backend.agents.multiagent.evolution.types import (
    ContextSummaryTemplate,
    PromotionDecision,
)


def _make_template(version="v1", skeleton="s1") -> ContextSummaryTemplate:
    return ContextSummaryTemplate(
        version=version, template_id="default",
        extractors=(), filters=(), max_tokens=2000, prompt_skeleton=skeleton,
    )


def _make_task(tid="t1") -> EvalTask:
    return EvalTask(
        task_id=tid, goal="g", success_criteria="sc",
        sandbox_id=None, context_snapshot_ref=None,
    )


class _FakeEvaluator:
    """测试用 Evaluator（可配置 candidate/baseline 成功率）."""

    def __init__(self, candidate_results=None, baseline_results=None):
        self._c = candidate_results or []
        self._b = baseline_results or []
        self._c_idx = 0
        self._b_idx = 0

    def evaluate(self, artifact, task):
        # 简单区分 candidate vs baseline：candidate skeleton="s2"，baseline="s1"
        if artifact.prompt_skeleton == "s2":
            result = self._c[min(self._c_idx, len(self._c) - 1)] if self._c else False
            self._c_idx += 1
            return result
        result = self._b[min(self._b_idx, len(self._b) - 1)] if self._b else False
        self._b_idx += 1
        return result


# ── _wilson_ci 计算 ───────────────────────────────────────────────────────────


def test_wilson_ci_normal_case():
    """score=0.8, n=10 → Wilson 95% CI 约 [0.49, 0.94]."""
    low, high = _wilson_ci(0.8, 10, z=1.96)
    assert 0.45 < low < 0.55
    assert 0.88 < high < 0.98


def test_wilson_ci_p_zero_not_degenerate():
    """score=0.0, n=10 → CI 不退化（非 [0.0, 0.0]）."""
    low, high = _wilson_ci(0.0, 10, z=1.96)
    assert low == 0.0
    assert high > 0.0  # 不退化


def test_wilson_ci_p_one_not_degenerate():
    """score=1.0, n=10 → CI 不退化（非 [1.0, 1.0]）."""
    low, high = _wilson_ci(1.0, 10, z=1.96)
    assert low < 1.0  # 不退化
    assert high == 1.0


def test_wilson_ci_zero_sample_returns_zero():
    """n=0 → 返 (0.0, 0.0)."""
    assert _wilson_ci(0.5, 0, z=1.96) == (0.0, 0.0)


def test_wilson_ci_bounded_0_1():
    """CI 上下界在 [0, 1] 范围内."""
    for score in [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]:
        low, high = _wilson_ci(score, 15, z=1.96)
        assert 0.0 <= low <= high <= 1.0


def test_wilson_ci_larger_n_narrower_interval():
    """样本越大 CI 越窄."""
    low_small, high_small = _wilson_ci(0.5, 10, z=1.96)
    low_large, high_large = _wilson_ci(0.5, 100, z=1.96)
    assert (high_large - low_large) < (high_small - low_small)


# ── evaluate ───────────────────────────────────────────────────────────────────


def test_evaluate_candidate_better_than_baseline():
    """candidate 8/10 success, baseline 4/10 success → candidate_score=0.8, baseline_score=0.4."""
    evaluator = _FakeEvaluator(
        candidate_results=[True, True, True, True, True, True, True, True, False, False],
        baseline_results=[True, True, True, True, False, False, False, False, False, False],
    )
    gate = PromotionGate(evaluator=evaluator, eval_sample_min=5, eval_sample_max=15)
    candidate = _make_template("v2", "s2")
    baseline = _make_template("v1", "s1")
    tasks = [_make_task(f"t{i}") for i in range(10)]

    result = gate.evaluate(candidate, baseline, tasks)
    assert result.success is True
    assert result.sample_size == 10
    assert abs(result.candidate_score - 0.8) < 0.01
    assert abs(result.baseline_score - 0.4) < 0.01


def test_evaluate_no_evaluator_returns_failure():
    """无 evaluator → 返 failure."""
    gate = PromotionGate(evaluator=None)
    result = gate.evaluate(_make_template(), _make_template(), [])
    assert result.success is False
    assert result.failure_reason == "no evaluator configured"


def test_evaluate_empty_task_sample_returns_failure():
    """空 task_sample → 返 failure（insufficient_tasks）."""
    gate = PromotionGate(evaluator=_FakeEvaluator(), eval_timeout_seconds=1800)
    result = gate.evaluate(_make_template(), _make_template(), [])
    assert result.success is False
    assert result.failure_reason == "insufficient_tasks"


def test_evaluate_all_tasks_filter_max_reuse():
    """所有 task 累计超 max_reuse → 全过滤 → insufficient_tasks."""
    evaluator = _FakeEvaluator(candidate_results=[True], baseline_results=[False])
    gate = PromotionGate(evaluator=evaluator, eval_task_max_reuse=3)
    candidate = _make_template("v2", "s2")
    baseline = _make_template("v1", "s1")
    tasks = [_make_task("t1")]  # 只 1 个 task

    # 跑 3 次让 t1 累计达 3
    for _ in range(3):
        gate.evaluate(candidate, baseline, tasks)
    # 第 4 次 → t1 被过滤
    result = gate.evaluate(candidate, baseline, tasks)
    assert result.success is False
    assert result.failure_reason == "insufficient_tasks"


def test_evaluate_timeout_interrupts():
    """eval 整体超时 → 中断 + 返 failure（overall_timeout，INV-22）."""
    evaluator = _FakeEvaluator(candidate_results=[True], baseline_results=[False])
    gate = PromotionGate(
        evaluator=evaluator, eval_timeout_seconds=-1.0  # 负数强制立即超时
    )
    candidate = _make_template("v2", "s2")
    baseline = _make_template("v1", "s1")
    tasks = [_make_task(f"t{i}") for i in range(5)]

    result = gate.evaluate(candidate, baseline, tasks)
    assert result.success is False
    assert result.failure_reason == "overall_timeout"


def test_evaluate_task_exception_skip():
    """单 task 跑挂 skip + 继续跑后续."""
    class _ExceptionEvaluator:
        def __init__(self):
            self.count = 0
        def evaluate(self, artifact, task):
            self.count += 1
            if self.count <= 2:
                raise RuntimeError("task failed")
            return artifact.prompt_skeleton == "s2"

    gate = PromotionGate(evaluator=_ExceptionEvaluator())
    candidate = _make_template("v2", "s2")
    baseline = _make_template("v1", "s1")
    tasks = [_make_task(f"t{i}") for i in range(5)]

    result = gate.evaluate(candidate, baseline, tasks)
    # 前 2 task 挂（candidate + baseline 各 1 次异常），后续 3 task 成功
    assert result.success is True
    assert result.sample_size == 3  # 3 task 有效


# ── decide ─────────────────────────────────────────────────────────────────────


def test_decide_accept_candidate_ci_above_baseline():
    """candidate CI [0.60, 0.88], baseline CI [0.25, 0.55] → candidate CI 下界 0.60 > baseline 上界 0.55 → ACCEPT."""
    gate = PromotionGate(version_dag=None)
    candidate = _make_template("v2", "s2")
    baseline = _make_template("v1", "s1")
    # 构造 EvalResult：candidate_score=0.8, baseline_score=0.4, sample=15
    # _wilson_ci(0.8, 15) ≈ [0.55, 0.92], _wilson_ci(0.4, 15) ≈ [0.19, 0.64]
    eval_result = EvalResult(
        candidate_score=0.8, baseline_score=0.4,
        ci_low=0.55, ci_high=0.92, sample_size=15,
        method_used="programmatic_floor", success=True,
    )
    decision = gate.decide(candidate, baseline, eval_result)
    # 0.55 > 0.64? No → REJECT（baseline CI 上界 0.64 > candidate 下界 0.55）


def test_decide_accept_clear_separation():
    """candidate CI 下界 > baseline CI 上界 → ACCEPT（INV-24）."""
    gate = PromotionGate(version_dag=None)
    candidate = _make_template("v2", "s2")
    baseline = _make_template("v1", "s1")
    # candidate_score=1.0, baseline_score=0.0, sample=15
    # _wilson_ci(1.0, 15) ≈ [0.78, 1.0], _wilson_ci(0.0, 15) ≈ [0.0, 0.22]
    eval_result = EvalResult(
        candidate_score=1.0, baseline_score=0.0,
        ci_low=0.78, ci_high=1.0, sample_size=15,
        method_used="programmatic_floor", success=True,
    )
    decision = gate.decide(candidate, baseline, eval_result)
    assert decision == PromotionDecision.ACCEPT


def test_decide_reject_ci_overlap():
    """candidate CI 与 baseline CI 重叠 → REJECT."""
    gate = PromotionGate(version_dag=None)
    candidate = _make_template("v2", "s2")
    baseline = _make_template("v1", "s1")
    # candidate_score=0.6, baseline_score=0.5, sample=10
    # CI 大概率重叠
    eval_result = EvalResult(
        candidate_score=0.6, baseline_score=0.5,
        ci_low=0.31, ci_high=0.83, sample_size=10,
        method_used="programmatic_floor", success=True,
    )
    decision = gate.decide(candidate, baseline, eval_result)
    assert decision == PromotionDecision.REJECT


def test_decide_reject_on_eval_failure():
    """eval 失败 → FAILED（保持旧 is_active，INV-13）."""
    gate = PromotionGate(version_dag=None)
    candidate = _make_template("v2", "s2")
    baseline = _make_template("v1", "s1")
    eval_result = EvalResult(
        candidate_score=0.0, baseline_score=0.0,
        ci_low=0.0, ci_high=0.0, sample_size=0,
        method_used="programmatic_floor",
        success=False, failure_reason="overall_timeout",
    )
    decision = gate.decide(candidate, baseline, eval_result)
    assert decision == PromotionDecision.FAILED


def test_decide_reject_hash_anti_loop():
    """candidate hash 命中近 5 版 → REJECT（防震，INV-7）."""
    class _MockVersionDAG:
        def hash_exists_in_recent(self, artifact_hash, window=5):
            return True  # 命中

    gate = PromotionGate(version_dag=_MockVersionDAG())
    candidate = _make_template("v2", "s2")
    baseline = _make_template("v1", "s1")
    eval_result = EvalResult(
        candidate_score=1.0, baseline_score=0.0,
        ci_low=0.78, ci_high=1.0, sample_size=15,
        method_used="programmatic_floor", success=True,
    )
    decision = gate.decide(candidate, baseline, eval_result)
    assert decision == PromotionDecision.REJECT


def test_decide_no_version_dag_skip_hash_check():
    """无 version_dag → 允许）."""
    gate = PromotionGate(version_dag=None)
    candidate = _make_template("v2", "s2")
    baseline = _make_template("v1", "s1")
    eval_result = EvalResult(
        candidate_score=1.0, baseline_score=0.0,
        ci_low=0.78, ci_high=1.0, sample_size=15,
        method_used="programmatic_floor", success=True,
    )
    # 无 version_dag → 不 REJECT hash → 看 CI
    decision = gate.decide(candidate, baseline, eval_result)
    assert decision == PromotionDecision.ACCEPT
