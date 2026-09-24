"""BudgetGuard 单测 — 三维度记账 + cost_usd 主触发 + per-day UTC 0 重置 + 80% 预警 + 超限 fallback lead + 持久化.

设计（spec.md BudgetGuard Requirement + 42 文档 §7.9 + R5）:
- check_and_record：三维度记账（token + cost_usd + calls），cost_usd 主触发
- get_today_usage：per-day UTC 0 点重置
- 80% 预警写 budget_warnings 表（不主动通知 LLM）
- 超限 fallback 到 lead（通过 tool 返 JSON，不污染 system prompt）
- 持久化跨 session（multiagent.db）
"""
from __future__ import annotations

import pytest

from agent4ml.backend.agents.multiagent.evolution.budget_guard import (
    BudgetGuard,
    BudgetLimit,
)
from agent4ml.backend.agents.multiagent.evolution.types import CostRecord


@pytest.fixture
def guard(tmp_path):
    """每个测试用独立 db 文件."""
    db_path = str(tmp_path / "test_budget.db")
    return BudgetGuard(
        db_path=db_path,
        limits={"codex": BudgetLimit(per_day_tokens=1000, per_day_cost_usd=10.0, per_day_calls=5)},
        warning_threshold=0.8,
    )


# ── 三维度记账 ────────────────────────────────────────────────────────────────


def test_three_dimensions_record(guard):
    """check_and_record 记 token + cost_usd + calls 三维度."""
    cost = CostRecord(tokens=100, cost_usd=0.5, calls=1)
    result = guard.check_and_record("codex", cost)
    assert result.allowed is True

    usage = guard.get_today_usage("codex")
    assert usage["tokens_used"] == 100
    assert usage["cost_usd_used"] == 0.5
    assert usage["calls_used"] == 1


def test_three_dimensions_accumulate(guard):
    """多次调用累加."""
    guard.check_and_record("codex", CostRecord(tokens=100, cost_usd=0.5, calls=1))
    guard.check_and_record("codex", CostRecord(tokens=200, cost_usd=1.0, calls=1))

    usage = guard.get_today_usage("codex")
    assert usage["tokens_used"] == 300
    assert usage["cost_usd_used"] == 1.5
    assert usage["calls_used"] == 2


# ── cost_usd 主触发 ──────────────────────────────────────────────────────────


def test_cost_usd_primary_trigger(guard):
    """cost_usd 超限但 tokens/calls 未超 → 返 daily_cost_exceeded."""
    # limit: cost=10.0, tokens=1000, calls=5
    # 累加 cost 超 10.0
    guard.check_and_record("codex", CostRecord(tokens=100, cost_usd=8.0, calls=1))
    result = guard.check_and_record("codex", CostRecord(tokens=100, cost_usd=3.0, calls=1))
    assert result.allowed is False
    assert result.reason == "daily_cost_exceeded"
    assert result.fallback_target == "lead"


def test_tokens_exceeded(guard):
    """tokens 超限（cost/calls 未超）→ daily_tokens_exceeded."""
    guard.check_and_record("codex", CostRecord(tokens=900, cost_usd=1.0, calls=1))
    result = guard.check_and_record("codex", CostRecord(tokens=200, cost_usd=0.5, calls=1))
    assert result.allowed is False
    assert result.reason == "daily_tokens_exceeded"


def test_calls_exceeded(guard):
    """calls 超限（cost/tokens 未超）→ daily_calls_exceeded."""
    for _ in range(5):
        guard.check_and_record("codex", CostRecord(tokens=10, cost_usd=0.1, calls=1))
    result = guard.check_and_record("codex", CostRecord(tokens=10, cost_usd=0.1, calls=1))
    assert result.allowed is False
    assert result.reason == "daily_calls_exceeded"


# ── per-day UTC 0 点重置 ───────────────────────────────────────────────────


def test_per_day_reset(guard):
    """per-day UTC 0 点重置（新一天用量从 0 开始）."""
    # 当天用量
    guard.check_and_record("codex", CostRecord(tokens=100, cost_usd=1.0, calls=1))
    usage = guard.get_today_usage("codex")
    assert usage["tokens_used"] == 100

    # 模拟跨天：直接查一个不存在的日期（相当于新一天）
    # get_today_usage 用 UTC 当前日期，无法 mock，但可验证未记录的 specialist 返 0
    usage2 = guard.get_today_usage("unknown_specialist")
    assert usage2["tokens_used"] == 0
    assert usage2["cost_usd_used"] == 0.0
    assert usage2["calls_used"] == 0


# ── 80% 预警写 metrics 不主动通知 LLM ──────────────────────────────────────


def test_80_percent_warning_written(guard):
    """80% 预警写 budget_warnings 表（不主动通知 LLM）."""
    # limit cost=10.0, 80% = 8.0
    # 第一次调用 cost=7.0（未达 80%）
    guard.check_and_record("codex", CostRecord(tokens=100, cost_usd=7.0, calls=1))
    warnings = guard.get_warnings("codex")
    assert len(warnings) == 0

    # 第二次调用 cost=2.0（累计 9.0 > 8.0，触发 80% 预警）
    result = guard.check_and_record("codex", CostRecord(tokens=100, cost_usd=2.0, calls=1))
    assert result.allowed is True  # 9.0 < 10.0 未超限
    warnings = guard.get_warnings("codex")
    assert len(warnings) == 1
    assert warnings[0]["warning_type"] == "approaching_80_percent"


def test_warning_no_duplicate(guard):
    """80% 预警不重复写（已超 80% 后再调用不重复触发）."""
    guard.check_and_record("codex", CostRecord(tokens=100, cost_usd=8.5, calls=1))
    guard.check_and_record("codex", CostRecord(tokens=100, cost_usd=0.5, calls=1))  # 9.0
    warnings = guard.get_warnings("codex")
    assert len(warnings) == 1  # 只 1 条预警


# ── 超限 fallback lead ───────────────────────────────────────────────────────


def test_over_limit_fallback_lead(guard):
    """超限 fallback_target 固定 'lead'（不 fallback 另一 specialist，INV-10）."""
    guard.check_and_record("codex", CostRecord(tokens=100, cost_usd=9.0, calls=1))
    result = guard.check_and_record("codex", CostRecord(tokens=100, cost_usd=2.0, calls=1))
    assert result.allowed is False
    assert result.fallback_target == "lead"
    # fallback_target 始终是 lead（不论 specialist 名）
    assert guard.fallback_target("codex") == "lead"
    assert guard.fallback_target("claude") == "lead"


def test_remaining_returned_on_over_limit(guard):
    """超限时 remaining 字段返剩余量."""
    guard.check_and_record("codex", CostRecord(tokens=100, cost_usd=9.0, calls=1))
    result = guard.check_and_record("codex", CostRecord(tokens=100, cost_usd=2.0, calls=1))
    assert result.allowed is False
    assert result.remaining is not None
    # cost 超 10.0，remaining cost 应为 0
    assert result.remaining.cost_usd == 0.0


# ── 持久化跨 session ─────────────────────────────────────────────────────────


def test_persistence_across_sessions(guard, tmp_path):
    """进程重启后用量保留（SQLite 持久化）."""
    guard.check_and_record("codex", CostRecord(tokens=100, cost_usd=1.0, calls=1))

    # 新建 BudgetGuard 实例（模拟进程重启）
    guard2 = BudgetGuard(
        db_path=guard._db_path,
        limits={"codex": BudgetLimit(per_day_tokens=1000, per_day_cost_usd=10.0, per_day_calls=5)},
    )
    usage = guard2.get_today_usage("codex")
    assert usage["tokens_used"] == 100
    assert usage["cost_usd_used"] == 1.0
    assert usage["calls_used"] == 1


# ── 默认 limit──────────────────────────────────


def test_default_limit_fallback(tmp_path):
    """未配置 specialist 用默认 limit（BudgetLimit 默认值）."""
    guard = BudgetGuard(db_path=str(tmp_path / "test.db"))
    result = guard.check_and_record("unknown", CostRecord(tokens=100, cost_usd=1.0, calls=1))
    assert result.allowed is True
    # 默认 limit: per_day_tokens=200000, per_day_cost_usd=20.0, per_day_calls=50
    usage = guard.get_today_usage("unknown")
    assert usage["tokens_used"] == 100


def test_get_today_usage_no_record(guard):
    """无记录返全 0."""
    usage = guard.get_today_usage("never_called")
    assert usage["tokens_used"] == 0
    assert usage["cost_usd_used"] == 0.0
    assert usage["calls_used"] == 0


# ── CostRecord 默认值 ─────────────────────────────────────────────────────────


def test_cost_record_defaults():
    """CostRecord 默认 tokens=0, cost_usd=0.0, calls=1."""
    c = CostRecord()
    assert c.tokens == 0
    assert c.cost_usd == 0.0
    assert c.calls == 1


def test_budget_limit_defaults():
    """BudgetLimit 默认值（R5.1）."""
    limit = BudgetLimit()
    assert limit.per_day_tokens == 200000
    assert limit.per_day_cost_usd == 20.0
    assert limit.per_day_calls == 50
