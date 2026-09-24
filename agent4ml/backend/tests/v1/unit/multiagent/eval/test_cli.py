"""L3 CLI skeleton 单测.

测试要点:
- 5 verb 都返 NotImplementedError（暂不实现，L3-9.2 决策 b）
- L3_VERBS 包含 5 个 verb
"""
from __future__ import annotations

import pytest

from agent4ml.backend.agents.multiagent.eval.cli import (
    L3_VERBS,
    l3_decision_log,
    l3_degraded,
    l3_eval_history,
    l3_health,
    l3_status,
)


class TestL3CLISkeleton:
    def test_l3_verbs_has_5(self):
        assert L3_VERBS == ("status", "health", "decision-log", "eval-history", "degraded")

    def test_status_not_implemented(self):
        with pytest.raises(NotImplementedError):
            l3_status()

    def test_health_not_implemented(self):
        with pytest.raises(NotImplementedError):
            l3_health()

    def test_decision_log_not_implemented(self):
        with pytest.raises(NotImplementedError):
            l3_decision_log()

    def test_eval_history_not_implemented(self):
        with pytest.raises(NotImplementedError):
            l3_eval_history()

    def test_degraded_not_implemented(self):
        with pytest.raises(NotImplementedError):
            l3_degraded()
