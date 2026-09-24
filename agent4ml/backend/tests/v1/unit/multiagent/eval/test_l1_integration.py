"""L3Config + MultiAgentConfig.l3 单测.

测试要点（L1-L3 联动）:
- L3Config 默认值
- L3Config.llm_judge_weights 复用 skill TaskQualityJudge 权重值
- MultiAgentConfig.l3 字段存在
- STARTUP_ONLY_FIELDS 包含 l3.enabled

注：_write_decision_log_async 在 tools.py，但 tools.py import langchain_core（pre-existing 缺失），
其测试在 Batch 14 集成验证或 langchain 环境补测.
"""
from __future__ import annotations

import pytest

from agent4ml.backend.agents.multiagent.config import (
    L3Config,
    MultiAgentConfig,
    STARTUP_ONLY_FIELDS,
)


class TestL3Config:
    def test_defaults(self):
        config = L3Config()
        assert config.enabled is False
        assert config.default_eval_method == "programmatic"
        assert config.llm_judge_model is None
        assert config.health_check_window == 20
        assert config.degradation_threshold == 0.4
        assert config.degradation_delta == 0.15
        assert config.decision_log_retention_days == 90
        assert config.decision_log_archive_enabled is True

    def test_llm_judge_weights_match_skill(self):
        """llm_judge_weights 复用 skill TaskQualityJudge 权重值（D-L3-13）."""
        config = L3Config()
        assert config.llm_judge_weights == {
            "task_completion": 0.50,
            "response_quality": 0.35,
            "efficiency": 0.05,
            "tool_usage": 0.10,
        }

    def test_multiagent_config_has_l3(self):
        """MultiAgentConfig.l3 字段存在 + 默认 L3Config."""
        config = MultiAgentConfig()
        assert isinstance(config.l3, L3Config)
        assert config.l3.enabled is False

    def test_startup_only_fields_includes_l3(self):
        """STARTUP_ONLY_FIELDS 包含 l3.enabled."""
        assert "l3.enabled" in STARTUP_ONLY_FIELDS
