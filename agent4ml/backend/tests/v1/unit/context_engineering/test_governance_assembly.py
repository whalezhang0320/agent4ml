"""治理层装配集成测：strategy 驱动挂载 / 公共始终挂 / 默认 minimal / 未注册跳过。"""

from __future__ import annotations

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from agent4ml.backend.agents.capabilities.registry import CapabilityRegistry
from agent4ml.backend.agents.config.schema import ContextGovernanceConfig
from agent4ml.backend.agents.context_engineering.builder import (
    build_governance_middlewares,
)
from agent4ml.backend.agents.context_engineering.registry import get_strategy_class
from agent4ml.backend.agents.leader.agent import LeaderAgent
from agent4ml.backend.agents.leader.factory import make_lead_agent
from agent4ml.backend.agents.middlewares.message_normalizer_middleware import (
    MessageNormalizerMiddleware,
)
from agent4ml.backend.agents.middlewares.tagged_context_middleware import (
    TaggedContextMiddleware,
)
from agent4ml.backend.agents.reporting.markdown_reporter import MarkdownReporter


def test_minimal_unregistered_assembles_public2_only() -> None:
    """minimal 非 bundle 名，未注册 → 仅公共 2，跳过 StrategyMiddleware。"""
    ms = build_governance_middlewares(ContextGovernanceConfig(strategy="minimal"))
    assert len(ms) == 2
    assert [type(m).__name__ for m in ms] == [
        "TaggedContextMiddleware",
        "MessageNormalizerMiddleware",
    ]


def test_public_2_always_present() -> None:
    ms = build_governance_middlewares(ContextGovernanceConfig(strategy="minimal"))
    assert any(isinstance(m, TaggedContextMiddleware) for m in ms)
    assert any(isinstance(m, MessageNormalizerMiddleware) for m in ms)


def test_unknown_strategy_class_raises() -> None:
    with pytest.raises(KeyError):
        get_strategy_class("nonexistent")


def test_default_config_strategy_default() -> None:
    from agent4ml.backend.agents.config.loader import load_config

    assert load_config().context_governance.strategy == "default"


def test_make_lead_agent_with_governance_builds_graph() -> None:
    registry = CapabilityRegistry(
        models={"researcher": FakeListChatModel(responses=["ok"])},
        tools={},
        reporter=MarkdownReporter(),
        artifact_store=object(),
    )
    agent = make_lead_agent(
        capability_registry=registry,
        context_governance=ContextGovernanceConfig(strategy="minimal"),
    )
    assert isinstance(agent, LeaderAgent)
    assert agent.graph is not None


def test_make_lead_agent_without_governance_still_works() -> None:
    registry = CapabilityRegistry(
        models={"researcher": FakeListChatModel(responses=["ok"])},
        tools={},
        reporter=MarkdownReporter(),
        artifact_store=object(),
    )
    agent = make_lead_agent(capability_registry=registry)
    assert isinstance(agent, LeaderAgent)
    assert agent.graph is not None
