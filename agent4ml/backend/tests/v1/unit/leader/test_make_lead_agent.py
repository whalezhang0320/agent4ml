from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.tools import BaseTool

from agent4ml.backend.agents.capabilities.registry import CapabilityRegistry
from agent4ml.backend.agents.leader.agent import LeaderAgent
from agent4ml.backend.agents.leader.factory import make_lead_agent
from agent4ml.backend.agents.reporting.markdown_reporter import MarkdownReporter


class _StubTool(BaseTool):
    name: str = "web_search_mcp"
    description: str = "stub"

    def _run(self, query: str = "") -> str:
        return "stub result"


def _fake_model() -> FakeListChatModel:
    return FakeListChatModel(responses=["ok"])


def _registry(with_tool: bool = True) -> CapabilityRegistry:
    tools = {"web_search_mcp": _StubTool()} if with_tool else {}
    return CapabilityRegistry(
        models={"researcher": _fake_model()},
        tools=tools,
        reporter=MarkdownReporter(),
        artifact_store=object(),
    )


def test_make_lead_agent_returns_leader_agent_with_graph() -> None:
    registry = _registry()
    agent = make_lead_agent(capability_registry=registry, middleware_manager=None)
    assert isinstance(agent, LeaderAgent)
    assert agent.graph is not None
    assert agent.capability_registry is registry


def test_make_lead_agent_via_runnable_config() -> None:
    registry = _registry()
    config = {"configurable": {"expert_mode": True, "capability_registry": registry}}
    agent = make_lead_agent(runnable_config=config)
    assert isinstance(agent, LeaderAgent)


def test_make_lead_agent_raises_without_registry() -> None:
    import pytest

    with pytest.raises(ValueError, match="capability_registry is required"):
        make_lead_agent()


def test_make_lead_agent_no_tools_still_works() -> None:
    registry = _registry(with_tool=False)
    agent = make_lead_agent(capability_registry=registry)
    assert isinstance(agent, LeaderAgent)
    assert agent.graph is not None


def test_make_lead_agent_expert_mode_param() -> None:
    """expert_mode=True 参数应装配激进 middleware（Todo enforce_completion=True）。"""
    registry = _registry()
    agent = make_lead_agent(expert_mode=True, capability_registry=registry)
    assert isinstance(agent, LeaderAgent)
    # graph 编译应成功，含 checkpointer


def test_make_lead_agent_default_mode_param() -> None:
    """expert_mode=False 参数应装配温和 middleware（Todo enforce_completion=False）。"""
    registry = _registry()
    agent = make_lead_agent(expert_mode=False, capability_registry=registry)
    assert isinstance(agent, LeaderAgent)
    assert agent.graph is not None
