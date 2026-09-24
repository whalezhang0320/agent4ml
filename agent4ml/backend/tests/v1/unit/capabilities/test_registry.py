from unittest.mock import MagicMock

import pytest

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool

from agent4ml.backend.agents.capabilities.registry import (
    CapabilityMissingError,
    CapabilityRegistry,
)


def _mock_model(name: str = "researcher") -> BaseChatModel:
    m = MagicMock(spec=BaseChatModel)
    m.name = name
    return m


def _mock_tool(name: str = "web_search_mcp") -> BaseTool:
    t = MagicMock(spec=BaseTool)
    t.name = name
    return t


def test_registry_returns_registered_capabilities() -> None:
    model = _mock_model("researcher")
    tool = _mock_tool("web_search_mcp")
    registry = CapabilityRegistry(
        models={"researcher": model, "reporter": _mock_model("reporter")},
        tools={"web_search_mcp": tool},
        reporter=object(),
        artifact_store=object(),
    )

    assert registry.get_model("researcher") is model
    assert registry.get_tool("web_search_mcp") is tool
    assert registry.get_reporter() is not None
    assert registry.get_artifact_store() is not None


def test_registry_missing_capability_error_is_clear() -> None:
    registry = CapabilityRegistry(models={}, tools={}, reporter=None, artifact_store=None)

    with pytest.raises(CapabilityMissingError, match="model not registered: researcher"):
        registry.get_model("researcher")
