"""Multi-Agent Protocol 契约层单测 — 8 Protocol mock 实现 + Registry。"""
from __future__ import annotations

import dataclasses

import pytest

from agent4ml.backend.agents.multiagent.credential_provider import (
    Credential,
    CredentialProvider,
)
from agent4ml.backend.agents.multiagent.exceptions import SpecialistNotFoundError
from agent4ml.backend.agents.multiagent.registry import SpecialistRegistry
from agent4ml.backend.agents.multiagent.result_summarizer import ResultSummarizer
from agent4ml.backend.agents.multiagent.sandbox_binder import (
    BoundSandbox,
    SandboxBinder,
)
from agent4ml.backend.agents.multiagent.context_summarizer import ContextSummarizer
from agent4ml.backend.agents.multiagent.specialist import SpecialistAgent
from agent4ml.backend.agents.multiagent.specialist_runtime import SpecialistRuntime
from agent4ml.backend.agents.multiagent.subagent import SubagentProvider
from agent4ml.backend.agents.multiagent.types import (
    ArtifactRef,
    SpecialistCapabilities,
    SpecialistCapability,
    SpecialistRawResult,
    SpecialistRequest,
    SpecialistResult,
    SubagentRequest,
    SubagentResult,
)


# ---------------------------------------------------------------------------
# Mock 实现（duck typing，不显式继承 Protocol）
# ---------------------------------------------------------------------------


class _MockSpecialist:
    """SpecialistAgent mock（实现 name + capabilities + invoke）。"""

    def __init__(self, name: str, caps: tuple[SpecialistCapability, ...] = ()) -> None:
        self._name = name
        self._caps = SpecialistCapabilities(capabilities=caps)

    @property
    def name(self) -> str:
        return self._name

    @property
    def capabilities(self) -> SpecialistCapabilities:
        return self._caps

    def invoke(self, request: SpecialistRequest) -> SpecialistRawResult:
        return SpecialistRawResult(raw_output=f"mock:{self._name}")


class _MockRuntime:
    """SpecialistRuntime mock（实现 invoke）。"""

    def invoke(self, request: SpecialistRequest) -> SpecialistRawResult:
        return SpecialistRawResult(raw_output="runtime-mock")


class _MockContextSummarizer:
    """ContextSummarizer mock（实现 summarize）。"""

    def summarize(self, state, goal: str, success_criteria: str) -> str:
        return f"ctx:{goal}"


class _MockResultSummarizer:
    """ResultSummarizer mock（实现 summarize）。"""

    def summarize(
        self, raw_output: str, artifacts, goal: str, success_criteria: str
    ) -> SpecialistResult:
        return SpecialistResult(
            specialist_name="mock", summary=raw_output[:50], success=True
        )


class _MockSandboxBinder:
    """SandboxBinder mock（实现 bind）。"""

    def bind(self, specialist_name: str, sandbox_id: str) -> BoundSandbox:
        return BoundSandbox(sandbox_id=sandbox_id, specialist_name=specialist_name)


class _MockCredentialProvider:
    """CredentialProvider mock（实现 get_credential）。"""

    def __init__(self, credential: Credential | None = None) -> None:
        self._cred = credential

    def get_credential(self) -> Credential | None:
        return self._cred


class _MockSubagentProvider:
    """SubagentProvider mock（实现 spawn）。"""

    def spawn(self, request: SubagentRequest) -> SubagentResult:
        return SubagentResult(summary="subagent-mock", success=True)


# ---------------------------------------------------------------------------
# Protocol 结构化类型验证（mock 不显式继承 Protocol 也能用）
# ---------------------------------------------------------------------------


def test_specialist_agent_duck_typed():
    s = _MockSpecialist("codex", caps=(SpecialistCapability.CODING,))
    assert s.name == "codex"
    assert s.capabilities.has(SpecialistCapability.CODING)
    raw = s.invoke(SpecialistRequest(
        goal="g", success_criteria="sc", context_summary="cs",
        sandbox_id=None, artifacts_path=None,
    ))
    assert raw.raw_output == "mock:codex"


def test_specialist_runtime_duck_typed():
    rt = _MockRuntime()
    raw = rt.invoke(SpecialistRequest(
        goal="g", success_criteria="sc", context_summary="cs",
        sandbox_id=None, artifacts_path=None,
    ))
    assert raw.raw_output == "runtime-mock"


def test_context_summarizer_duck_typed():
    cs = _MockContextSummarizer()
    summary = cs.summarize(state={}, goal="g", success_criteria="sc")
    assert summary == "ctx:g"


def test_result_summarizer_duck_typed():
    rs = _MockResultSummarizer()
    result = rs.summarize(
        raw_output="out", artifacts=[], goal="g", success_criteria="sc",
    )
    assert result.success is True
    assert result.specialist_name == "mock"


def test_sandbox_binder_duck_typed():
    binder = _MockSandboxBinder()
    bound = binder.bind("codex", "sb1")
    assert bound.sandbox_id == "sb1"
    assert bound.specialist_name == "codex"


def test_credential_provider_duck_typed():
    cred = Credential(kind="codex")
    provider = _MockCredentialProvider(cred)
    assert provider.get_credential() is cred

    empty_provider = _MockCredentialProvider(None)
    assert empty_provider.get_credential() is None


def test_subagent_provider_duck_typed():
    provider = _MockSubagentProvider()
    result = provider.spawn(SubagentRequest(
        goal="g", success_criteria="sc", context_summary="cs",
        sandbox_id=None, artifacts_path=None,
    ))
    assert result.success is True


# ---------------------------------------------------------------------------
# BoundSandbox + Credential frozen
# ---------------------------------------------------------------------------


def test_bound_sandbox_frozen():
    b = BoundSandbox(sandbox_id="sb1", specialist_name="codex")
    with pytest.raises(dataclasses.FrozenInstanceError):
        b.sandbox_id = "x"


def test_credential_frozen():
    c = Credential(kind="codex")
    with pytest.raises(dataclasses.FrozenInstanceError):
        c.kind = "x"


# ---------------------------------------------------------------------------
# SpecialistRegistry
# ---------------------------------------------------------------------------


def test_registry_register_and_get():
    reg = SpecialistRegistry()
    s = _MockSpecialist("codex", caps=(SpecialistCapability.CODING,))
    name = reg.register(s)
    assert name == "codex"
    assert reg.get("codex") is s


def test_registry_get_missing_raises():
    reg = SpecialistRegistry()
    with pytest.raises(SpecialistNotFoundError):
        reg.get("nonexistent")


def test_registry_list_all():
    reg = SpecialistRegistry()
    reg.register(_MockSpecialist("codex", caps=(SpecialistCapability.CODING,)))
    reg.register(_MockSpecialist("claude", caps=(SpecialistCapability.REVIEW,)))
    assert sorted(reg.list_specialists()) == ["claude", "codex"]


def test_registry_list_by_capability():
    reg = SpecialistRegistry()
    reg.register(_MockSpecialist("codex", caps=(SpecialistCapability.CODING,)))
    reg.register(_MockSpecialist("claude", caps=(SpecialistCapability.REVIEW,)))
    reg.register(_MockSpecialist("codex2", caps=(SpecialistCapability.CODING, SpecialistCapability.RESEARCH)))

    coding = reg.list_specialists(SpecialistCapability.CODING)
    assert sorted(coding) == ["codex", "codex2"]

    review = reg.list_specialists(SpecialistCapability.REVIEW)
    assert review == ["claude"]


def test_registry_register_overwrites():
    reg = SpecialistRegistry()
    s1 = _MockSpecialist("codex", caps=(SpecialistCapability.CODING,))
    s2 = _MockSpecialist("codex", caps=(SpecialistCapability.RESEARCH,))
    reg.register(s1)
    reg.register(s2)
    assert reg.get("codex") is s2
    assert len(reg) == 1


def test_registry_contains():
    reg = SpecialistRegistry()
    reg.register(_MockSpecialist("codex"))
    assert "codex" in reg
    assert "claude" not in reg


def test_registry_register_from_config_not_implemented():
    """register_from_config 契约留接口，Batch 10 bootstrap 实现。"""
    reg = SpecialistRegistry()
    with pytest.raises(NotImplementedError):
        reg.register_from_config(["codex", "claude"])


def test_registry_empty_list():
    reg = SpecialistRegistry()
    assert reg.list_specialists() == []
    assert reg.list_specialists(SpecialistCapability.CODING) == []
    assert len(reg) == 0
