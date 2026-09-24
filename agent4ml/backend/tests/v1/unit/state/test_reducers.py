import pytest

from agent4ml.backend.agents.state.reducers import ReducerConflictError, merge_thread_state
from agent4ml.backend.agents.state.types import (
    Artifact,
    Citation,
    ReflectionItem,
    Source,
)


def test_sources_are_deduplicated_by_source_id_or_url() -> None:
    state = {"sources": [Source(source_id="s1", url="https://a.test", title="A")]}
    patch = {
        "sources": [
            Source(source_id="s1", url="https://a.test/updated", title="A2"),
            Source(source_id="s2", url="https://a.test", title="A3"),
            Source(source_id="s3", url="https://b.test", title="B"),
        ]
    }

    merged = merge_thread_state(state, patch)

    assert [source.source_id for source in merged["sources"]] == ["s1", "s3"]
    assert merged["sources"][0].title == "A2"


def test_citations_artifacts_and_reflections_merge_by_identity() -> None:
    state = {
        "citations": [
            Citation(citation_id="c1", source_id="s1", quote="Q", claim="C")
        ],
        "artifacts": [
            Artifact(artifact_id="a1", artifact_type="markdown", title="Draft", path="x")
        ],
        "reflection_items": [
            ReflectionItem(
                item_id="r1",
                scope="run",
                kind="evidence_gap",
                question="Need source?",
                status="open",
            )
        ],
    }
    patch = {
        "citations": [
            Citation(citation_id="c1", source_id="s1", quote="Q2", claim="C2")
        ],
        "artifacts": [
            Artifact(
                artifact_id="a1",
                artifact_type="markdown",
                title="Final",
                path="final.md",
            )
        ],
        "reflection_items": [
            ReflectionItem(
                item_id="r1",
                scope="run",
                kind="evidence_gap",
                question="Need source?",
                status="addressed",
            )
        ],
    }

    merged = merge_thread_state(state, patch)

    assert merged["citations"][0].quote == "Q2"
    assert merged["artifacts"][0].path == "final.md"
    assert merged["reflection_items"][0].status == "addressed"


def test_metadata_cannot_override_core_field_semantics() -> None:
    with pytest.raises(ReducerConflictError, match="metadata cannot contain core field"):
        merge_thread_state({}, {"metadata": {"plan": "hidden"}})


def test_final_report_last_write_wins() -> None:
    """final_report reducer 改 last-write-wins：多轮 chat 覆盖旧报告不崩溃。"""
    merged = merge_thread_state({"final_report": "报告A"}, {"final_report": "报告B"})
    assert merged["final_report"] == "报告B"


def test_final_report_none_preserves_current() -> None:
    """incoming=None 时保留当前值。"""
    merged = merge_thread_state({"final_report": "报告A"}, {"final_report": None})
    assert merged["final_report"] == "报告A"


def test_final_report_first_write() -> None:
    """首次写入（current=None）。"""
    merged = merge_thread_state({}, {"final_report": "报告A"})
    assert merged["final_report"] == "报告A"


def test_errors_are_appended_and_limited() -> None:
    state = {"errors": [{"error_id": f"e{i}", "message": str(i)} for i in range(95)]}
    patch = {"errors": [{"error_id": f"e{i}", "message": str(i)} for i in range(95, 110)]}

    merged = merge_thread_state(state, patch)

    assert len(merged["errors"]) == 100
    assert merged["errors"][0]["error_id"] == "e10"
    assert merged["errors"][-1]["error_id"] == "e109"


# --- Field-level reducer tests (for Annotated[type, reducer] in ThreadState) ---


def test_field_level_merge_sources_deduplicates() -> None:
    from agent4ml.backend.agents.state.reducers import merge_sources

    current = [Source(source_id="s1", url="https://a.test", title="A")]
    incoming = [
        Source(source_id="s1", url="https://a.test/updated", title="A2"),
        Source(source_id="s2", url="https://b.test", title="B"),
    ]
    result = merge_sources(current, incoming)
    assert len(result) == 2
    assert result[0].title == "A2"


def test_field_level_merge_errors_limits() -> None:
    from agent4ml.backend.agents.state.reducers import merge_errors

    current = [{"error_id": f"e{i}"} for i in range(95)]
    incoming = [{"error_id": f"e{i}"} for i in range(95, 110)]
    result = merge_errors(current, incoming)
    assert len(result) == 100


def test_field_level_merge_observations_appends() -> None:
    from agent4ml.backend.agents.state.reducers import merge_observations

    current = ["obs1"]
    incoming = ["obs2", "obs3"]
    result = merge_observations(current, incoming)
    assert result == ["obs1", "obs2", "obs3"]


def test_threadstate_extends_agent_state() -> None:
    from agent4ml.backend.agents.state.types import ThreadState

    # TypedDict doesn't support issubclass at runtime; verify field inheritance
    # by collecting annotations from the MRO.
    annotations: dict = {}
    for cls in ThreadState.__mro__:
        annotations.update(getattr(cls, "__annotations__", {}))
    assert "messages" in annotations  # inherited from AgentState
    assert "sources" in annotations  # declared in ThreadState


def test_threadstate_as_create_agent_state_schema() -> None:
    from unittest.mock import MagicMock

    from langchain.agents import create_agent
    from langchain_core.language_models import BaseChatModel
    from langchain_core.messages import AIMessage

    from agent4ml.backend.agents.state.types import ThreadState

    m = MagicMock(spec=BaseChatModel)
    m.bind_tools.return_value = m
    m.invoke.return_value = AIMessage(content="ok")
    graph = create_agent(model=m, tools=[], state_schema=ThreadState)
    assert graph.__class__.__name__ == "CompiledStateGraph"
