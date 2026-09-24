"""merge_orchestration reducer 单测 — 去重追加 + None 处理 + 回归。"""
from __future__ import annotations

from agent4ml.backend.agents.multiagent.types import ArtifactRef
from agent4ml.backend.agents.state.reducers import merge_orchestration
from agent4ml.backend.agents.state.thread_state import create_initial_thread_state


def test_new_none_preserves_existing():
    existing = {"active_specialists": ["codex"], "specialist_artifacts": []}
    assert merge_orchestration(existing, None) is existing


def test_existing_none_returns_new():
    new = {"active_specialists": ["codex"]}
    result = merge_orchestration(None, new)
    assert result is new


def test_both_none_returns_none():
    assert merge_orchestration(None, None) is None


def test_active_specialists_dedupe_by_name():
    existing = {"active_specialists": ["codex", "claude"]}
    new = {"active_specialists": ["codex", "subagent"]}
    result = merge_orchestration(existing, new)
    assert result["active_specialists"] == ["codex", "claude", "subagent"]


def test_active_specialists_preserves_order():
    existing = {"active_specialists": ["codex"]}
    new = {"active_specialists": ["claude", "codex", "subagent"]}
    result = merge_orchestration(existing, new)
    assert result["active_specialists"] == ["codex", "claude", "subagent"]


def test_specialist_artifacts_dedupe_by_path_with_dataclass():
    existing = {
        "specialist_artifacts": [
            ArtifactRef(path="/a.py", artifact_type="code", specialist_name="codex"),
        ]
    }
    new = {
        "specialist_artifacts": [
            ArtifactRef(path="/a.py", artifact_type="code", specialist_name="codex", description="updated"),
            ArtifactRef(path="/b.py", artifact_type="code", specialist_name="codex"),
        ]
    }
    result = merge_orchestration(existing, new)
    artifacts = result["specialist_artifacts"]
    assert len(artifacts) == 2
    assert artifacts[0].path == "/a.py"
    assert artifacts[0].description == "updated"
    assert artifacts[1].path == "/b.py"


def test_specialist_artifacts_dedupe_by_path_with_dict():
    existing = {
        "specialist_artifacts": [
            {"path": "/a.py", "artifact_type": "code", "specialist_name": "codex"},
        ]
    }
    new = {
        "specialist_artifacts": [
            {"path": "/a.py", "artifact_type": "code", "specialist_name": "codex", "description": "v2"},
            {"path": "/b.py", "artifact_type": "code", "specialist_name": "codex"},
        ]
    }
    result = merge_orchestration(existing, new)
    artifacts = result["specialist_artifacts"]
    assert len(artifacts) == 2
    assert artifacts[0]["description"] == "v2"
    assert artifacts[1]["path"] == "/b.py"


def test_partial_new_only_active_specialists():
    existing = {"specialist_artifacts": [ArtifactRef(path="/a.py", artifact_type="code", specialist_name="codex")]}
    new = {"active_specialists": ["codex"]}
    result = merge_orchestration(existing, new)
    assert result["active_specialists"] == ["codex"]
    assert len(result["specialist_artifacts"]) == 1


def test_partial_new_only_artifacts():
    existing = {"active_specialists": ["codex"]}
    new = {"specialist_artifacts": [ArtifactRef(path="/a.py", artifact_type="code", specialist_name="codex")]}
    result = merge_orchestration(existing, new)
    assert result["active_specialists"] == ["codex"]
    assert len(result["specialist_artifacts"]) == 1


def test_initial_thread_state_has_orchestration_none():
    state = create_initial_thread_state("test")
    assert state["orchestration"] is None


def test_initial_thread_state_existing_fields_preserved():
    """回归：加 orchestration 字段后既有字段不丢。"""
    state = create_initial_thread_state("test")
    assert state["user_input"] == "test"
    assert state["messages"] == []
    assert state["observations"] == []
    assert state["sources"] == []
    assert state["artifacts"] == []
    assert state["errors"] == []
    assert state["metadata"] == {}
    assert state["governance"] is None
    assert state["sandbox"] is None


def test_merge_orchestration_empty_lists():
    existing = {"active_specialists": [], "specialist_artifacts": []}
    new = {"active_specialists": [], "specialist_artifacts": []}
    result = merge_orchestration(existing, new)
    assert result["active_specialists"] == []
    assert result["specialist_artifacts"] == []
