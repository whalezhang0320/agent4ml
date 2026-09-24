"""L2 CLI skeleton tests - 8 verbs return NotImplementedError + does not affect core.

Design (spec.md agent-core L2 CLI Requirement):
- 8 verbs: status / history / artifacts / artifact / metrics / unblock / rollback / intent
- Each returns NotImplementedError when called
- Does not affect L2 core functionality
"""
from __future__ import annotations

import pytest

from agent4ml.backend.agents.multiagent.evolution.cli import (
    L2_VERBS,
    l2_artifact,
    l2_artifacts,
    l2_history,
    l2_intent,
    l2_metrics,
    l2_rollback,
    l2_status,
    l2_unblock,
)


# -- 8 verbs defined -----------------------------------------------------------


def test_l2_verbs_count():
    assert len(L2_VERBS) == 8


def test_l2_verbs_values():
    assert set(L2_VERBS) == {
        "status", "history", "artifacts", "artifact",
        "metrics", "unblock", "rollback", "intent",
    }


# -- each verb returns NotImplementedError ------------------------------------


def test_l2_status_not_implemented():
    with pytest.raises(NotImplementedError):
        l2_status()


def test_l2_history_not_implemented():
    with pytest.raises(NotImplementedError):
        l2_history()


def test_l2_artifacts_not_implemented():
    with pytest.raises(NotImplementedError):
        l2_artifacts()


def test_l2_artifact_not_implemented():
    with pytest.raises(NotImplementedError):
        l2_artifact("art_001")


def test_l2_metrics_not_implemented():
    with pytest.raises(NotImplementedError):
        l2_metrics()


def test_l2_unblock_not_implemented():
    with pytest.raises(NotImplementedError):
        l2_unblock()


def test_l2_rollback_not_implemented():
    with pytest.raises(NotImplementedError):
        l2_rollback("art_001", "v1")


def test_l2_intent_not_implemented():
    with pytest.raises(NotImplementedError):
        l2_intent()


# -- CLI does not affect L2 core ---------------------------------------------


def test_cli_skeleton_does_not_import_core_components():
    """CLI skeleton does not import VersionDAG/PromotionGate/etc (no side effects)."""
    import ast
    from pathlib import Path

    import agent4ml.backend.agents.multiagent.evolution.cli as mod

    source = Path(mod.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    import_names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                import_names.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                import_names.append(alias.name)
            if node.module:
                import_names.append(node.module)
    # CLI skeleton should not import L2 core components
    assert not any("VersionDAG" in name for name in import_names)
    assert not any("PromotionGate" in name for name in import_names)
    assert not any("EvolutionMutator" in name for name in import_names)
    assert not any("TriggerManager" in name for name in import_names)
