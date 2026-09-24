from __future__ import annotations

from pathlib import Path

import pytest

from agent4ml.backend.agents.memory.strategies.default._constants import (
    DECAY_COEFFICIENTS,
    DECAY_PARAMS,
    RETRIEVAL_WEIGHTS,
)
from agent4ml.backend.agents.memory.strategies.default.strategy import build_default_provider


class TestBuildDefaultProvider:
    """build_default_provider 测试移到 test_strategy.py（Layer 2 实现后）。"""

    def test_moved_to_test_strategy(self) -> None:
        """L2 实现后 build_default_provider 不再抛 NotImplementedError，测试在 test_strategy.py。"""
        from agent4ml.backend.agents.memory.strategies.default.strategy import build_default_provider
        # 验证不再抛 NotImplementedError（调用需 store+retriever 参数，这里只验证可 import）
        assert callable(build_default_provider)


class TestDecayParams:
    def test_has_three_types(self) -> None:
        assert "episodic" in DECAY_PARAMS
        assert "semantic" in DECAY_PARAMS
        assert "procedural" in DECAY_PARAMS

    def test_episodic_values(self) -> None:
        assert DECAY_PARAMS["episodic"] == {"base_strength": 0.7, "decay_rate": 0.1}

    def test_semantic_values(self) -> None:
        assert DECAY_PARAMS["semantic"] == {"base_strength": 0.8, "decay_rate": 0.02}

    def test_procedural_values(self) -> None:
        assert DECAY_PARAMS["procedural"] == {"base_strength": 0.9, "decay_rate": 0.005}


class TestRetrievalWeights:
    def test_similarity_0_7(self) -> None:
        assert RETRIEVAL_WEIGHTS["similarity"] == 0.7

    def test_strength_0_3(self) -> None:
        assert RETRIEVAL_WEIGHTS["strength"] == 0.3

    def test_weights_sum_to_one(self) -> None:
        assert RETRIEVAL_WEIGHTS["similarity"] + RETRIEVAL_WEIGHTS["strength"] == pytest.approx(1.0)


class TestDecayCoefficients:
    def test_access_boost_0_1(self) -> None:
        assert DECAY_COEFFICIENTS["access_boost"] == 0.1

    def test_importance_boost_0_05(self) -> None:
        assert DECAY_COEFFICIENTS["importance_boost"] == 0.05


class TestDirectoryStructureIsomorphic:
    """与 agents/context_engineering/strategies/default/ 同构（48 §2.2）。"""

    def _memory_default_dir(self) -> Path:
        return Path(__file__).resolve().parents[4] / "agents" / "memory" / "strategies" / "default"

    def _context_eng_default_dir(self) -> Path:
        return Path(__file__).resolve().parents[4] / "agents" / "context_engineering" / "strategies" / "default"

    def test_has_strategy_py(self) -> None:
        assert (self._memory_default_dir() / "strategy.py").exists()

    def test_has_constants_py(self) -> None:
        assert (self._memory_default_dir() / "_constants.py").exists()

    def test_has_init_py(self) -> None:
        assert (self._memory_default_dir() / "__init__.py").exists()

    def test_same_structure_as_context_engineering(self) -> None:
        """context_engineering/strategies/default/ 也有 strategy.py + _constants.py + __init__.py。"""
        ce_dir = self._context_eng_default_dir()
        assert (ce_dir / "strategy.py").exists()
        assert (ce_dir / "_constants.py").exists()
        assert (ce_dir / "__init__.py").exists()
