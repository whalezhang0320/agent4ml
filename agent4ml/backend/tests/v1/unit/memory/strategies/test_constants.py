from __future__ import annotations

from agent4ml.backend.agents.memory.strategies.default._constants import (
    ASSOCIATE_DEFAULTS,
    BM25_PARAMS,
    CONSOLIDATE_PARAMS,
    DECAY_COEFFICIENTS,
    DECAY_PARAMS,
    FORGET_THRESHOLDS,
    RETRIEVAL_WEIGHTS,
)


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


class TestDecayCoefficients:
    def test_access_boost_0_1(self) -> None:
        assert DECAY_COEFFICIENTS["access_boost"] == 0.1

    def test_importance_boost_0_05(self) -> None:
        assert DECAY_COEFFICIENTS["importance_boost"] == 0.05


class TestForgetThresholds:
    def test_strength_threshold_0_1(self) -> None:
        assert FORGET_THRESHOLDS["strength_threshold"] == 0.1

    def test_ttl_hours_720(self) -> None:
        assert FORGET_THRESHOLDS["ttl_hours"] == 720

    def test_conflict_window_hours_24(self) -> None:
        assert FORGET_THRESHOLDS["conflict_window_hours"] == 24


class TestConsolidateParams:
    def test_min_traces_2(self) -> None:
        assert CONSOLIDATE_PARAMS["min_traces_to_consolidate"] == 2

    def test_max_traces_10(self) -> None:
        assert CONSOLIDATE_PARAMS["max_traces_to_consolidate"] == 10

    def test_default_consolidated_type_semantic(self) -> None:
        assert CONSOLIDATE_PARAMS["default_consolidated_type"] == "semantic"

    def test_default_importance_boost_0_1(self) -> None:
        assert CONSOLIDATE_PARAMS["default_importance_boost"] == 0.1


class TestAssociateDefaults:
    def test_default_strength_0_5(self) -> None:
        assert ASSOCIATE_DEFAULTS["default_strength"] == 0.5

    def test_default_type_related(self) -> None:
        assert ASSOCIATE_DEFAULTS["default_type"] == "related"

    def test_max_associations_per_trace_20(self) -> None:
        assert ASSOCIATE_DEFAULTS["max_associations_per_trace"] == 20


class TestBm25Params:
    def test_k1_1_5(self) -> None:
        assert BM25_PARAMS["k1"] == 1.5

    def test_b_0_75(self) -> None:
        assert BM25_PARAMS["b"] == 0.75

    def test_epsilon_0_25(self) -> None:
        assert BM25_PARAMS["epsilon"] == 0.25
