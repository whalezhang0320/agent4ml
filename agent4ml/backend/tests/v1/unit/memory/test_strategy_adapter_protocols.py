from __future__ import annotations

from agent4ml.backend.agents.memory.adapters.graph_store import GraphStore
from agent4ml.backend.agents.memory.adapters.vector_store import VectorStore
from agent4ml.backend.agents.memory.decay_policy import DecayPolicy
from agent4ml.backend.agents.memory.forget_policy import ForgetPolicy
from agent4ml.backend.agents.memory.persona_policy import PersonaPolicy
from agent4ml.backend.agents.memory.schema import MemoryTrace, MemoryType


# ---------------------------------------------------------------------------
# Mock 实现
# ---------------------------------------------------------------------------


class _MockDecay:
    def compute_strength(self, trace: MemoryTrace, now: float) -> float:
        return 0.5


class _MockForget:
    def should_forget(self, trace: MemoryTrace, now: float) -> bool:
        return False


class _MockPersona:
    def get_static_profile(self, user_id: str) -> dict: return {}
    def get_dynamic_profile(self, user_id: str) -> dict: return {}
    def update_profile(self, user_id: str, facts: dict) -> None: ...


class _MockVector:
    def upsert(self, trace: MemoryTrace) -> None: ...
    def search(self, query_embedding: list[float], top_k: int = 5) -> list[tuple[str, float]]:
        return []
    def remove(self, trace_id: str) -> None: ...
    def rebuild(self, traces: list[MemoryTrace]) -> None: ...


class _MockGraph:
    def upsert_node(self, trace: MemoryTrace) -> None: ...
    def upsert_edge(self, trace_id_a: str, trace_id_b: str, *,
                    strength: float = 0.5, type: str = "related") -> None: ...
    def expand(self, trace_ids: list[str], *, max_depth: int = 2,
               min_strength: float = 0.3) -> list[tuple[str, float]]: return []
    def remove_node(self, trace_id: str) -> None: ...
    def rebuild(self, traces: list[MemoryTrace]) -> None: ...


# ---------------------------------------------------------------------------
# 策略 Protocol 测试
# ---------------------------------------------------------------------------


class TestDecayPolicy:
    def test_mock_isinstance(self) -> None:
        assert isinstance(_MockDecay(), DecayPolicy)

    def test_compute_strength_method(self) -> None:
        assert hasattr(DecayPolicy, "compute_strength")

    def test_returns_float(self) -> None:
        trace = MemoryTrace(id="t", content="c", type=MemoryType.EPISODIC)
        assert isinstance(_MockDecay().compute_strength(trace, 1000.0), float)


class TestForgetPolicy:
    def test_mock_isinstance(self) -> None:
        assert isinstance(_MockForget(), ForgetPolicy)

    def test_should_forget_method(self) -> None:
        assert hasattr(ForgetPolicy, "should_forget")

    def test_no_resolve_conflict(self) -> None:
        """B3：ForgetPolicy 删 resolve_conflict（矛盾走 reconsolidate/consolidate）。"""
        assert not hasattr(ForgetPolicy, "resolve_conflict")

    def test_should_forget_returns_bool(self) -> None:
        trace = MemoryTrace(id="t", content="c", type=MemoryType.EPISODIC)
        assert isinstance(_MockForget().should_forget(trace, 1000.0), bool)


class TestPersonaPolicy:
    def test_mock_isinstance(self) -> None:
        assert isinstance(_MockPersona(), PersonaPolicy)

    def test_three_methods(self) -> None:
        assert hasattr(PersonaPolicy, "get_static_profile")
        assert hasattr(PersonaPolicy, "get_dynamic_profile")
        assert hasattr(PersonaPolicy, "update_profile")

    def test_get_static_profile_returns_dict(self) -> None:
        assert isinstance(_MockPersona().get_static_profile("u"), dict)


# ---------------------------------------------------------------------------
# adapter Protocol 测试
# ---------------------------------------------------------------------------


class TestVectorStoreProtocol:
    def test_mock_isinstance(self) -> None:
        assert isinstance(_MockVector(), VectorStore)

    def test_four_methods(self) -> None:
        methods = [m for m in dir(VectorStore) if not m.startswith("_")]
        assert "upsert" in methods
        assert "search" in methods
        assert "remove" in methods
        assert "rebuild" in methods

    def test_search_returns_list_of_tuples(self) -> None:
        result = _MockVector().search([0.1, 0.2], top_k=5)
        assert isinstance(result, list)


class TestGraphStoreProtocol:
    def test_mock_isinstance(self) -> None:
        assert isinstance(_MockGraph(), GraphStore)

    def test_five_methods(self) -> None:
        methods = [m for m in dir(GraphStore) if not m.startswith("_")]
        assert "upsert_node" in methods
        assert "upsert_edge" in methods
        assert "expand" in methods
        assert "remove_node" in methods
        assert "rebuild" in methods

    def test_expand_returns_list_of_tuples(self) -> None:
        result = _MockGraph().expand(["t1"], max_depth=2, min_strength=0.3)
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# import 防火墙：adapter 不依赖具体实现类
# ---------------------------------------------------------------------------


class TestAdapterImportFirewall:
    def test_vector_store_only_depends_schema(self) -> None:
        """VectorStore 仅 import MemoryTrace from schema，不 import Chroma/Milvus/Faiss。"""
        import agent4ml.backend.agents.memory.adapters.vector_store as mod
        names = [n.lower() for n in dir(mod)]
        assert "chroma" not in names
        assert "milvus" not in names
        assert "faiss" not in names

    def test_graph_store_only_depends_schema(self) -> None:
        """GraphStore 仅 import Neo4j/Graphiti/NetworkX。"""
        import agent4ml.backend.agents.memory.adapters.graph_store as mod
        names = [n.lower() for n in dir(mod)]
        assert "neo4j" not in names
        assert "graphiti" not in names
        assert "networkx" not in names

    def test_adapter_protocols_no_cross_dep(self) -> None:
        """adapter Protocol 不依赖核心 Protocol（MemoryStore/Retriever/Manager）。"""
        import agent4ml.backend.agents.memory.adapters.vector_store as vmod
        import agent4ml.backend.agents.memory.adapters.graph_store as gmod
        assert "MemoryStore" not in dir(vmod)
        assert "Retriever" not in dir(vmod)
        assert "MemoryStore" not in dir(gmod)
        assert "Retriever" not in dir(gmod)
