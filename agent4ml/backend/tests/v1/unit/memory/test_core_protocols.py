from __future__ import annotations

from agent4ml.backend.agents.memory.memory_manager import MemoryManager
from agent4ml.backend.agents.memory.memory_provider import MemoryProvider
from agent4ml.backend.agents.memory.memory_store import MemoryStore
from agent4ml.backend.agents.memory.retriever import Retriever
from agent4ml.backend.agents.memory.schema import MemoryTrace, MemoryType
from agent4ml.backend.agents.memory.types import MemoryFilter, MemoryQuery, RetrievalResult


# ---------------------------------------------------------------------------
# Mock 实现：用于验证 Protocol 契约（runtime_checkable 结构性子类型）
# ---------------------------------------------------------------------------


class _MockStore:
    def add(self, trace: MemoryTrace) -> None: ...
    def get(self, trace_id: str) -> MemoryTrace | None: return None
    def update(self, trace: MemoryTrace) -> None: ...
    def batch_update(self, traces: list[MemoryTrace]) -> None: ...
    def remove(self, trace_id: str) -> None: ...
    def list_by_type(self, type: MemoryType) -> list[MemoryTrace]: return []
    def list_by_filter(self, filter: MemoryFilter) -> list[MemoryTrace]: return []
    def list_all(self) -> list[MemoryTrace]: return []


class _MockRetriever:
    def retrieve(self, query: MemoryQuery) -> list[RetrievalResult]: return []


class _MockManager:
    def encode(self, content: str, type: MemoryType, *, importance: float = 0.5,
               source: str | None = None, metadata: dict | None = None) -> MemoryTrace:
        return MemoryTrace(id="t", content=content, type=type)

    def associate(self, trace_id_a: str, trace_id_b: str, *,
                  strength: float = 0.5, type: str = "related") -> None: ...

    def consolidate(self, trace_ids: list[str], merged_content: str) -> MemoryTrace:
        return MemoryTrace(id="t", content=merged_content, type=MemoryType.SEMANTIC)

    def reconsolidate(self, trace_id: str, new_content: str) -> MemoryTrace:
        return MemoryTrace(id=trace_id, content=new_content, type=MemoryType.EPISODIC)


class _MockProvider:
    def store(self) -> _MockStore: return _MockStore()
    def retriever(self) -> _MockRetriever: return _MockRetriever()
    def manager(self) -> _MockManager: return _MockManager()


class TestMemoryStoreProtocol:
    def test_mock_isinstance(self) -> None:
        assert isinstance(_MockStore(), MemoryStore)

    def test_method_count(self) -> None:
        """8 方法签名完整：add/get/update/batch_update/remove/list_by_type/list_by_filter/list_all。"""
        methods = [m for m in dir(MemoryStore) if not m.startswith("_")]
        assert "add" in methods
        assert "get" in methods
        assert "update" in methods
        assert "batch_update" in methods
        assert "remove" in methods
        assert "list_by_type" in methods
        assert "list_by_filter" in methods
        assert "list_all" in methods

    def test_retrieves_none_for_missing(self) -> None:
        """Protocol 契约：get 不存在返 None（非抛错）。"""
        store = _MockStore()
        assert store.get("missing") is None


class TestRetrieverProtocol:
    def test_mock_isinstance(self) -> None:
        assert isinstance(_MockRetriever(), Retriever)

    def test_retrieve_method_exists(self) -> None:
        assert hasattr(Retriever, "retrieve")

    def test_retrieve_returns_empty_list(self) -> None:
        retriever = _MockRetriever()
        result = retriever.retrieve(MemoryQuery(text="q"))
        assert result == []


class TestMemoryManagerProtocol:
    def test_mock_isinstance(self) -> None:
        assert isinstance(_MockManager(), MemoryManager)

    def test_four_operations_no_retrieve(self) -> None:
        """ 仅四操作（无 retrieve），Retrieve 移至 Retriever。"""
        methods = [m for m in dir(MemoryManager) if not m.startswith("_")]
        assert "encode" in methods
        assert "associate" in methods
        assert "consolidate" in methods
        assert "reconsolidate" in methods
        assert "retrieve" not in methods

    def test_encode_returns_memory_trace(self) -> None:
        manager = _MockManager()
        trace = manager.encode("content", MemoryType.EPISODIC)
        assert isinstance(trace, MemoryTrace)

    def test_associate_returns_none(self) -> None:
        manager = _MockManager()
        assert manager.associate("a", "b") is None

    def test_consolidate_returns_memory_trace(self) -> None:
        manager = _MockManager()
        trace = manager.consolidate(["a", "b"], "merged")
        assert isinstance(trace, MemoryTrace)

    def test_reconsolidate_returns_memory_trace(self) -> None:
        manager = _MockManager()
        trace = manager.reconsolidate("t1", "new content")
        assert isinstance(trace, MemoryTrace)


class TestMemoryProviderProtocol:
    def test_mock_isinstance(self) -> None:
        assert isinstance(_MockProvider(), MemoryProvider)

    def test_three_component_methods(self) -> None:
        """组合三组件：store + retriever + manager。"""
        methods = [m for m in dir(MemoryProvider) if not m.startswith("_")]
        assert "store" in methods
        assert "retriever" in methods
        assert "manager" in methods

    def test_store_returns_memory_store(self) -> None:
        provider = _MockProvider()
        assert isinstance(provider.store(), MemoryStore)

    def test_retriever_returns_retriever(self) -> None:
        provider = _MockProvider()
        assert isinstance(provider.retriever(), Retriever)

    def test_manager_returns_memory_manager(self) -> None:
        provider = _MockProvider()
        assert isinstance(provider.manager(), MemoryManager)


class TestImportNoCycle:
    def test_memory_store_only_depends_schema_types(self) -> None:
        """MemoryStore 仅依赖 schema + types，不依赖实现（无循环）。"""
        import agent4ml.backend.agents.memory.memory_store as mod
        # 不应 import MemoryProvider/Manager/Retriever（避免循环）
        assert "memory_provider" not in dir(mod)
        assert "memory_manager" not in dir(mod)

    def test_retriever_only_depends_types(self) -> None:
        """Retriever 仅依赖 types，不依赖 schema/store/manager。"""
        import agent4ml.backend.agents.memory.retriever as mod
        assert "MemoryStore" not in dir(mod)
        assert "MemoryManager" not in dir(mod)

    def test_memory_manager_only_depends_schema(self) -> None:
        """MemoryManager 仅依赖 schema，不依赖 store/retriever。"""
        import agent4ml.backend.agents.memory.memory_manager as mod
        assert "MemoryStore" not in dir(mod)
        assert "Retriever" not in dir(mod)

    def test_memory_provider_depends_three_protocols(self) -> None:
        """MemoryProvider 依赖 MemoryStore + Retriever + MemoryManager 三 Protocol。"""
        import agent4ml.backend.agents.memory.memory_provider as mod
        assert hasattr(mod, "MemoryStore")
        assert hasattr(mod, "Retriever")
        assert hasattr(mod, "MemoryManager")
