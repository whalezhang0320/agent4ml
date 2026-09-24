from __future__ import annotations

import json
import threading
import time
from unittest.mock import MagicMock, call

import pytest
from langchain_core.messages import HumanMessage

from agent4ml.backend.agents.memory.config import set_memory_config, MemoryConfig
from agent4ml.backend.agents.memory.schema import MemoryTrace, MemoryType
from agent4ml.backend.agents.memory.worker import MemoryTask, MemoryWorker


def _make_trace(trace_id: str, content: str = "test") -> MemoryTrace:
    """构造 mock MemoryTrace。"""
    import time as _time
    now = _time.time()
    return MemoryTrace(
        id=trace_id,
        content=content,
        type=MemoryType.EPISODIC,
        strength=0.7,
        base_strength=0.7,
        decay_rate=0.1,
        access_count=0,
        last_accessed=now,
        importance=0.5,
        associations=(),
        embedding=None,
        source="test",
        created_at=now,
        metadata={},
        operation_log=(),
    )


def _make_llm(response_content: str) -> MagicMock:
    """构造 mock LLM，invoke 返指定 content。"""
    llm = MagicMock()
    response = MagicMock()
    response.content = response_content
    llm.invoke.return_value = response
    return llm


def _make_manager(encoded_traces: list[MemoryTrace] | None = None) -> MagicMock:
    """构造 mock DefaultMemoryManager。"""
    manager = MagicMock()
    manager._store = MagicMock()
    if encoded_traces:
        manager._store.get.side_effect = lambda tid: next(
            (t for t in encoded_traces if t.id == tid), None
        )
        # list_by_type 返回非 forgotten 的 episodic traces
        manager._store.list_by_type.return_value = [
            t for t in encoded_traces if t.type == MemoryType.EPISODIC
        ]
    else:
        manager._store.list_by_type.return_value = []
    manager.encode.return_value = encoded_traces[0] if encoded_traces else _make_trace("default")
    return manager


class TestMemoryWorkerLifecycle:
    def test_start_launches_daemon_thread(self) -> None:
        worker = MemoryWorker(_make_manager(), _make_llm("[]"))
        worker.start()
        try:
            assert worker._thread is not None
            assert worker._thread.is_alive()
            assert worker._thread.daemon is True
            assert worker._thread.name == "memory-worker"
        finally:
            worker.shutdown(timeout=2.0)

    def test_start_idempotent(self) -> None:
        worker = MemoryWorker(_make_manager(), _make_llm("[]"))
        worker.start()
        first_thread = worker._thread
        worker.start()
        assert worker._thread is first_thread
        worker.shutdown(timeout=2.0)

    def test_shutdown_stops_thread(self) -> None:
        worker = MemoryWorker(_make_manager(), _make_llm("[]"))
        worker.start()
        worker.shutdown(timeout=2.0)
        assert worker._thread is None

    def test_shutdown_idempotent(self) -> None:
        worker = MemoryWorker(_make_manager(), _make_llm("[]"))
        worker.shutdown(timeout=1.0)
        worker.shutdown(timeout=1.0)


class TestMemoryWorkerSubmit:
    def test_submit_processes_task(self) -> None:
        processed = threading.Event()
        manager = _make_manager()
        llm = _make_llm("[]")
        worker = MemoryWorker(manager, llm)

        original_process = worker._process
        def patched_process(task):
            original_process(task)
            processed.set()
        worker._process = patched_process

        worker.start()
        try:
            task = MemoryTask(
                thread_id="t1",
                messages=[HumanMessage(content="hello")],
                turn_count=1,
            )
            worker.submit(task)
            assert processed.wait(timeout=3.0)
        finally:
            worker.shutdown(timeout=2.0)


class TestExtractAndEncode:
    def test_llm_returns_json_encodes(self) -> None:
        traces = [_make_trace("id1"), _make_trace("id2")]
        manager = _make_manager(traces)
        llm = _make_llm(json.dumps([
            {"content": "fact1", "type": "episodic", "importance": 0.8},
            {"content": "fact2", "type": "semantic", "importance": 0.6},
        ]))
        worker = MemoryWorker(manager, llm)
        task = MemoryTask("t1", [HumanMessage(content="hi")], 1)

        ids = worker._extract_and_encode(task)

        assert len(ids) == 2
        assert manager.encode.call_count == 2

    def test_llm_returns_non_json_returns_empty(self) -> None:
        manager = _make_manager()
        llm = _make_llm("not json at all")
        worker = MemoryWorker(manager, llm)
        task = MemoryTask("t1", [HumanMessage(content="hi")], 1)

        ids = worker._extract_and_encode(task)

        assert ids == []
        manager.encode.assert_not_called()

    def test_llm_raises_returns_empty(self) -> None:
        manager = _make_manager()
        llm = MagicMock()
        llm.invoke.side_effect = RuntimeError("LLM down")
        worker = MemoryWorker(manager, llm)
        task = MemoryTask("t1", [HumanMessage(content="hi")], 1)

        ids = worker._extract_and_encode(task)

        assert ids == []

    def test_llm_returns_non_list_returns_empty(self) -> None:
        manager = _make_manager()
        llm = _make_llm('{"not": "a list"}')
        worker = MemoryWorker(manager, llm)
        task = MemoryTask("t1", [HumanMessage(content="hi")], 1)

        ids = worker._extract_and_encode(task)

        assert ids == []

    def test_encode_failure_skipped(self) -> None:
        manager = _make_manager()
        manager.encode.side_effect = [Exception("encode fail"), _make_trace("ok")]
        llm = _make_llm(json.dumps([
            {"content": "fail", "type": "episodic"},
            {"content": "ok", "type": "episodic"},
        ]))
        worker = MemoryWorker(manager, llm)
        task = MemoryTask("t1", [HumanMessage(content="hi")], 1)

        ids = worker._extract_and_encode(task)

        assert len(ids) == 1  # only the successful one


class TestMaybeConsolidate:
    def test_below_threshold_no_consolidate(self) -> None:
        """store 中 episodic trace 数 < threshold → 不 consolidate。"""
        set_memory_config(MemoryConfig(use="default", phase2={"trigger_every_n_turns": 10}))
        traces = [_make_trace(f"id{i}", f"content{i}") for i in range(3)]
        manager = _make_manager(traces)
        llm = _make_llm("merged")
        worker = MemoryWorker(manager, llm)

        worker._maybe_consolidate(MemoryTask("t1", [], 1))

        manager.consolidate.assert_not_called()
        llm.invoke.assert_not_called()

    def test_above_threshold_consolidates(self) -> None:
        """store 中 episodic trace 数 >= threshold → consolidate。"""
        set_memory_config(MemoryConfig(use="default", phase2={"trigger_every_n_turns": 3}))
        traces = [_make_trace(f"id{i}", f"content{i}") for i in range(5)]
        manager = _make_manager(traces)
        llm = _make_llm("merged content")
        worker = MemoryWorker(manager, llm)

        worker._maybe_consolidate(MemoryTask("t1", [], 1))

        manager.consolidate.assert_called_once()

    def test_consolidate_max_10(self) -> None:
        """store 中 episodic trace 数 > 10 → 取最旧 10 条 consolidate。"""
        set_memory_config(MemoryConfig(use="default", phase2={"trigger_every_n_turns": 3}))
        traces = [_make_trace(f"id{i}", f"content{i}") for i in range(15)]
        manager = _make_manager(traces)
        llm = _make_llm("merged")
        worker = MemoryWorker(manager, llm)

        worker._maybe_consolidate(MemoryTask("t1", [], 1))

        args = manager.consolidate.call_args[0]
        assert len(args[0]) == 10  # E1 max

    def test_llm_consolidate_fail_no_raise(self) -> None:
        set_memory_config(MemoryConfig(use="default", phase2={"trigger_every_n_turns": 3}))
        traces = [_make_trace(f"id{i}") for i in range(5)]
        manager = _make_manager(traces)
        llm = MagicMock()
        llm.invoke.side_effect = RuntimeError("LLM down")
        worker = MemoryWorker(manager, llm)

        worker._maybe_consolidate(MemoryTask("t1", [], 1))

        manager.consolidate.assert_not_called()

    def test_consolidate_fail_no_raise(self) -> None:
        set_memory_config(MemoryConfig(use="default", phase2={"trigger_every_n_turns": 3}))
        traces = [_make_trace(f"id{i}") for i in range(5)]
        manager = _make_manager(traces)
        manager.consolidate.side_effect = ValueError("consolidate error")
        llm = _make_llm("merged")
        worker = MemoryWorker(manager, llm)

        worker._maybe_consolidate(MemoryTask("t1", [], 1))
