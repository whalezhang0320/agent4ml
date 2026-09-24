from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from agent4ml.backend.agents.memory.worker import MemoryTask
from agent4ml.backend.agents.middlewares.memory_consolidation_middleware import (
    MemoryConsolidationMiddleware,
)


def _make_state(messages: list, thread_id: str = "t1") -> dict:
    return {"messages": messages, "thread_id": thread_id}


def _make_worker() -> MagicMock:
    worker = MagicMock()
    return worker


class TestAbeforeModel:
    @pytest.mark.anyio
    async def test_returns_none(self) -> None:
        middleware = MemoryConsolidationMiddleware(_make_worker())
        result = await middleware.abefore_model({}, MagicMock())
        assert result is None


class TestAafterModel:
    @pytest.mark.anyio
    async def test_turn_zero_no_submit(self) -> None:
        worker = _make_worker()
        middleware = MemoryConsolidationMiddleware(worker, trigger_every_n_turns=10)
        state = _make_state([])

        await middleware.aafter_model(state, MagicMock())

        worker.submit.assert_not_called()

    @pytest.mark.anyio
    async def test_turn_not_divisible_no_submit(self) -> None:
        worker = _make_worker()
        middleware = MemoryConsolidationMiddleware(worker, trigger_every_n_turns=10)
        msgs = [HumanMessage(content=str(i)) for i in range(5)]
        state = _make_state(msgs)

        await middleware.aafter_model(state, MagicMock())

        worker.submit.assert_not_called()

    @pytest.mark.anyio
    async def test_turn_divisible_submits(self) -> None:
        worker = _make_worker()
        middleware = MemoryConsolidationMiddleware(worker, trigger_every_n_turns=10)
        msgs = [HumanMessage(content=str(i)) for i in range(10)]
        state = _make_state(msgs)

        await middleware.aafter_model(state, MagicMock())

        worker.submit.assert_called_once()
        task = worker.submit.call_args[0][0]
        assert isinstance(task, MemoryTask)
        assert task.turn_count == 10
        assert task.thread_id == "t1"

    @pytest.mark.anyio
    async def test_turn_20_submits(self) -> None:
        worker = _make_worker()
        middleware = MemoryConsolidationMiddleware(worker, trigger_every_n_turns=10)
        msgs = [HumanMessage(content=str(i)) for i in range(20)]
        state = _make_state(msgs)

        await middleware.aafter_model(state, MagicMock())

        worker.submit.assert_called_once()

    @pytest.mark.anyio
    async def test_returns_none_after_submit(self) -> None:
        worker = _make_worker()
        middleware = MemoryConsolidationMiddleware(worker, trigger_every_n_turns=10)
        msgs = [HumanMessage(content=str(i)) for i in range(10)]
        state = _make_state(msgs)

        result = await middleware.aafter_model(state, MagicMock())

        assert result is None

    @pytest.mark.anyio
    async def test_messages_truncated_to_n_times_2(self) -> None:
        worker = _make_worker()
        middleware = MemoryConsolidationMiddleware(worker, trigger_every_n_turns=5)
        msgs = [HumanMessage(content=str(i)) for i in range(20)]
        state = _make_state(msgs)

        await middleware.aafter_model(state, MagicMock())

        task = worker.submit.call_args[0][0]
        assert len(task.messages) == 10  # N*2 = 5*2

    @pytest.mark.anyio
    async def test_thread_id_from_state(self) -> None:
        worker = _make_worker()
        middleware = MemoryConsolidationMiddleware(worker, trigger_every_n_turns=2)
        msgs = [HumanMessage(content="a"), AIMessage(content="b"), HumanMessage(content="c")]
        state = {"messages": msgs, "thread_id": "my-thread-123"}

        await middleware.aafter_model(state, MagicMock())

        task = worker.submit.call_args[0][0]
        assert task.thread_id == "my-thread-123"

    @pytest.mark.anyio
    async def test_thread_id_default_unknown(self) -> None:
        worker = _make_worker()
        middleware = MemoryConsolidationMiddleware(worker, trigger_every_n_turns=2)
        msgs = [HumanMessage(content="a"), HumanMessage(content="b")]
        state = {"messages": msgs}  # no thread_id key

        await middleware.aafter_model(state, MagicMock())

        task = worker.submit.call_args[0][0]
        assert task.thread_id == "unknown"

    @pytest.mark.anyio
    async def test_n_equals_1_triggers_every_turn(self) -> None:
        worker = _make_worker()
        middleware = MemoryConsolidationMiddleware(worker, trigger_every_n_turns=1)
        msgs = [HumanMessage(content="a")]
        state = _make_state(msgs)

        await middleware.aafter_model(state, MagicMock())

        worker.submit.assert_called_once()
