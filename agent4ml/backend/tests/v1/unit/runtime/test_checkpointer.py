from langgraph.checkpoint.memory import InMemorySaver

from agent4ml.backend.agents.runtime.checkpointer import (
    get_checkpointer,
    reset_checkpointer,
)


def test_get_checkpointer_returns_singleton() -> None:
    reset_checkpointer()
    cp1 = get_checkpointer()
    cp2 = get_checkpointer()
    assert cp1 is cp2


def test_get_checkpointer_returns_in_memory_saver() -> None:
    reset_checkpointer()
    cp = get_checkpointer()
    assert isinstance(cp, InMemorySaver)


def test_reset_checkpointer_creates_new_instance() -> None:
    reset_checkpointer()
    cp1 = get_checkpointer()
    reset_checkpointer()
    cp2 = get_checkpointer()
    assert cp1 is not cp2
