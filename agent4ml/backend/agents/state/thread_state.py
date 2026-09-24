from __future__ import annotations

from agent4ml.backend.agents.state.types import ThreadState


def create_initial_thread_state(user_input: str) -> ThreadState:
    return {
        "messages": [],
        "user_input": user_input,
        "observations": [],
        "sources": [],
        "citations": [],
        "artifacts": [],
        "reflection_items": [],
        "errors": [],
        "metadata": {},
        "governance": None,
        "sandbox": None,
        "orchestration": None,
        "recalled_memories": None,
        "memory_updates": None,
    }
