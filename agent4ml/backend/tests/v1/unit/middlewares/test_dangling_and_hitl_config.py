"""Tests for DanglingToolCallMiddleware and HitlConfig."""

from __future__ import annotations

from types import SimpleNamespace

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agent4ml.backend.agents.config.schema import HitlConfig
from agent4ml.backend.agents.middlewares.dangling_tool_call_middleware import (
    DanglingToolCallMiddleware,
)


class TestHitlConfig:
    def test_defaults(self) -> None:
        cfg = HitlConfig()
        assert cfg.capability_failure_threshold == 2
        assert cfg.error_pattern_threshold == 3
        assert cfg.todo_stagnation_rounds == 5
        assert cfg.no_progress_timeout == 180
        assert cfg.max_help_requests == 3
        assert cfg.activity_heartbeat_interval == 10
        assert cfg.steer_enabled is True

    def test_custom_values(self) -> None:
        cfg = HitlConfig(max_help_requests=5, steer_enabled=False)
        assert cfg.max_help_requests == 5
        assert cfg.steer_enabled is False


class TestDanglingToolCallMiddleware:
    def _make_middleware(self) -> DanglingToolCallMiddleware:
        return DanglingToolCallMiddleware()

    def _make_state(self, messages: list) -> dict:
        return {"messages": messages}

    def test_no_dangling_returns_none(self) -> None:
        mw = self._make_middleware()
        state = self._make_state([
            AIMessage(content="hi", tool_calls=[{"id": "t1", "name": "bash", "args": {}}]),
            ToolMessage(content="ok", tool_call_id="t1", name="bash"),
        ])
        result = mw.before_model(state, SimpleNamespace())
        assert result is None

    def test_dangling_gets_synthetic_tool_message(self) -> None:
        mw = self._make_middleware()
        state = self._make_state([
            AIMessage(content="run cmd", tool_calls=[{"id": "t1", "name": "bash", "args": {"command": "ls"}}]),
        ])
        result = mw.before_model(state, SimpleNamespace())
        assert result is not None
        patch = result["messages"]
        assert len(patch) == 1
        assert isinstance(patch[0], ToolMessage)
        assert patch[0].tool_call_id == "t1"
        assert "interrupted" in patch[0].content

    def test_multiple_dangling_all_patched(self) -> None:
        mw = self._make_middleware()
        state = self._make_state([
            AIMessage(content="run", tool_calls=[
                {"id": "t1", "name": "bash", "args": {}},
                {"id": "t2", "name": "write_file", "args": {}},
            ]),
        ])
        result = mw.before_model(state, SimpleNamespace())
        assert result is not None
        assert len(result["messages"]) == 2

    def test_answered_call_not_patched(self) -> None:
        mw = self._make_middleware()
        state = self._make_state([
            AIMessage(content="run", tool_calls=[{"id": "t1", "name": "bash", "args": {}}]),
            ToolMessage(content="ok", tool_call_id="t1", name="bash"),
            AIMessage(content="run2", tool_calls=[{"id": "t2", "name": "bash", "args": {}}]),
        ])
        result = mw.before_model(state, SimpleNamespace())
        assert result is not None
        assert len(result["messages"]) == 1
        assert result["messages"][0].tool_call_id == "t2"

    def test_empty_messages_returns_none(self) -> None:
        mw = self._make_middleware()
        result = mw.before_model({"messages": []}, SimpleNamespace())
        assert result is None
