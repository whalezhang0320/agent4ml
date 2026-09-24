"""Tests for StallDetectionMiddleware and HelpRequestMiddleware."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.types import Command

from agent4ml.backend.agents.middlewares.help_request_middleware import HelpRequestMiddleware
from agent4ml.backend.agents.middlewares.stall_detection_middleware import (
    StallDetectionMiddleware,
)


def _make_runtime(journal: MagicMock | None = None, run_id: str = "run-1") -> SimpleNamespace:
    return SimpleNamespace(context={"journal": journal or MagicMock(), "run_id": run_id})


class TestHelpRequestMiddleware:
    def test_intercepts_ask_help_and_returns_command_goto_end(self) -> None:
        mw = HelpRequestMiddleware()
        journal = MagicMock()
        request = SimpleNamespace(
            runtime=_make_runtime(journal),
            tool_call={"name": "ask_help", "id": "tc1", "args": {
                "question": "Which DB?", "help_type": "approach_choice",
                "options": ["PostgreSQL", "SQLite"],
            }},
        )
        result = mw.wrap_tool_call(request, lambda r: None)
        assert isinstance(result, Command)
        assert result.goto == "__end__"
        msgs = result.update.get("messages", [])
        assert any("Which DB?" in str(m.content) for m in msgs)
        journal.append.assert_any_call("help.requested", {
            "run_id": "run-1", "help_type": "approach_choice", "question": "Which DB?",
        })

    def test_non_ask_help_calls_handler_normally(self) -> None:
        mw = HelpRequestMiddleware()
        request = SimpleNamespace(
            runtime=_make_runtime(),
            tool_call={"name": "bash", "id": "tc1", "args": {"command": "ls"}},
        )
        expected = ToolMessage(content="ok", tool_call_id="tc1", name="bash")
        result = mw.wrap_tool_call(request, lambda r: expected)
        assert result is expected


class TestStallDetectionMiddleware:
    def test_no_stuck_returns_handler_result(self) -> None:
        mw = StallDetectionMiddleware()
        request = SimpleNamespace(
            runtime=_make_runtime(),
            tool_call={"name": "bash", "id": "tc1", "args": {"command": "ls"}},
        )
        ok = ToolMessage(content="ok", tool_call_id="tc1", name="bash")
        result = mw.wrap_tool_call(request, lambda r: ok)
        assert result is ok

    def test_stuck_after_tool_failure_sets_pending_flag(self) -> None:
        mw = StallDetectionMiddleware()
        journal = MagicMock()
        runtime = _make_runtime(journal)

        def _fire(cmd, tc_id):
            err = ToolMessage(content="permission denied", tool_call_id=tc_id, name="bash", status="error")
            req = SimpleNamespace(runtime=runtime, state={},
                tool_call={"name": "bash", "id": tc_id, "args": {"command": cmd}})
            return mw.wrap_tool_call(req, lambda r: err)

        # 阈值 5：5 次不同命令同 capability 触发 stuck
        _fire("apt install postgresql", "t1")
        _fire("which postgres", "t2")
        _fire("pg_isready", "t3")
        _fire("service postgres status", "t4")
        result = _fire("find / -name pg", "t5")
        assert result is not None
        assert mw._pending_stuck.get("run-1") is True

    def test_after_model_includes_jump_to_end_when_stuck(self) -> None:
        mw = StallDetectionMiddleware()
        journal = MagicMock()
        runtime = _make_runtime(journal)

        def _fire(cmd, tc_id):
            err = ToolMessage(content="permission denied", tool_call_id=tc_id, name="bash", status="error")
            req = SimpleNamespace(runtime=runtime, state={},
                tool_call={"name": "bash", "id": tc_id, "args": {"command": cmd}})
            mw.wrap_tool_call(req, lambda r: err)

        # 阈值 5：5 次失败触发 stuck
        _fire("apt install postgresql", "t1")
        _fire("which postgres", "t2")
        _fire("pg_isready", "t3")
        _fire("service postgres status", "t4")
        _fire("find / -name pg", "t5")

        result = mw.after_model({}, runtime)
        assert result is not None
        assert result.get("jump_to") == "end"
        assert any("STALL DETECTED" in str(m.content) for m in result.get("messages", []))

    def test_stuck_exception_sets_pending_flag(self) -> None:
        mw = StallDetectionMiddleware()
        runtime = _make_runtime()

        def failing_handler(req):
            raise RuntimeError("SandboxCommandError: permission denied")

        # 阈值 5：5 次异常触发 stuck
        for i, cmd in enumerate(("apt install postgresql", "which postgres", "pg_isready", "service pg status", "find / -name pg")):
            req = SimpleNamespace(runtime=runtime, state={},
                tool_call={"name": "bash", "id": f"t{i}", "args": {"command": cmd}})
            with __import__("pytest").raises(RuntimeError):
                mw.wrap_tool_call(req, failing_handler)

        assert mw._pending_stuck.get("run-1") is True

    def test_help_count_limit_forces_finalize_in_after_model(self) -> None:
        mw = StallDetectionMiddleware(max_help_requests=1)
        journal = MagicMock()
        runtime = _make_runtime(journal)

        def _fire_failure(cmd: str, tc_id: str) -> Any:
            err = ToolMessage(content="permission denied", tool_call_id=tc_id, name="bash", status="error")
            req = SimpleNamespace(runtime=runtime, state={},
                tool_call={"name": "bash", "id": tc_id, "args": {"command": cmd}})
            return mw.wrap_tool_call(req, lambda r: err)

        # First batch of 5: sets pending_stuck, after_model pauses
        _fire_failure("apt install pg1", "t1")
        _fire_failure("which pg", "t2")
        _fire_failure("pg_isready", "t3")
        _fire_failure("service pg status", "t4")
        _fire_failure("find / -name pg", "t5")
        result = mw.after_model({}, runtime)
        assert result is not None  # paused

        # Second batch of 5: sets pending_stuck again, after_model force-finalizes
        _fire_failure("apt install pg4", "t4")
        _fire_failure("apt install pg5", "t5")
        _fire_failure("apt install pg6", "t6")
        _fire_failure("apt install pg7", "t7")
        _fire_failure("apt install pg8", "t8")
        result = mw.after_model({}, runtime)
        assert result is not None
        assert result.get("jump_to") == "end"
        msgs = result.get("messages", [])
        assert any("HELP EXHAUSTED" in str(m.content) for m in msgs)
