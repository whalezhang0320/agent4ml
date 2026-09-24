"""Tests for RunJournalMiddleware tool result status detection."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import ToolMessage

from agent4ml.backend.agents.middlewares.run_journal_middleware import (
    RunJournalMiddleware,
    _result_status,
)


class TestResultStatus:
    def test_tool_message_success(self) -> None:
        msg = ToolMessage(content="ok", tool_call_id="t1", name="bash")
        assert _result_status(msg) == "ok"

    def test_tool_message_error(self) -> None:
        msg = ToolMessage(content="denied", tool_call_id="t1", name="bash", status="error")
        assert _result_status(msg) == "error"

    def test_command_with_failure_error(self) -> None:
        result = SimpleNamespace(update={"errors": [{"kind": "failure", "message": "boom"}]})
        assert _result_status(result) == "error"

    def test_command_with_success_error(self) -> None:
        result = SimpleNamespace(update={"errors": [{"kind": "success"}]})
        assert _result_status(result) == "ok"

    def test_command_with_error_tool_message(self) -> None:
        err_msg = ToolMessage(content="denied", tool_call_id="t1", name="bash", status="error")
        result = SimpleNamespace(update={"messages": [err_msg]})
        assert _result_status(result) == "error"

    def test_plain_string_returns_ok(self) -> None:
        assert _result_status("some output") == "ok"

    def test_none_returns_ok(self) -> None:
        assert _result_status(None) == "ok"


class TestJournalRecordsErrorStatus:
    def _make_middleware(self) -> tuple[RunJournalMiddleware, MagicMock]:
        journal = MagicMock()
        runtime = SimpleNamespace(context={"journal": journal, "run_id": "run-1"})
        request = SimpleNamespace(runtime=runtime, tool_call={"name": "bash", "args": {}})
        mw = RunJournalMiddleware()
        return mw, journal

    def test_error_tool_message_records_error_status(self) -> None:
        mw, journal = self._make_middleware()
        runtime = SimpleNamespace(context={"journal": journal, "run_id": "run-1"})
        request = SimpleNamespace(runtime=runtime, tool_call={"name": "bash", "args": {}})

        err_msg = ToolMessage(
            content="permission denied", tool_call_id="t1", name="bash", status="error"
        )

        def handler(req):
            return err_msg

        mw.wrap_tool_call(request, handler)

        calls = journal.append.call_args_list
        finished = calls[-1]
        assert finished.args[0] == "tool.finished"
        assert finished.args[1]["status"] == "error"

    def test_success_tool_message_records_ok_status(self) -> None:
        mw, journal = self._make_middleware()
        runtime = SimpleNamespace(context={"journal": journal, "run_id": "run-1"})
        request = SimpleNamespace(runtime=runtime, tool_call={"name": "bash", "args": {}})

        ok_msg = ToolMessage(content="done", tool_call_id="t1", name="bash")

        def handler(req):
            return ok_msg

        mw.wrap_tool_call(request, handler)

        calls = journal.append.call_args_list
        finished = calls[-1]
        assert finished.args[0] == "tool.finished"
        assert finished.args[1]["status"] == "ok"
