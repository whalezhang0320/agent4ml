"""Tests for ask_help tool, SituationReport, and ThreadState extensions."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from agent4ml.backend.agents.agent_tools.builtin.ask_help import ask_help_tool
from agent4ml.backend.agents.observability.situation_report import (
    SituationReport,
    build_programmatic_report,
    synthesize_options_with_llm,
)
from agent4ml.backend.agents.observability.stall_tracker import ToolFailure
from agent4ml.backend.agents.state.types import ThreadState


class TestAskHelpTool:
    def test_tool_name_and_return(self) -> None:
        result = ask_help_tool.invoke({
            "question": "Which DB?",
            "help_type": "approach_choice",
        })
        assert "HelpRequestMiddleware" in result

    def test_tool_has_correct_name(self) -> None:
        assert ask_help_tool.name == "ask_help"


class TestSituationReport:
    def _make_failures(self) -> list[ToolFailure]:
        return [
            ToolFailure("postgres", "permission", "apt install pg", "denied", 0),
            ToolFailure("postgres", "not_found", "find / -name pg", "not found", 1),
        ]

    def test_programmatic_report_has_attempts_and_blocker(self) -> None:
        report = build_programmatic_report(
            self._make_failures(), "capability exhausted: postgres",
        )
        assert len(report.attempts) == 2
        assert "apt install pg" in report.attempts[0]["command"]
        assert "postgres" in report.blocker

    def test_to_text_includes_sections(self) -> None:
        report = build_programmatic_report(
            self._make_failures(), "capability exhausted: postgres",
            environment={"user": "gem", "docker": "unavailable"},
        )
        text = report.to_text()
        assert "### Attempts" in text
        assert "### Environment" in text
        assert "### Blocker" in text
        assert "gem" in text

    def test_llm_synthesis_adds_options(self) -> None:
        report = build_programmatic_report(
            self._make_failures(), "capability exhausted: postgres",
        )
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = SimpleNamespace(
            content="[RECOMMENDED] Switch to Docker Compose blueprint\nUse SQLite for demo"
        )
        result = synthesize_options_with_llm(report, mock_llm)
        assert len(result.options) == 2
        assert "Docker Compose" in result.options[0]
        assert result.recommendation == result.options[0]

    def test_llm_synthesis_handles_failure_gracefully(self) -> None:
        report = build_programmatic_report(
            self._make_failures(), "capability exhausted: postgres",
        )
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = RuntimeError("LLM unavailable")
        result = synthesize_options_with_llm(report, mock_llm)
        assert result.options == []


class TestThreadStateExtensions:
    def test_thread_state_accepts_help_fields(self) -> None:
        state: ThreadState = {
            "messages": [],
            "help_request_count": 1,
            "pending_help": {"reason": "test"},
        }
        assert state["help_request_count"] == 1
        assert state["pending_help"]["reason"] == "test"
