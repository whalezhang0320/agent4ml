from types import SimpleNamespace

from agent4ml.backend.agents.middlewares.run_journal_middleware import RunJournalMiddleware
from agent4ml.backend.agents.middlewares.summarization_middleware import SummarizationMiddleware
from agent4ml.backend.agents.middlewares.system_context_middleware import SystemContextMiddleware
from agent4ml.backend.agents.middlewares.title_middleware import TitleMiddleware
from agent4ml.backend.agents.middlewares.tool_call_middleware import ToolCallMiddleware


class _MockJournal:
    def __init__(self):
        self.events: list[tuple[str, dict]] = []

    def append(self, event_type: str, payload: dict) -> None:
        self.events.append((event_type, payload))


def _runtime(context=None):
    return SimpleNamespace(context=context or {})


def _tool_request(runtime):
    return SimpleNamespace(runtime=runtime)


def test_system_context_injects_metadata() -> None:
    mw = SystemContextMiddleware()
    result = mw.before_model({}, _runtime({"timezone": "Asia/Shanghai"}))
    assert result["metadata"]["system_context"]["agent"] == "Agent4ML deep research agent"
    assert result["metadata"]["system_context"]["timezone"] == "Asia/Shanghai"


def test_system_context_defaults_utc() -> None:
    mw = SystemContextMiddleware()
    result = mw.before_model({}, _runtime())
    assert result["metadata"]["system_context"]["timezone"] == "Asia/Shanghai"


def test_title_sets_from_research_question() -> None:
    mw = TitleMiddleware()
    result = mw.after_agent({"research_question": "Deep topic"}, _runtime())
    assert result["metadata"]["title"] == "Deep topic"


def test_title_falls_back_to_user_input() -> None:
    mw = TitleMiddleware()
    result = mw.after_agent({"user_input": "My question"}, _runtime())
    assert result["metadata"]["title"] == "My question"


def test_title_truncates_long_titles() -> None:
    mw = TitleMiddleware()
    result = mw.after_agent({"user_input": "x" * 100}, _runtime())
    assert len(result["metadata"]["title"]) == 60


# --- RunJournalMiddleware ---


def test_run_journal_records_agent_events() -> None:
    journal = _MockJournal()
    runtime = _runtime({"journal": journal, "run_id": "r1"})
    mw = RunJournalMiddleware()
    mw.before_agent({}, runtime)
    mw.after_agent({}, runtime)
    event_types = [e[0] for e in journal.events]
    assert "agent.started" in event_types
    assert "agent.finished" in event_types


def test_run_journal_records_model_events() -> None:
    journal = _MockJournal()
    runtime = _runtime({"journal": journal, "run_id": "r1"})
    mw = RunJournalMiddleware()
    mw.before_model({}, runtime)
    mw.after_model({}, runtime)
    event_types = [e[0] for e in journal.events]
    assert "llm.request" in event_types
    assert "llm.response" in event_types


def test_run_journal_wrap_tool_call_records_events() -> None:
    journal = _MockJournal()
    runtime = _runtime({"journal": journal, "run_id": "r1"})
    mw = RunJournalMiddleware()

    def handler(request):
        return "tool_result"

    result = mw.wrap_tool_call(_tool_request(runtime), handler)
    assert result == "tool_result"
    event_types = [e[0] for e in journal.events]
    assert "tool.called" in event_types
    assert "tool.finished" in event_types


def test_run_journal_no_journal_no_error() -> None:
    runtime = _runtime({"run_id": "r1"})
    mw = RunJournalMiddleware()
    mw.before_agent({}, runtime)  # should not raise


# --- Stub middlewares ---


def test_summarization_is_agent_middleware_noop() -> None:
    mw = SummarizationMiddleware()
    assert mw.before_model({}, _runtime()) is None


def test_tool_call_is_agent_middleware_noop() -> None:
    mw = ToolCallMiddleware()
    assert mw.before_agent({}, _runtime()) is None
