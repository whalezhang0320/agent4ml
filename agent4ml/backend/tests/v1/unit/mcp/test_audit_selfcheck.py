"""Batch B6 self-check: McpAuditMiddleware (structural, no real agent runtime)."""
import asyncio
import time
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool

from agent4ml.backend.agents.mcp.audit import McpAuditMiddleware
from agent4ml.backend.agents.mcp.guards.credential_sanitizer import CredentialSanitizer
from agent4ml.backend.agents.mcp.registry import ToolEntry, ToolRegistry
from agent4ml.backend.agents.middlewares.tagged_context_middleware import (
    AGENT4ML_EXTERNALIZED,
    AGENT4ML_EXTERNALIZED_PATH,
)


@tool("web_search")
def web_search(query: str) -> str:
    """search"""
    return "result"


class FakeJournal:
    """简易 journal 收集事件。"""
    def __init__(self):
        self.events = []

    def append(self, event_type, payload=None):
        self.events.append({"event_type": event_type, "payload": payload or {}})


class FakeRuntime:
    """简易 runtime，含 journal。"""
    def __init__(self, journal):
        self.context = {"journal": journal}


class FakeRequest:
    """简易 ToolCallRequest。"""
    def __init__(self, tool_call, runtime):
        self.tool_call = tool_call
        self.runtime = runtime
        self.state = {}


def test_circuit_open_event():
    """熔断器 open 时不调用，记 circuit_open 事件。"""
    reg = ToolRegistry()
    reg.register(ToolEntry(tool=web_search, source="mcp", server_name="freeweb"))
    # 触发 3 次失败 → open
    for _ in range(3):
        reg.mark_unhealthy("web_search")

    journal = FakeJournal()
    middleware = McpAuditMiddleware(reg)
    runtime = FakeRuntime(journal)
    request = FakeRequest({"name": "web_search", "id": "call_1", "args": {"query": "test"}}, runtime)

    async def handler(req):
        raise AssertionError("should not call handler when circuit open")

    result = asyncio.run(middleware.awrap_tool_call(request, handler))
    assert isinstance(result, ToolMessage), f"expected ToolMessage, got {type(result)}"
    assert "circuit open" in result.content
    assert len(journal.events) == 1
    assert journal.events[0]["event_type"] == "tool.call"
    assert journal.events[0]["payload"]["status"] == "circuit_open"
    assert journal.events[0]["payload"]["duration_ms"] == 0
    print("PASS: circuit open → circuit_open event, no handler call")


def test_success_event():
    """成功调用记 ok 事件 + record_success。"""
    reg = ToolRegistry()
    entry = ToolEntry(tool=web_search, source="mcp", server_name="freeweb")
    reg.register(entry)

    journal = FakeJournal()
    middleware = McpAuditMiddleware(reg)
    runtime = FakeRuntime(journal)
    request = FakeRequest({"name": "web_search", "id": "call_2", "args": {"query": "test"}}, runtime)

    async def handler(req):
        return ToolMessage(content="search results here", tool_call_id="call_2")

    result = asyncio.run(middleware.awrap_tool_call(request, handler))
    assert isinstance(result, ToolMessage)
    assert len(journal.events) == 1
    event = journal.events[0]
    assert event["event_type"] == "tool.call"
    assert event["payload"]["status"] == "ok"
    assert event["payload"]["tool_name"] == "web_search"
    assert event["payload"]["source"] == "mcp"
    assert event["payload"]["duration_ms"] >= 0
    print("PASS: success → ok event")


def test_error_event_with_sanitization():
    """失败调用记 error 事件 + 凭证脱敏。"""
    reg = ToolRegistry()
    entry = ToolEntry(tool=web_search, source="mcp", server_name="freeweb")
    reg.register(entry)

    sanitizer = CredentialSanitizer()
    journal = FakeJournal()
    middleware = McpAuditMiddleware(reg, sanitizer)
    runtime = FakeRuntime(journal)
    request = FakeRequest({"name": "web_search", "id": "call_3", "args": {"query": "test"}}, runtime)

    async def handler(req):
        raise ValueError("auth failed for token ghp_abcdefghijklmnopqrstuvwxyz0123456789")

    try:
        asyncio.run(middleware.awrap_tool_call(request, handler))
        raise AssertionError("should re-raise")
    except ValueError:
        pass

    assert len(journal.events) == 1
    event = journal.events[0]
    assert event["event_type"] == "tool.call"
    assert event["payload"]["status"] == "error"
    assert "[REDACTED]" in event["payload"]["error"]
    assert "ghp_abc" not in event["payload"]["error"]
    print("PASS: error → sanitized error event")


def test_externalized_linkage():
    """外化触发时审计事件含 externalized=true + path。"""
    reg = ToolRegistry()
    entry = ToolEntry(tool=web_search, source="mcp", server_name="freeweb")
    reg.register(entry)

    journal = FakeJournal()
    middleware = McpAuditMiddleware(reg)
    runtime = FakeRuntime(journal)
    request = FakeRequest({"name": "web_search", "id": "call_4", "args": {"query": "big result"}}, runtime)

    async def handler(req):
        result = ToolMessage(content="preview...", tool_call_id="call_4")
        result.additional_kwargs[AGENT4ML_EXTERNALIZED] = True
        result.additional_kwargs[AGENT4ML_EXTERNALIZED_PATH] = ".agent4ml/externalized/web_search-xxx.txt"
        return result

    asyncio.run(middleware.awrap_tool_call(request, handler))
    assert len(journal.events) == 1
    event = journal.events[0]
    assert event["payload"]["externalized"] is True
    assert event["payload"]["externalized_path"] == ".agent4ml/externalized/web_search-xxx.txt"
    print("PASS: externalized linkage → externalized=true + path")


def test_no_journal_silent():
    """无 journal 时静默不报错。"""
    reg = ToolRegistry()
    reg.register(ToolEntry(tool=web_search, source="mcp", server_name="freeweb"))
    middleware = McpAuditMiddleware(reg)
    runtime = FakeRuntime(None)  # 无 journal
    request = FakeRequest({"name": "web_search", "id": "call_5", "args": {}}, runtime)

    async def handler(req):
        return ToolMessage(content="ok", tool_call_id="call_5")

    result = asyncio.run(middleware.awrap_tool_call(request, handler))
    assert isinstance(result, ToolMessage)
    print("PASS: no journal → silent, no error")


def test_unknown_source():
    """registry 无此工具 → source=unknown，仍审计。"""
    reg = ToolRegistry()
    journal = FakeJournal()
    middleware = McpAuditMiddleware(reg)
    runtime = FakeRuntime(journal)
    request = FakeRequest({"name": "nonexistent", "id": "call_6", "args": {}}, runtime)

    async def handler(req):
        return ToolMessage(content="ok", tool_call_id="call_6")

    asyncio.run(middleware.awrap_tool_call(request, handler))
    assert journal.events[0]["payload"]["source"] == "unknown"
    print("PASS: unknown tool → source=unknown, still audited")


if __name__ == "__main__":
    test_circuit_open_event()
    test_success_event()
    test_error_event_with_sanitization()
    test_externalized_linkage()
    test_no_journal_silent()
    test_unknown_source()
    print("\nAll B6 self-checks passed.")
