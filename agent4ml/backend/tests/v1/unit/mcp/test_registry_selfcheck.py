"""Batch B2+B4 self-check: registry + circuit breaker."""
from agent4ml.backend.agents.mcp import CircuitBreaker, ToolEntry, ToolRegistry
from agent4ml.backend.agents.mcp.config import McpConfig
from langchain_core.tools import tool


@tool("web_search")
def web_search(query: str) -> str:
    """search web"""
    return "result"


@tool("ddg_search")
def ddg_search(query: str) -> str:
    """ddg builtin search"""
    return "result"


def test_register_dedup():
    """builtin 优先级 > mcp，后入也能覆盖（builtin > mcp > sandbox）。"""
    config = McpConfig(core_tools=["web_search"])
    reg = ToolRegistry(config)
    e_mcp = ToolEntry(tool=web_search, source="mcp", server_name="freeweb")
    reg.register(e_mcp)
    assert reg.get("web_search").source == "mcp"
    e_builtin = ToolEntry(tool=web_search, source="builtin")
    reg.register(e_builtin)
    # builtin 优先级高 → 覆盖 mcp
    assert reg.get("web_search").source == "builtin", "builtin should override mcp (higher priority)"
    print("PASS: register dedup (builtin overrides mcp by priority)")


def test_builtin_override():
    """builtin 先入，mcp 后入 → builtin 优先级高，保留 builtin。"""
    reg = ToolRegistry()
    e_builtin = ToolEntry(tool=web_search, source="builtin")
    reg.register(e_builtin)
    assert reg.get("web_search").source == "builtin"
    e_mcp = ToolEntry(tool=web_search, source="mcp", server_name="freeweb")
    reg.register(e_mcp)
    # builtin 先入 + 优先级高 → 保留 builtin
    assert reg.get("web_search").source == "builtin"
    print("PASS: builtin override (builtin first, mcp skip)")


def test_get_tools_by_group():
    """core/sandbox/deferred 分组。"""
    config = McpConfig(core_tools=["web_search"])
    reg = ToolRegistry(config)
    reg.register(ToolEntry(tool=web_search, source="mcp", server_name="freeweb"))
    reg.register(ToolEntry(tool=ddg_search, source="builtin"))

    @tool("bash")
    def bash(cmd: str) -> str:
        """bash"""
        return "r"

    reg.register(ToolEntry(tool=bash, source="sandbox"))

    core = reg.get_tools_by_group(["core"])
    assert [t.name for t in core] == ["web_search"], f"core: {core}"

    sandbox = reg.get_tools_by_group(["sandbox"])
    assert [t.name for t in sandbox] == ["bash"], f"sandbox: {sandbox}"

    deferred = reg.get_tools_by_group(["deferred"])
    assert [t.name for t in deferred] == ["ddg_search"], f"deferred: {deferred}"

    all_tools = reg.get_tools_by_group(["core", "sandbox", "deferred"])
    assert len(all_tools) == 3
    print("PASS: get_tools_by_group (core/sandbox/deferred)")


def test_fallback_chain():
    """fallback_chains: freeweb:web_search 熔断 → builtin:ddg_search 兜底。"""
    config = McpConfig(
        fallback_chains={"web_search": ["freeweb:web_search", "builtin:ddg_search"]}
    )
    reg = ToolRegistry(config)
    reg.register(ToolEntry(tool=web_search, source="mcp", server_name="freeweb"))
    reg.register(ToolEntry(tool=ddg_search, source="builtin"))

    # freeweb healthy → 返 freeweb
    entry = reg.get_with_fallback("web_search")
    assert entry.tool.name == "web_search", f"got {entry.tool.name}"
    assert entry.source == "mcp"

    # freeweb 熔断 → builtin 兜底
    for _ in range(3):
        reg.mark_unhealthy("web_search")
    entry = reg.get_with_fallback("web_search")
    assert entry.tool.name == "ddg_search", f"fallback got {entry.tool.name}"
    assert entry.source == "builtin"
    print("PASS: fallback chain (mcp breaker open → builtin)")


def test_fallback_all_dead():
    """fallback 链全挂 → None。"""
    config = McpConfig(
        fallback_chains={"web_search": ["freeweb:web_search", "builtin:ddg_search"]}
    )
    reg = ToolRegistry(config)
    reg.register(ToolEntry(tool=web_search, source="mcp", server_name="freeweb"))
    reg.register(ToolEntry(tool=ddg_search, source="builtin"))
    for _ in range(3):
        reg.mark_unhealthy("web_search")
    for _ in range(3):
        reg.mark_unhealthy("ddg_search")
    entry = reg.get_with_fallback("web_search")
    assert entry is None, f"expected None, got {entry}"
    print("PASS: fallback all dead → None")


def test_circuit_breaker_state_machine():
    """closed→open→half_open→closed（探针成功）。"""
    cb = CircuitBreaker()
    assert cb.state == "closed"
    assert cb.allow_call() is True

    cb.record_failure()
    cb.record_failure()
    assert cb.state == "closed", "2 failures still closed"
    assert cb.allow_call() is True

    cb.record_failure()
    assert cb.state == "open", "3 failures → open"
    assert cb.allow_call() is False, "open should reject"

    # 模拟 cooldown 过（直接改 opened_at）
    import time
    cb.opened_at = time.time() - 61
    assert cb.allow_call() is True, "cooldown passed probe"
    assert cb.state == "half_open"

    cb.record_success()
    assert cb.state == "closed", "probe success → closed"
    assert cb.failure_count == 0
    print("PASS: circuit breaker closed→open→half_open→closed")


def test_circuit_breaker_half_open_failure():
    """half_open 探针失败 → 重置 open。"""
    cb = CircuitBreaker()
    for _ in range(3):
        cb.record_failure()
    assert cb.state == "open"
    import time
    cb.opened_at = time.time() - 61
    cb.allow_call()  # → half_open
    assert cb.state == "half_open"
    cb.record_failure()
    assert cb.state == "open", "half_open failure → reset open"
    print("PASS: circuit breaker half_open failure → open")


def test_mark_healthy_unhealthy():
    """mark_unhealthy/mark_healthy 触发 breaker。"""
    reg = ToolRegistry()
    reg.register(ToolEntry(tool=web_search, source="mcp", server_name="freeweb"))
    reg.mark_unhealthy("web_search")
    reg.mark_unhealthy("web_search")
    reg.mark_healthy("web_search")
    entry = reg.get("web_search")
    assert entry.breaker.state == "closed", f"after reset: {entry.breaker.state}"
    assert entry.breaker.failure_count == 0
    print("PASS: mark_unhealthy/mark_healthy")


def test_get_metadata():
    """get_metadata 返回工具元数据。"""
    reg = ToolRegistry()
    reg.register(ToolEntry(
        tool=web_search, source="mcp", server_name="freeweb",
        metadata={"typical_output_tokens": 800},
    ))
    meta = reg.get_metadata("web_search")
    assert meta == {"typical_output_tokens": 800}
    assert reg.get_metadata("nonexistent") is None
    print("PASS: get_metadata")


if __name__ == "__main__":
    test_register_dedup()
    test_builtin_override()
    test_get_tools_by_group()
    test_fallback_chain()
    test_fallback_all_dead()
    test_circuit_breaker_state_machine()
    test_circuit_breaker_half_open_failure()
    test_mark_healthy_unhealthy()
    test_get_metadata()
    print("\nAll B2+B4 self-checks passed.")
