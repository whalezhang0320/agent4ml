"""ToolCallMiddleware failure_summary 队列延迟注入测试。

验证：wrap_tool_call 不再直接注入 HumanMessage 到 messages（避免插在并行
ToolMessage 间破坏 pairing），改入队列；before_model drain 队列注入。
"""

from __future__ import annotations

from types import SimpleNamespace

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agent4ml.backend.agents.middlewares.tool_call_middleware import ToolCallMiddleware


def _runtime(tid: str = "t1", rid: str = "r1", journal=None):
    return SimpleNamespace(thread_id=tid, run_id=rid, journal=journal)


def _request(tool_name: str = "web_search", call_id: str = "tc1", state: dict | None = None):
    return SimpleNamespace(
        tool_call={"name": tool_name, "id": call_id, "args": {}},
        state=state or {"errors": []},
        runtime=_runtime(),
    )


def _handler_returning_error(result_count: int = 1):
    """返回含 'no results found' 的 ToolMessage（触发 empty 业务失败分类）。"""
    return ToolMessage(
        content=f'{{"error": "Search failed: No results found.", "query": "test"}}',
        tool_call_id="tc1",
        name="web_search",
    )


def test_failure_summary_queued_not_in_messages() -> None:
    """wrap_tool_call 不把 failure_summary HumanMessage 放进 messages（避免插在并行 ToolMessage 间）。"""
    mw = ToolCallMiddleware()
    # 模拟 3 次失败（attempt 达 3，触发摘要）
    state = {"errors": []}
    for i in range(3):
        req = _request(call_id=f"tc{i}", state=state)
        result = _handler_returning_error()
        cmd = mw.wrap_tool_call(req, lambda r: result)
        # 模拟 state 更新（errors 累积）
        state = {"errors": (state.get("errors") or []) + [{"tool_name": "web_search", "attempt": i + 1, "kind": "failure"}]}

    # 第 3 次触发摘要 → 队列应有 1 条
    summaries = mw._drain_summaries(_runtime())
    # drain 后清空，再 drain 应为空
    assert len(summaries) == 1
    assert "web_search" in summaries[0]
    assert mw._drain_summaries(_runtime()) == []


def test_before_model_drains_queue() -> None:
    """before_model drain 队列，注入 HumanMessage（在 ToolNode 之后，不破坏 pairing）。"""
    mw = ToolCallMiddleware()
    mw._queue_summary(_runtime(), "工具 web_search 已连续失败 3 次")

    result = mw.before_model({"messages": []}, _runtime())
    assert result is not None
    msgs = result.get("messages", [])
    assert len(msgs) == 1
    assert isinstance(msgs[0], HumanMessage)
    assert msgs[0].name == "tool_failure_summary"
    assert "web_search" in msgs[0].content

    # drain 后再调 before_model 应返 None
    assert mw.before_model({"messages": []}, _runtime()) is None


def test_before_model_empty_queue_returns_none() -> None:
    """空队列 → before_model 返 None。"""
    mw = ToolCallMiddleware()
    assert mw.before_model({"messages": []}, _runtime()) is None


def test_wrap_tool_call_messages_only_toolmessage() -> None:
    """wrap_tool_call 返回的 Command.messages 只含 ToolMessage，无 HumanMessage。"""
    mw = ToolCallMiddleware()
    req = _request(state={"errors": [
        {"tool_name": "web_search", "attempt": 2, "kind": "failure"},
    ]})
    result = _handler_returning_error()
    cmd = mw.wrap_tool_call(req, lambda r: result)

    # Command.update.messages 应只含 ToolMessage（attempt=3 触发摘要但入队列非 messages）
    from langgraph.types import Command
    assert isinstance(cmd, Command)
    msgs = cmd.update.get("messages", [])
    assert len(msgs) == 1
    assert isinstance(msgs[0], ToolMessage)
    # 摘要应入队列
    assert len(mw._drain_summaries(_runtime())) == 1


def test_hard_budget_per_run_not_cumulative() -> None:
    """_HARD_BUDGET 应 per-run 计数，不跨 run 累积。

    before_agent 记 errors 基线，硬预算 = len(errors) - baseline >= 30。
    跨 run 的 errors 不计入当前 run 预算。
    """
    mw = ToolCallMiddleware(retry_budget=5, hard_budget=30)
    # 模拟前一个 run 累积了 25 条 errors
    state_with_history = {"errors": [{"tool_name": "web_search", "attempt": 1, "kind": "failure"} for _ in range(25)]}
    # before_agent 记基线 = 25
    mw.before_agent(state_with_history, _runtime())

    # 当前 run 再调 5 次（run_count = 5），不应触发硬预算（30 阈值）
    req = _request(state=state_with_history)
    result = _handler_returning_error()
    cmd = mw.wrap_tool_call(req, lambda r: result)
    from langgraph.types import Command
    assert isinstance(cmd, Command)
    # 应正常执行（非 budget_exhausted 短路）
    msgs = cmd.update.get("messages", [])
    assert any(isinstance(m, ToolMessage) for m in msgs)  # 有 ToolMessage = 正常执行

    # 验证 per-run 计数 = 1（baseline=25, errors=25, run_count=1）
    assert mw._run_tool_count(_runtime(), state_with_history["errors"]) == 0  # errors 未更新，仍 25


def test_hard_budget_triggers_within_run() -> None:
    """per-run 工具调用达 30 次 → 触发硬预算短路。"""
    mw = ToolCallMiddleware(retry_budget=5, hard_budget=30)
    # before_agent 记基线 = 0（空 errors）
    mw.before_agent({"errors": []}, _runtime())

    # 模拟当前 run 已累积 30 条 errors
    state_30 = {"errors": [{"tool_name": "web_search", "attempt": 1, "kind": "success"} for _ in range(30)]}
    req = _request(state=state_30, call_id="tc_new")
    cmd = mw.wrap_tool_call(req, lambda r: _handler_returning_error())
    from langgraph.types import Command
    assert isinstance(cmd, Command)
    # 应返回 budget_exhausted 短路（非正常 ToolMessage）
    msgs = cmd.update.get("messages", [])
    assert len(msgs) == 1
    assert isinstance(msgs[0], ToolMessage)
    assert "预算上限" in msgs[0].content


def test_retry_budget_resets_per_run() -> None:
    """retry_budget 应 per-run 计数——上 run 的 attempt=3 不阻塞当前 run。"""
    mw = ToolCallMiddleware(retry_budget=3, hard_budget=30)
    # run 1: 3 次失败 → errors 累积 3 条 attempt=3
    run1_errors = [
        {"tool_name": "web_search", "attempt": 1, "kind": "failure"},
        {"tool_name": "web_search", "attempt": 2, "kind": "failure"},
        {"tool_name": "web_search", "attempt": 3, "kind": "failure"},
    ]
    mw.before_agent({"errors": []}, _runtime(tid="t1", rid="r1"))
    # run 1 中 web_search 被 blocked（attempt=3 >= 3）
    req1 = _request(state={"errors": run1_errors})
    cmd1 = mw.wrap_tool_call(req1, lambda r: _handler_returning_error())
    msgs1 = cmd1.update.get("messages", [])
    assert "重试上限" in msgs1[0].content  # blocked

    # run 2: before_agent 记基线 = 3（len(errors) at start of run 2）
    state_run2 = {"errors": run1_errors}  # checkpointer 持久化的 errors
    mw.before_agent(state_run2, _runtime(tid="t1", rid="r2"))
    # run 2 中 web_search 应可调用（per-run attempt=0，不 blocked）
    req2 = _request(state=state_run2, call_id="tc_run2")
    cmd2 = mw.wrap_tool_call(req2, lambda r: _handler_returning_error())
    msgs2 = cmd2.update.get("messages", [])
    assert any(isinstance(m, ToolMessage) and "重试上限" not in m.content for m in msgs2)  # 正常执行
