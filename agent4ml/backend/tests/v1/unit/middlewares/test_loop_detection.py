from langchain_core.messages import AIMessage, HumanMessage

from agent4ml.backend.agents.middlewares.loop_detection_middleware import (
    LoopDetectionConfig,
    LoopDetectionMiddleware,
    _detect_loop,
    _hash_args,
)


def _ai_with_tool(name: str, args: dict, content: str = "") -> AIMessage:
    return AIMessage(
        content=content,
        tool_calls=[{"name": name, "args": args, "id": f"call-{name}-{args}", "type": "tool_call"}],
    )


def test_hash_args_stable() -> None:
    assert _hash_args({"a": 1, "b": 2}) == _hash_args({"b": 2, "a": 1})


def test_hash_args_truncates_long() -> None:
    long_args = {"q": "x" * 200}
    h = _hash_args(long_args)
    assert len(h) <= 100


def test_detect_loop_returns_none_when_no_repeats() -> None:
    msgs = [
        _ai_with_tool("web_search", {"query": "a"}),
        _ai_with_tool("web_search", {"query": "b"}),
    ]
    assert _detect_loop(msgs, window=10, threshold=3) is None


def test_detect_loop_returns_tool_when_threshold_hit() -> None:
    msgs = [
        _ai_with_tool("web_search", {"query": "北京天气"}),
        _ai_with_tool("web_search", {"query": "北京天气"}),
        _ai_with_tool("web_search", {"query": "北京天气"}),
    ]
    assert _detect_loop(msgs, window=10, threshold=3) == "web_search"


def test_detect_loop_different_args_not_triggered() -> None:
    msgs = [
        _ai_with_tool("web_search", {"query": "北京"}),
        _ai_with_tool("web_search", {"query": "上海"}),
        _ai_with_tool("web_search", {"query": "广州"}),
    ]
    assert _detect_loop(msgs, window=10, threshold=3) is None


def test_detect_loop_window_limits_scan() -> None:
    # 5 条历史 + window=3 只扫最后 3 条，早期重复不计
    msgs = [
        _ai_with_tool("web_search", {"query": "x"}),
        _ai_with_tool("web_search", {"query": "x"}),
        _ai_with_tool("browse", {"url": "y"}),
        _ai_with_tool("browse", {"url": "y"}),
    ]
    # window=4 看到全部 4 条，web_search x2 + browse x2，threshold=3 都不触发
    assert _detect_loop(msgs, window=4, threshold=3) is None


def test_middleware_disabled_returns_none() -> None:
    mw = LoopDetectionMiddleware(LoopDetectionConfig(enabled=False))
    state = {"messages": [_ai_with_tool("web_search", {"q": "x"})] * 3}
    assert mw.after_model(state, runtime=None) is None


def test_middleware_no_loop_returns_none() -> None:
    mw = LoopDetectionMiddleware()
    state = {"messages": [_ai_with_tool("web_search", {"q": "a"}), _ai_with_tool("web_search", {"q": "b"})]}
    assert mw.after_model(state, runtime=None) is None


def test_middleware_loop_clears_tool_calls_and_jumps() -> None:
    mw = LoopDetectionMiddleware()
    state = {
        "messages": [
            _ai_with_tool("web_search", {"query": "x"}),
            _ai_with_tool("web_search", {"query": "x"}),
            _ai_with_tool("web_search", {"query": "x"}),
        ]
    }
    result = mw.after_model(state, runtime=None)
    assert result is not None
    assert result.get("jump_to") == "model"
    msgs = result.get("messages", [])
    assert len(msgs) == 2
    # 第一条是清空 tool_calls 的 AIMessage
    cleared = msgs[0]
    assert isinstance(cleared, AIMessage)
    assert cleared.tool_calls == []
    assert cleared.additional_kwargs.get("loop_detected") == "web_search"
    # 第二条是引导 HumanMessage
    assert isinstance(msgs[1], HumanMessage)
    assert msgs[1].name == "loop_detection"
    assert "web_search" in msgs[1].content


def test_middleware_loop_reuses_id_for_pairing() -> None:
    """Regression: cleared AIMessage must reuse original id so add_messages
    replaces (not appends). Otherwise the original tool_calls-bearing AIMessage
    stays in history; jump_to model skips ToolNode → no ToolMessage → 400
    'insufficient tool messages following tool_calls' on next model call."""
    mw = LoopDetectionMiddleware()
    last_ai = AIMessage(
        id="ai-original-id",
        content="",
        tool_calls=[{"name": "web_search", "args": {"query": "x"}, "id": "tc1", "type": "tool_call"}],
        additional_kwargs={"tool_calls": [{"name": "web_search", "args": {"query": "x"}, "id": "tc1"}]},
    )
    state = {"messages": [_ai_with_tool("web_search", {"query": "x"}), _ai_with_tool("web_search", {"query": "x"}), last_ai]}
    result = mw.after_model(state, runtime=None)
    assert result is not None
    cleared = result["messages"][0]
    assert isinstance(cleared, AIMessage)
    # id 复用 → add_messages 替换原消息，非追加
    assert cleared.id == "ai-original-id"
    # additional_kwargs 里陈旧 tool_calls 必须剥掉，防 _has_tool_call_intent 误判
    assert "tool_calls" not in (cleared.additional_kwargs or {})
    assert cleared.tool_calls == []
    assert cleared.additional_kwargs.get("loop_detected") == "web_search"


def test_middleware_no_tool_calls_on_last_ai_returns_none() -> None:
    mw = LoopDetectionMiddleware()
    # 最后一条 AIMessage 无 tool_calls（模型想退出），即使历史有重复也不熔断
    state = {
        "messages": [
            _ai_with_tool("web_search", {"q": "x"}),
            _ai_with_tool("web_search", {"q": "x"}),
            _ai_with_tool("web_search", {"q": "x"}),
            AIMessage(content="最终答案", tool_calls=[]),
        ]
    }
    assert mw.after_model(state, runtime=None) is None
