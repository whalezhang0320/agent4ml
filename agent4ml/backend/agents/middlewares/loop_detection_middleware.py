"""LoopDetectionMiddleware — ReAct 死循环熔断器。

default 模式无 Todo 完成度强制时，模型可能卡死循环（同工具同参数反复调），
浪费大量 token 直到 recursion_limit 耗尽。本 middleware 在 after_model 检测
重复 (tool_name, args_hash) 调用，超阈熔断：清最后一条 AIMessage 的 tool_calls
+ 注入终止引导 + jump_to model 强制模型基于已有信息收尾。

全模式挂（default + expert）。借鉴 deer-flow LoopDetectionMiddleware 思路。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, override

from langchain.agents.middleware.types import AgentMiddleware, hook_config
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.runtime import Runtime

from agent4ml.backend.agents.state.types import ThreadState


@dataclass(frozen=True)
class LoopDetectionConfig:
    """LoopDetection 配置。"""

    enabled: bool = True
    window: int = 10  # 扫描最近 N 条消息
    threshold: int = 3  # 同 (tool, args) 出现 M 次触发


def _hash_args(args: Any) -> str:
    """参数哈希。json sort_keys 截断 100 字，避免微小参数差异逃检测。"""
    try:
        return json.dumps(args, sort_keys=True, ensure_ascii=False)[:100]
    except (TypeError, ValueError):
        return str(args)[:100]


def _detect_loop(
    messages: list[Any],
    window: int = 10,
    threshold: int = 3,
) -> str | None:
    """扫描最近 window 条消息，找重复 (tool_name, args_hash)。

    返回重复的 tool_name 或 None。
    """
    recent = messages[-window:] if window > 0 else messages
    call_counts: dict[tuple[str, str], int] = {}
    for msg in recent:
        if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
            for tc in msg.tool_calls:
                if not isinstance(tc, dict):
                    continue
                name = tc.get("name", "") or ""
                args = tc.get("args", {}) or {}
                key = (name, _hash_args(args))
                call_counts[key] = call_counts.get(key, 0) + 1
    for (name, _), count in call_counts.items():
        if count >= threshold:
            return name
    return None


def _build_guidance(loop_tool: str) -> str:
    return (
        "<system_reminder>\n"
        f"检测到工具 {loop_tool} 重复调用循环。"
        "请基于已有信息给出最终答案，不要再重复调用该工具。\n"
        "</system_reminder>"
    )


class LoopDetectionMiddleware(AgentMiddleware):
    """ReAct 死循环熔断器。after_model 检测重复工具调用，超阈清 tool_calls + jump。"""

    state_schema = ThreadState  # type: ignore[assignment]

    def __init__(self, config: LoopDetectionConfig | None = None) -> None:
        self._config = config or LoopDetectionConfig()

    @hook_config(can_jump_to=["model"])
    @override
    def after_model(self, state: ThreadState, runtime: Runtime) -> dict[str, Any] | None:
        if not self._config.enabled:
            return None
        messages = state.get("messages") or []
        loop_tool = _detect_loop(messages, self._config.window, self._config.threshold)
        if loop_tool is None:
            return None

        last_ai = next((m for m in reversed(messages) if isinstance(m, AIMessage)), None)
        if not last_ai or not getattr(last_ai, "tool_calls", None):
            return None

        # 清 tool_calls 强制模型给最终答案，保留 content + 标记 loop_detected。
        # 复用原 id → add_messages 按 id 替换原消息（非追加），否则原 AIMessage 的
        # tool_calls 仍留 history，jump_to model 跳过 ToolNode 后无 ToolMessage 配对 →
        # 下一轮 model 调用 400: insufficient tool messages following tool_calls。
        # 同步剥 additional_kwargs.tool_calls/function_call 防 _has_tool_call_intent 误判。
        new_kwargs = {
            k: v for k, v in (last_ai.additional_kwargs or {}).items()
            if k not in ("tool_calls", "function_call")
        }
        new_kwargs["loop_detected"] = loop_tool
        cleared = AIMessage(
            id=last_ai.id,
            content=last_ai.content or "",
            tool_calls=[],
            additional_kwargs=new_kwargs,
        )
        guidance = HumanMessage(
            name="loop_detection",
            additional_kwargs={"hide_from_ui": True},
            content=_build_guidance(loop_tool),
        )
        return {"messages": [cleared, guidance], "jump_to": "model"}

    @hook_config(can_jump_to=["model"])
    @override
    async def aafter_model(self, state: ThreadState, runtime: Runtime) -> dict[str, Any] | None:
        return self.after_model(state, runtime)
