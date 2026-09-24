"""MessageNormalizerMiddleware — wrap_model_call 合并多 SystemMessage 为单条 leading（公共固定）。

strict backend（vLLM/Qwen/Anthropic）拒绝非 leading SystemMessage。公共需求，
固定 real 不经 registry。noop=provider 报错（功能坏）。仅动 request payload，
不动 checkpoint state，保 history scanner 工作。
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import override

from langchain.agents.middleware.types import (
    AgentMiddleware,
    ModelCallResult,
    ModelRequest,
    ModelResponse,
)
from langchain_core.messages import SystemMessage

from agent4ml.backend.agents.state.types import ThreadState


def _flatten_content(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and "text" in item:
                parts.append(item["text"])
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(content)


class MessageNormalizerMiddleware(AgentMiddleware):
    """wrap_model_call 合并 SystemMessage 为单条 leading（公共固定）。"""

    state_schema = ThreadState  # type: ignore[assignment]

    @staticmethod
    def _coalesce(request: ModelRequest) -> ModelRequest | None:
        in_msg_systems = [m for m in request.messages if isinstance(m, SystemMessage)]
        if not in_msg_systems:
            return None
        parts: list[SystemMessage] = []
        if request.system_message is not None:
            parts.append(request.system_message)
        parts.extend(in_msg_systems)
        first = parts[0]
        merged_kwargs: dict = {}
        for p in parts:
            merged_kwargs.update(p.additional_kwargs or {})
        merged = SystemMessage(
            content="\n\n".join(_flatten_content(p.content) for p in parts),
            id=first.id,
            additional_kwargs=merged_kwargs,
        )
        non_system = [m for m in request.messages if not isinstance(m, SystemMessage)]
        return request.override(system_message=merged, messages=non_system)

    @override
    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelCallResult:
        coalesced = self._coalesce(request)
        return handler(coalesced if coalesced is not None else request)

    @override
    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        coalesced = self._coalesce(request)
        return await handler(coalesced if coalesced is not None else request)
