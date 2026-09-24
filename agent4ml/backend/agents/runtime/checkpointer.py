"""Checkpointer 工厂 — LangGraph graph 状态持久化单例。

借鉴 deer-flow runtime/checkpointer/provider.py。MVP 用 InMemorySaver（进程内），
进阶可升 SqliteSaver（config 驱动，接口不变）。

checkpointer 是 create_agent(checkpointer=...) 编译参数，LangGraph 内部处理持久化，
无 hook。thread_id 在 config.configurable 透传，自动按 thread 存取 state。
"""

from __future__ import annotations

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Checkpointer

_cp: Checkpointer | None = None


def get_checkpointer() -> Checkpointer:
    """返回全局 checkpointer 单例。MVP: InMemorySaver（进程内，非持久化）。

    首次调用创建 InMemorySaver，后续返回同一实例。模式切换重建 graph 时
    复用同一单例 + 同一 thread_id，使 state 跨模式累积保留。
    """
    global _cp
    if _cp is None:
        _cp = InMemorySaver()
    return _cp


def reset_checkpointer() -> None:
    """重置单例，强制下次 get_checkpointer() 创建新实例。测试用。"""
    global _cp
    _cp = None
