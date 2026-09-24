"""PersonaPolicy Protocol — 用户画像策略契约。

承接 `Hezao-MemDesign-Docs/agent4ml/00-long-term-memory-foundation.md` §6.2
+ `48-memory-l1-base-layer.md` §4 Step 5.6。

默认实现：StaticDynamicPersona（strategies/default/persona.py，Layer 6）。
借鉴 supermemory profile.static + dynamic 二分。
可替换：FlatPersona / NoPersona。

INVARIANT: Protocol 纯契约零实现，Layer 1 仅定义接口，Layer 6 填实现。
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class PersonaPolicy(Protocol):
    """用户画像策略协议（00 §6.2 supermemory profile.static+dynamic）。

    默认实现 StaticDynamicPersona（strategies/default/persona.py，Layer 6）。
    可替换 FlatPersona / NoPersona。
    """

    def get_static_profile(self, user_id: str) -> dict:
        """稳定事实（偏好 / 长期目标）。"""
        ...

    def get_dynamic_profile(self, user_id: str) -> dict:
        """近期活动（当前任务 / 最近对话）。"""
        ...

    def update_profile(self, user_id: str, facts: dict) -> None:
        """更新画像（从 semantic 记忆归纳）。"""
        ...
