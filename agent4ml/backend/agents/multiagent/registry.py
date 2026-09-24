"""SpecialistRegistry — specialist 注册发现（非全局单例）。

设计（spec.md SpecialistRegistry Requirement + boundary.md §3.3）:
- 不是全局单例：作为 CapabilityRegistry 的 capability 注入（Batch 3）
- register + get + list_specialists + register_from_config
- get 缺失抛 SpecialistNotFoundError
- list_specialists 按能力过滤（SpecialistCapability）
- duck typing：接受任何实现 SpecialistAgent Protocol 的对象，不需显式继承
"""
from __future__ import annotations

from agent4ml.backend.agents.multiagent.exceptions import SpecialistNotFoundError
from agent4ml.backend.agents.multiagent.specialist import SpecialistAgent
from agent4ml.backend.agents.multiagent.types import SpecialistCapability


class SpecialistRegistry:
    """specialist 注册发现（内部可变 dict，非 frozen）。

    作为 CapabilityRegistry.specialist_registry 注入（Batch 3）。
    bootstrap 装配时 register_from_config 反射加载（Batch 10）。
    """

    def __init__(self) -> None:
        self._specialists: dict[str, SpecialistAgent] = {}

    def register(self, specialist: SpecialistAgent) -> str:
        """注册 specialist，返 specialist.name。同名覆盖（last-write-wins）。"""
        name = specialist.name
        self._specialists[name] = specialist
        return name

    def get(self, name: str) -> SpecialistAgent:
        """获取 specialist。缺失抛 SpecialistNotFoundError。"""
        try:
            return self._specialists[name]
        except KeyError:
            raise SpecialistNotFoundError(name)

    def list_specialists(
        self,
        capability: SpecialistCapability | None = None,
    ) -> list[str]:
        """列出 specialist name。按能力过滤（None 返全部）。"""
        names = list(self._specialists.keys())
        if capability is None:
            return names
        return [
            name
            for name in names
            if self._specialists[name].capabilities.has(capability)
        ]

    def register_from_config(self, use_list: list[str]) -> None:
        """反射加载 specialist（bootstrap 装配时调用）。

        Batch 10 bootstrap 实现：按 use_list 反射 import specialist 类 +
        凭证检测 + disabled 不注册。此处留接口契约。
        """
        raise NotImplementedError(
            "register_from_config implemented in bootstrap (Batch 10)"
        )

    def __len__(self) -> int:
        return len(self._specialists)

    def __contains__(self, name: object) -> bool:
        return name in self._specialists
