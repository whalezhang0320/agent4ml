"""registry — CLI 命令元数据统一注册表。

把 ``commands.py`` 原本硬编码的 ``handlers = {"/help": ...}`` dict 升级为
``CommandRegistry``，让 ``_cmd_help`` 文案和 ``/`` 补全菜单（``SlashCommandCompleter``）
共用同一份 ``CommandSpec.description``，避免两处维护。

同时预留 ``register_skill(name, description, handler)`` 接口——与
``agents/prompts/system/leader/extensions.md`` 里"后续 skills… 在此注入"的占位概念打通。
本期该方法**不被任何业务代码调用**（Agent4ML 无 CLI 侧 skill loader），仅接口先行焊好，
等 skill 系统落地时在 bootstrap 阶段遍历调 ``register_skill`` 即可，UI 层零改动。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal


@dataclass
class CommandSpec:
    """单条 `/` 命令的元数据。

    Attributes:
        name: 命令名（含 `/` 前缀，如 ``/help``）。
        description: 一句话描述，供 ``/help`` 文案与补全菜单 ``display_meta`` 共用。
        handler: 处理函数，签名与原 ``commands.py`` 里 ``_cmd_*`` 一致；
            返回 ``True`` 表示退出 CLI，``False`` 或 ``None`` 表示继续循环。
        source: ``"builtin"``（硬编码命令）或 ``"skill"``（未来 skill loader 注册）。
            本期仅用于数据层标记，不在 UI 上区分。
    """

    name: str
    description: str
    handler: Callable[..., Any]
    source: Literal["builtin", "skill"] = "builtin"


class CommandRegistry:
    """命令注册表——保序存储 ``CommandSpec``，供补全菜单与 ``/help`` 共用。

    注册顺序即 ``list_all()`` 返回顺序，``_cmd_help`` 输出顺序由此决定。
    """

    def __init__(self) -> None:
        self._specs: list[CommandSpec] = []
        self._by_name: dict[str, CommandSpec] = {}

    def register(self, spec: CommandSpec) -> None:
        """注册一条命令（builtin 或 skill）。同名命令后注册者覆盖前者。"""
        self._by_name[spec.name] = spec
        # 保序：若 name 已存在则替换原位，否则追加
        existing_idx = next(
            (i for i, s in enumerate(self._specs) if s.name == spec.name), None
        )
        if existing_idx is not None:
            self._specs[existing_idx] = spec
        else:
            self._specs.append(spec)

    def register_skill(self, name: str, description: str, handler: Callable[..., Any]) -> None:
        """预留接口：未来 skill loader 调此方法注册 skill 命令。

        等价于 ``register(CommandSpec(name, description, handler, source="skill"))``。
        本期无调用方——Agent4ML 还没有 CLI 侧 skill 发现/加载机制，做假数据会误导用户。
        """
        self.register(CommandSpec(name=name, description=description, handler=handler, source="skill"))

    def list_all(self) -> list[CommandSpec]:
        """返回全部已注册命令（按注册顺序）。补全菜单与 ``/help`` 共用此数据源。"""
        return list(self._specs)

    def get(self, name: str) -> CommandSpec | None:
        """按命令名查询；未命中返回 ``None``。``handle_command`` 分发用。"""
        return self._by_name.get(name)
