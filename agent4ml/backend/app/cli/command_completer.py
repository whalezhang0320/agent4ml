"""command_completer — ``/`` 命令模糊补全菜单 + ``/skill <name>`` skill 名补全。

消费 ``CommandRegistry.list_all()``，仅在输入以 ``/`` 开头时触发命令名补全；
``/skill `` 后参数词补全子命令（list/off/enable/disable/install）优先，无子命令命中再补
``skill_provider()`` 返回的 active skill 名（Claude Code 式选择窗）。

选中行/未选中行配色由 ``main.py`` 的 ``PromptSession`` ``Style`` 配置。
"""
from __future__ import annotations

from typing import Callable

from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.document import Document

from agent4ml.backend.app.cli.registry import CommandRegistry

_SKILL_SUBCOMMANDS = ("list", "off", "enable", "disable", "install")


class SlashCommandCompleter(Completer):
    """``/`` 命令补全 + ``/skill <name>`` skill 名补全。

    - word 以 ``/`` 开头 → 命令名子串补全
    - ``/skill `` 后参数词 → 子命令前缀补全（优先）；无子命令命中 → skill 名前缀补全
    - skill_provider 默认 None（无 skill 名补全，兼容既有）
    """

    def __init__(
        self,
        registry: CommandRegistry,
        skill_provider: Callable[[], list[str]] | None = None,
    ) -> None:
        self._registry = registry
        self._skill_provider = skill_provider

    def get_completions(self, document: Document, complete_event):  # type: ignore[no-untyped-def]
        word = document.get_word_before_cursor(WORD=True)

        # 命令名补全：仅 / 开头
        if word.startswith("/"):
            word_lower = word.lower()
            for spec in self._registry.list_all():
                if word_lower in spec.name.lower():
                    yield Completion(
                        text=spec.name,
                        start_position=-len(word),
                        display=spec.name,
                        display_meta=spec.description,
                    )
            return

        # /skill <arg> 补全：行以 /skill + 空白 开头，cursor 在参数位
        stripped = document.text_before_cursor.lstrip()
        if stripped.startswith("/skill") and len(stripped) > 6 and stripped[6] in (" ", "\t"):
            arg_word = word
            arg_lower = arg_word.lower()
            # 子命令优先：有命中则只补子命令，不补 skill 名（避免 enable 被 skill 名遮蔽）
            sub_matches = [s for s in _SKILL_SUBCOMMANDS if s.startswith(arg_lower)]
            if sub_matches:
                for s in sub_matches:
                    yield Completion(
                        text=s,
                        start_position=-len(arg_word),
                        display=s,
                        display_meta="subcommand",
                    )
                return
            # 无子命令命中 → 补 active skill 名
            if self._skill_provider is not None:
                try:
                    names = self._skill_provider() or []
                except Exception:
                    names = []
                for n in names:
                    if n.lower().startswith(arg_lower):
                        yield Completion(
                            text=n,
                            start_position=-len(arg_word),
                            display=n,
                            display_meta="skill",
                        )
