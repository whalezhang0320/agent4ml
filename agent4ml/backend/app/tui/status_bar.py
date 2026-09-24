"""StatusBar — 底部细状态行 Widget。

左侧：运行中显示 ``<spinner> running  esc interrupt``——spinner 用 rich.spinner.Spinner
（Braille dots），运行时 auto_refresh=0.1 驱动 10fps 旋转，给用户"agent 在动、
没卡死"的视觉反馈；空闲时 auto_refresh=0.0，spinner 不刷新不耗 CPU。
右侧：``{tokens} ({pct}%)   ctrl+p commands``。
模型/模式信息移到 InputBox 的 Build 信息行，此处仅保留运行态与用量提示，色系统一。
背景透明（不叠灰底蒙版），直接落在 Screen 底色上，和 InputBox 的实体框区分开——
只有输入框本身是"盒子"，状态提示是裸露在盒子外的纯文字，贴合参考设计。
"""

from __future__ import annotations

import time
from typing import Any

from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text
from textual.widget import Widget

from agent4ml.backend.app.tui import theme


def _format_tokens(tokens: int) -> str:
    if tokens >= 1_000_000:
        return f"{tokens / 1_000_000:.1f}M"
    if tokens >= 1_000:
        return f"{tokens / 1_000:.1f}K"
    return str(tokens)


class StatusBar(Widget):
    """底部细状态行——左运行态（含 spinner 动画）、右用量与命令提示，透明背景无蒙版。"""

    DEFAULT_CSS = f"""
    StatusBar {{
        height: 1;
        background: transparent;
        color: {theme.TEXT_SECONDARY};
        padding: 0 2;
        margin: 0 2 0 2;
    }}
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._cli_state: dict[str, Any] = {}
        # rich.spinner.Spinner：render(time.time()) 返回当前帧 Text。
        # 不传 text= 参数——spinner 单字符在前，" running" 文字在 render() 里
        # 手工 append，保证顺序是 ``⠋ running``（rich 默认 text 在 spinner 前，
        # 会变成 ``running ⠋``，不符合"spinner 引导 + 文字跟随"的视觉直觉）。
        self._spinner = Spinner("dots", style=theme.ACCENT_ASSISTANT)

    def _on_mount(self, event: Any) -> None:
        # 空闲时 auto_refresh=None 完全不起 timer（不耗 CPU）；运行时由 update_state 置 0.1。
        # 注意：Textual 的 auto_refresh setter 在 interval 非 None 时会 set_interval，
        # 传 0.0 会创建 interval=0 的 timer → 关闭时 ZeroDivisionError。off 状态必须用 None。
        self.auto_refresh = None

    def update_state(self, cli_state: dict[str, Any]) -> None:
        """更新展示状态——同时根据 _running 切换 auto_refresh 启停 spinner 动画。"""
        self._cli_state = dict(cli_state)
        running = bool(cli_state.get("_running", False))
        # 0.1s = 10fps：Braille dots 10 帧循环刚好 1 秒一圈，平滑且低开销。
        # None = 不起 timer（idle 完全静默）；0.0 会触发 Textual 的 ZeroDivisionError，禁用。
        self.auto_refresh = 0.1 if running else None
        # 非 running 时 auto_refresh=None 不再触发自动刷新，需手工 refresh 一次把
        # spinner 落幕后的静态状态画出来（如刚结束 run 后清掉左侧动效）
        self.refresh()

    def render(self) -> Any:
        cli_state = self._cli_state
        tokens = cli_state.get("current_tokens", 0)
        fraction = cli_state.get("current_fraction", 0.0)
        running = bool(cli_state.get("_running", False))
        pct = fraction * 100.0
        pct_color = theme.ACCENT_WARN if pct >= 90 else theme.TEXT_SECONDARY

        left = Text()
        if running:
            # spinner 当前帧（单字符 + ACCENT_ASSISTANT 色）+ " running" 同色
            frame = self._spinner.render(time.time())
            left.append_text(frame)
            left.append(" running", style=theme.ACCENT_ASSISTANT)
            left.append("  esc interrupt", style=theme.TEXT_DIM)

        right = Text()
        if tokens:
            right.append(f"{_format_tokens(tokens)} ", style=theme.TEXT_SECONDARY)
            right.append(f"({pct:.1f}%)", style=pct_color)
            right.append("     ", style="")
        right.append("ctrl+p ", style=theme.TEXT_DIM)
        right.append("commands", style=theme.TEXT_SECONDARY)

        grid = Table.grid(expand=True)
        grid.add_column(justify="left", ratio=1)
        grid.add_column(justify="right")
        grid.add_row(left, right)
        return grid
