"""CommandPalette — Ctrl+P 命令面板（占位 UI，暂不接实际执行逻辑）。

样式对齐参考设计：居中浮层 + 顶部 ``Commands``/``esc`` 标题行 + 搜索输入行 +
分组列表（左侧命令名，右侧对应的 / 斜杠命令提示，靠右对齐）。

当前只做视觉展示与 esc/ctrl+p 关闭交互，列表项本身不绑定执行——真正的命令
派发已经有 ``/xxx`` 输入通道（``app.py._handle_command``），这里先把 UI 结构
预留出来，后续要把面板项接上真实 handler 时再补。
"""

from __future__ import annotations

from textual import events
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Input, Static

from rich.text import Text

from agent4ml.backend.app.tui import theme

# 分组列表：(分组标题, [(命令名, 提示文案), ...])。提示文案用 Agent4ML 已有的
# / 斜杠命令占位（真实可用命令见 app/cli/commands.py），不是键位绑定。
_GROUPS: list[tuple[str, list[tuple[str, str]]]] = [
    ("Suggested", [
        ("Expand last round", "/expand"),
        ("Switch to expert mode", "/expert"),
        ("Switch to default mode", "/default"),
    ]),
    ("Session", [
        ("Show model routing", "/model"),
        ("Thread info", "/thread"),
        ("Toggle thinking display", "/thinking"),
        ("List available tools", "/tools"),
        ("Generate report", "/report"),
        ("Prompt manager", "/prompt"),
        ("Clear screen", "/clear"),
        ("Exit", "/exit"),
    ]),
]


class CommandPalette(ModalScreen[None]):
    """Ctrl+P 弹出的命令面板浮层。"""

    BINDINGS = [
        Binding("escape", "dismiss_palette", "Close", show=False),
        Binding("ctrl+p", "dismiss_palette", "Close", show=False),
    ]

    DEFAULT_CSS = f"""
    CommandPalette {{
        align: center middle;
        background: $background 0%;
    }}
    CommandPalette > #palette-box {{
        width: 70;
        max-width: 90%;
        height: auto;
        max-height: 80%;
        background: {theme.SURFACE};
        border: solid {theme.BORDER};
        padding: 1 2;
    }}
    CommandPalette #palette-title {{
        height: 1;
        margin: 0 0 1 0;
    }}
    CommandPalette #palette-title-right {{
        width: auto;
        height: 1;
        border: solid {theme.BORDER};
        padding: 0 1;
        color: {theme.TEXT_SECONDARY};
    }}
    CommandPalette #palette-search {{
        height: 1;
        border: none;
        background: {theme.SURFACE_ALT};
        color: {theme.TEXT_PRIMARY};
        padding: 0 1;
        margin: 0 0 1 0;
    }}
    CommandPalette #palette-search:focus {{
        border: none;
        background: {theme.SURFACE_ALT};
    }}
    CommandPalette .palette-group-title {{
        height: 1;
        color: {theme.ACCENT_BRAND};
        margin: 1 0 0 0;
    }}
    CommandPalette .palette-row {{
        height: 1;
    }}
    """

    def compose(self) -> ComposeResult:
        with Container(id="palette-box"):
            with Horizontal(id="palette-title"):
                yield Static("Commands", id="palette-title-left")
                yield Static("esc", id="palette-title-right")
            yield Input(placeholder="Search", id="palette-search")
            with Vertical(id="palette-list"):
                for group_name, items in _GROUPS:
                    yield Static(group_name, classes="palette-group-title")
                    for label, hint in items:
                        yield Static(self._row_text(label, hint), classes="palette-row")

    def _row_text(self, label: str, hint: str) -> Text:
        pad = max(1, 50 - len(label))
        t = Text()
        t.append(label, style=theme.TEXT_PRIMARY)
        t.append(" " * pad)
        t.append(hint, style=theme.TEXT_DIM)
        return t

    def on_mount(self) -> None:
        self.query_one("#palette-title-left", Static).styles.color = theme.TEXT_PRIMARY
        self.query_one("#palette-search", Input).focus()

    def on_click(self, event: events.Click) -> None:
        """点击面板卡片以外的区域（浮层背景）即关闭——鼠标 Click 事件从叶子
        widget 一路冒泡到 Screen；只有点在卡片外的空白处时 event.widget 才
        会是 Screen 自身（卡片内任何 widget 都会先截获），据此区分"点外部"。
        """
        if event.widget is self:
            self.dismiss(None)

    def action_dismiss_palette(self) -> None:
        self.dismiss(None)
