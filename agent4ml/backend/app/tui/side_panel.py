"""SidePanel — 宽屏时右侧会话信息面板（textual Static + rich）。

展示：Context 用量 / 上下文窗口 / Compact 进度条 / MCP / 版本。
终端足够宽（>= 阈值）时由 App 挂 ``wide`` class 显示，窄屏隐藏。
配色统一从 ``theme`` 取值，明细行前缀用限定的语义强调色区分类别，避免杂乱。
"""

from __future__ import annotations

from typing import Any

from rich.console import Group
from rich.text import Text
from textual.widgets import Static

from agent4ml.backend.app.tui import theme


def _format_tokens(tokens: int) -> str:
    if tokens >= 1_000_000:
        return f"{tokens / 1_000_000:.1f}M"
    if tokens >= 1_000:
        return f"{tokens / 1_000:.1f}K"
    return str(tokens)


def _bar(fraction: float, width: int = 22) -> Text:
    fraction = max(0.0, min(1.0, fraction))
    filled = int(round(fraction * width))
    t = Text()
    t.append("█" * filled, style=theme.BAR_FILL)
    t.append("█" * (width - filled), style=theme.BAR_EMPTY)
    return t


class SidePanel(Static):
    """右侧会话信息面板。

    左侧竖线不用 CSS ``border-left``——见 D12：真实终端（Windows Terminal/
    PowerShell）下 border-left 在部分行会渲染出异常的白色线段（headless SVG
    导出复现不了，怀疑是该终端对 box-drawing 字符的边框特殊路径处理有关），
    改成手工在每行内容前拼一个受 theme.BORDER 控制的竖线字符，完全绕开
    Textual border 渲染路径，同时 ``height: 1fr`` 撑满整列，未占满的空白行
    背景仍是 SURFACE_ALT（不再有竖线，纯背景色分隔即可，不影响辨识度）。
    """

    DEFAULT_CSS = f"""
    SidePanel {{
        width: 44;
        height: 1fr;
        background: {theme.SURFACE_ALT};
        color: {theme.TEXT_SECONDARY};
        padding: 1 2 1 1;
    }}
    """

    def update_state(self, cli_state: dict[str, Any]) -> None:
        tokens = cli_state.get("current_tokens", 0)
        fraction = cli_state.get("current_fraction", 0.0)
        window = cli_state.get("current_window", 0)
        mcp_count = cli_state.get("mcp_count", 0)
        model = cli_state.get("model", "?")
        mode = cli_state.get("mode", "default")
        sandbox_id = cli_state.get("sandbox_id", "")
        mcp_servers = cli_state.get("mcp_servers", [])
        pct = fraction * 100.0
        pct_color = theme.ACCENT_WARN if pct >= 80 else theme.MARK_WINDOW

        def header(text: str) -> Text:
            return Text(text, style=f"bold {theme.ACCENT_BRAND}")

        def mark_row(mark_color: str, label: str, value: str) -> Text:
            t = Text()
            t.append("■ ", style=mark_color)
            t.append(f"{label}: ", style=theme.TEXT_SECONDARY)
            t.append(value, style=theme.TEXT_PRIMARY)
            return t

        lines: list[Any] = []

        # Context——顶部大字号高亮当前用量
        lines.append(header("Context"))
        ctx_line = Text()
        ctx_line.append(f"{_format_tokens(tokens)} ", style=f"bold {theme.MARK_WINDOW}")
        ctx_line.append(f"({pct:.1f}%)", style=pct_color)
        lines.append(ctx_line)
        lines.append(Text(""))

        # Context Detail——每行前缀不同语义色方块，呼应参考设计但收敛色阶
        lines.append(header("Context Detail"))
        lines.append(mark_row(theme.MARK_WINDOW, "Context window", f"{_format_tokens(window)} ({pct:.1f}%)"))
        lines.append(mark_row(theme.MARK_MODEL, "Model", model))
        lines.append(mark_row(theme.MARK_MODE, "Mode", mode))
        lines.append(mark_row(theme.MARK_MCP, "MCP tools", str(mcp_count)))
        free = max(0.0, 1.0 - fraction) * 100.0
        lines.append(mark_row(theme.MARK_FREE, "Free space", f"{free:.1f}%"))
        lines.append(Text(""))

        # Compact 进度条
        lines.append(header("Compact"))
        bar = _bar(fraction)
        bar.append(f"  {pct:.1f}%", style=pct_color)
        lines.append(bar)
        usable = window - tokens if window else 0
        lines.append(Text(
            f"{_format_tokens(tokens)} usage · {_format_tokens(max(0, usable))} usable",
            style=theme.TEXT_DIM,
        ))
        lines.append(Text(""))

        # Sandbox——沙箱状态
        lines.append(header("Sandbox"))
        if sandbox_id:
            lines.append(mark_row(theme.MARK_WINDOW, "Sandbox ID", sandbox_id))
            lines.append(mark_row(theme.MARK_FREE, "Status", "active"))
        else:
            lines.append(mark_row(theme.TEXT_DIM, "Sandbox ID", "—"))
            lines.append(mark_row(theme.TEXT_DIM, "Status", "idle"))
        lines.append(Text(""))

        # MCP——已加载 server 列表
        lines.append(header("MCP"))
        if mcp_servers:
            total_tools = sum(s.get("tool_count", 0) for s in mcp_servers)
            for s in mcp_servers:
                mark = theme.MARK_FREE if s.get("health_state") == "healthy" else theme.ACCENT_WARN
                status_dot = "●" if s.get("health_state") == "healthy" else "○"
                row = Text()
                row.append("■ ", style=mark)
                row.append(f"{s['name']}", style=theme.TEXT_PRIMARY)
                row.append(f"  {s['transport']}  {status_dot}", style=theme.TEXT_DIM)
                lines.append(row)
            lines.append(mark_row(theme.MARK_MCP, "tools", f"{total_tools} loaded"))
        else:
            lines.append(mark_row(theme.TEXT_DIM, "servers", "none"))
        lines.append(Text(""))

        # 底部版本
        lines.append(Text("Agent4ML v1.0.0", style=theme.TEXT_DIM))

        # 手工竖线前缀（非空行才加，空行只留背景，避免竖线断续更显眼）
        prefixed: list[Any] = []
        for line in lines:
            if isinstance(line, Text) and not line.plain:
                prefixed.append(line)
                continue
            bar_line = Text("│ ", style=theme.BORDER)
            bar_line.append_text(line)
            prefixed.append(bar_line)

        self.update(Group(*prefixed))
