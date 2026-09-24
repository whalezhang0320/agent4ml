"""ConversationLog — 可滚动对话区 Widget（textual RichLog）。

渲染 StreamEvent → rich renderable，写入 RichLog 自动滚动。
Thought 折叠 + 计时逻辑与 ``app/cli/stream_handler.py`` 的 StreamRenderer 一致，
但输出目标是 RichLog.write() 而非 console.print()。

复用 ``stream_handler`` 的 ``_tool_color`` / ``_truncate_args`` / ``_result_summary``
helper，避免逻辑重复——TUI 与 CLI 渲染口径完全一致。
"""

from __future__ import annotations

import time
from typing import Any

from rich.box import Box
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text
from textual.widgets import RichLog

from agent4ml.backend.app.cli.stream_handler import (
    _result_summary,
    _tool_color,
    _truncate_args,
)
from agent4ml.backend.app.services.stream_service import StreamEvent
from agent4ml.backend.app.tui import theme

# 仅左侧竖条的 box——左强调条 + 填充底，用户消息与助手回答共用同一视觉语言，
# 仅强调色不同（用户=蓝紫，助手=青绿），呼应参考设计的对话栏形式。
# 8 行分别对应 top/head/head_row/mid/row/foot_row/foot/bottom：Panel 用 top 画
# 首行、mid 画内容行、bottom 画末行——若仅 mid 行留竖条字符，首/末行会被渲染成
# 空白（背景仍填充），导致竖条比卡片矮一头一尾；这里每一行左列都设为同一字符，
# 保证竖条从卡片第一行到最后一行完整贯通。用 "│"（轻量竖线，U+2502）而非块
# 字符（如 "▏"/"▌"）——块字符的"墨色"贴在字符格左侧边缘，视觉上会比同列的
# InputBox（Textual border-left: solid，渲染同款 "│"）看起来往左偏了小半个
# 字符；换成同一根线字符后两处竖条视觉粗细一致、左边缘完全对齐。
_LEFT_BAR = Box(
    "│   \n"
    "│   \n"
    "│   \n"
    "│   \n"
    "│   \n"
    "│   \n"
    "│   \n"
    "│   \n"
)


class ConversationLog(RichLog):
    """对话区——接收 StreamEvent 并渲染为 rich renderable，自动滚动。

    state 字段与 ``StreamRenderer.state`` 对齐，供 /expand 等命令读取。
    """

    can_focus = False
    """不可聚焦——RichLog 默认可聚焦，鼠标点击对话区会触发 Textual 内置
    ``:focus`` 态的背景高亮蒙版（tint），且该蒙版覆盖范围和底部 InputBox
    的矩形边界不对齐，视觉上像"错位的灰色蒙版"。对话区只用于展示/滚动，
    不需要键盘焦点，直接禁用 focus 从根源上消除这块蒙版。
    """

    DEFAULT_CSS = f"""
    ConversationLog {{
        height: 1fr;
        background: {theme.BG};
        border: none;
        padding: 0 2;
        scrollbar-size: 0 0;
    }}
    """


    def __init__(self) -> None:
        super().__init__(markup=True, auto_scroll=True)
        self.state: dict[str, Any] = {
            "full_answer": "",
            "tool_results": [],
            "thinking_enabled": True,
            "thinking_log": [],
            "_thinking_t0": None,
            "_thinking_buffer": "",
            "round_t0": None,
            "model": None,
            "_round_active": False,
        }

    def render_banner(self, banner_text: Text) -> None:
        """启动时渲染 Logo banner。"""
        self.write(banner_text)
        self.write("")

    def render_user_input(self, text: str) -> None:
        """用户输入——全宽左竖条填充卡片。"""
        self._flush_thinking()
        self.write(Panel(
            Text(text, style=theme.TEXT_PRIMARY),
            box=_LEFT_BAR,
            border_style=theme.ACCENT_USER,
            style=f"on {theme.SURFACE}",
            padding=(0, 2),
            expand=True,
        ))
        self.write("")

    def render_event(self, event: StreamEvent) -> None:
        """StreamEvent → rich renderable，写入 RichLog。"""
        etype = event["type"]

        # budget_update：不渲染到对话区，仅更新外部 cli_state（由 App 处理）
        if etype == "budget_update":
            return

        # 新轮首个事件时清上一轮 state
        if not self.state["_round_active"]:
            self.state["full_answer"] = ""
            self.state["tool_results"] = []
            self.state["thinking_log"] = []
            self.state["_round_active"] = True

        # 从 thinking 切到其他事件时 flush
        if etype != "thinking" and self.state["_thinking_t0"] is not None:
            self._flush_thinking()

        if etype == "thinking":
            self._render_thinking(event)
        elif etype == "answer":
            self._render_answer(event)
        elif etype == "tool_start":
            self._render_tool_start(event)
        elif etype == "tool_end":
            self._render_tool_end(event)
        elif etype == "skill_active":
            self._render_skill_active(event)
        elif etype == "done":
            self._render_done()
        elif etype == "error":
            self._render_error(event)
        elif etype == "compaction_start":
            self.write(Text(f"[compaction] {event.get('tool_name', '')} triggered", style="dim"))
        elif etype == "compaction_progress":
            self.write(Text(f"  {event['content']}", style="dim"))
        elif etype == "compaction_end":
            saved = event.get("tool_result") or ""
            self.write(Text(f"[compaction] done {event['content']} (saved={saved})", style="dim"))

    def _render_thinking(self, event: StreamEvent) -> None:
        """thinking——仅累积 buffer，不实时输出。与 CLI StreamRenderer 一致：
        段落结束（_flush_thinking）才打一行 ``+ Thought: Xms`` 摘要，原文供 /expand。
        """
        if not self.state["thinking_enabled"]:
            return
        content = event["content"]
        if not content:
            return
        if self.state["_thinking_t0"] is None:
            self.state["_thinking_t0"] = time.monotonic()
        self.state["_thinking_buffer"] += content

    def _flush_thinking(self) -> None:
        """thinking 段落收尾——打一行 ``+ Thought: Xms`` 摘要 + 记录 thinking_log。"""
        t0 = self.state["_thinking_t0"]
        if t0 is None:
            return
        elapsed_ms = int((time.monotonic() - t0) * 1000)
        text = self.state["_thinking_buffer"]
        self.write(Text(f"+ Thought: {elapsed_ms}ms", style=theme.ACCENT_THOUGHT))
        self.state["thinking_log"].append({"ms": elapsed_ms, "text": text})
        self.state["_thinking_t0"] = None
        self.state["_thinking_buffer"] = ""

    def _render_answer(self, event: StreamEvent) -> None:
        content = event["content"]
        if content:
            self.state["full_answer"] += content

    def _render_tool_start(self, event: StreamEvent) -> None:
        """工具调用请求行——与 CLI 一致：``tool_name(args)...`` 单行展示参数摘要。"""
        tool_name = event["tool_name"] or "unknown"
        args_str = _truncate_args(event["tool_args"])
        color = _tool_color(tool_name)
        line = Text()
        line.append(f"{tool_name}", style=color)
        if args_str:
            line.append(f"({args_str})...", style=theme.TEXT_DIM)
        else:
            line.append("...", style=theme.TEXT_DIM)
        self.write(line)

    def _render_skill_active(self, event: StreamEvent) -> None:
        skills = event.get("skills") or []
        if not skills:
            return
        label = Text("Skills", style=theme.TEXT_DIM)
        for skill in skills:
            label.append("\n  " + str(skill), style=theme.TEXT_DIM)
        self.write(label)

    def _render_tool_end(self, event: StreamEvent) -> None:
        """工具结果行——与 CLI 一致：``✓ tool_name → result_summary``。

        结果全文仍存入 state["tool_results"] 供 /expand 查看；此处仅展示 80 字摘要，
        让用户实时看到工具是否成功、大致返回了什么，而不是完全静默。
        """
        tool_name = event["tool_name"] or "unknown"
        summary = _result_summary(event["tool_result"])
        color = _tool_color(tool_name)
        line = Text()
        line.append("  ✓ ", style=color)
        line.append(tool_name, style=color)
        line.append(f" → {summary}", style=theme.TEXT_DIM)
        self.write(line)
        if event["tool_result"]:
            self.state["tool_results"].append({"tool": tool_name, "result": event["tool_result"]})

    def _render_done(self) -> None:
        full = self.state["full_answer"].strip()
        if full:
            from agent4ml.backend.app.services.stream_service import _strip_skills_leak
            full = _strip_skills_leak(full)
        if full:
            self.write("")
            self.write(Markdown(full, style=theme.TEXT_PRIMARY))
        # 耗时尾行——■ Build · model · Ns
        round_t0 = self.state.get("round_t0")
        model = self.state.get("model")
        if round_t0 is not None and model:
            import time
            elapsed = time.monotonic() - round_t0
            tail = Text()
            tail.append("  ■ ", style=theme.ACCENT_ASSISTANT)
            tail.append("Build", style=theme.TEXT_SECONDARY)
            tail.append(f" · {model} · {elapsed:.1f}s", style=theme.TEXT_DIM)
            self.write(tail)
        self.write("")
        self.state["full_answer"] = ""
        self.state["_round_active"] = False
        self.state["round_t0"] = None

    def _render_error(self, event: StreamEvent) -> None:
        self.write(Text(f"✗ {event['content']}", style=theme.ACCENT_WARN))

    def expand_last_round(self) -> None:
        """展开上一轮 Thought + 工具结果全文。"""
        thinking_log = self.state.get("thinking_log", [])
        results = self.state.get("tool_results", [])
        if not results and not thinking_log:
            self.write(Text("No previous round data to expand", style=theme.TEXT_DIM))
            return
        for entry in thinking_log:
            self.write(Panel(
                Text(entry["text"], style=theme.TEXT_SECONDARY),
                title=f"Thought ({entry['ms']}ms)",
                title_style=theme.ACCENT_THOUGHT,
                border_style=theme.BORDER,
            ))
        for r in results:
            self.write(Panel(
                Text(r["result"], style=theme.TEXT_SECONDARY),
                title=r["tool"],
                title_style=theme.TOOL_BLUE,
                border_style=theme.BORDER,
            ))
