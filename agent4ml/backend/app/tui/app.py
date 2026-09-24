"""Agent4MLTUI — 全屏 TUI 应用（textual）。

双状态布局：
- **欢迎态**（``welcome-mode``）：logo 居中 + 副标题 + 居中输入框 + 一行提示。
- **对话态**（``conversation-mode``）：左侧对话列（滚动区 + 底部深灰蒙版输入 box + 细状态行），
  终端足够宽时右侧显示可交互会话信息面板（Context/Compact/MCP/版本）。

首次提交输入时从欢迎态切换到对话态。``agent4ml`` 启动此 TUI；``agent4ml cli`` 回退旧 CLI。
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from textual import work
from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Static, TextArea
from textual.binding import Binding
from textual.message import Message
from textual import events

from agent4ml.backend.app.cli.banner import render_logo
from agent4ml.backend.app.cli.commands import get_registry, handle_command
from agent4ml.backend.app.services.stream_service import Agent4MLStreamClient
from agent4ml.backend.app.tui import theme
from agent4ml.backend.app.tui.command_palette import CommandPalette
from agent4ml.backend.app.tui.conversation import ConversationLog
from agent4ml.backend.app.tui.help_screen import HelpRequestScreen
from agent4ml.backend.app.tui.mcp_panel import McpPanel
from agent4ml.backend.app.tui.side_panel import SidePanel
from agent4ml.backend.app.tui.status_bar import StatusBar

# 宽屏阈值：>= 此列数时展开右侧会话信息面板（仅全屏/极宽终端触发）。
_WIDE_THRESHOLD = 160


@dataclass
class DraftState:
    """未提交的用户草稿；展示层不得修改 ``text``。"""

    text: str = ""
    visible_line_limit: int = 10
    character_limit: int = 800

    @property
    def line_count(self) -> int:
        return max(len(self.text.splitlines()), 1)

    @property
    def character_count(self) -> int:
        return len(self.text)

    @property
    def is_summarized(self) -> bool:
        return (
            self.line_count > self.visible_line_limit
            or self.character_count > self.character_limit
        )


# ---- Widgets ----


class WelcomeView(Container):
    """欢迎页——logo 居中 + 副标题 + 居中输入框 + 一行提示。

    ``#logo``/``#subtitle``/``#tip`` 用 ``width: 100%`` + ``content-align``/
    ``text-align: center`` 居中——它们是纯文本展示，占满整行再居中文字最稳妥。

    ``#welcome-input`` 不能直接放在 WelcomeView 下用父级 ``align: center middle``
    居中：Textual 的 vertical 布局 align 是把"整组子控件"作为一个块居中（块宽度
    取全部子控件里最宽的那个），块内每个子控件仍然是左对齐拼接，不是逐个居中。
    一旦同一层出现 width:100% 的兄弟（logo/subtitle/tip），或子控件宽度互不相同，
    这个块宽度就会被撑到跟父容器一样宽，居中 offset 直接变 0——表现为除最宽的
    那个子控件外，其余子控件全部贴左，输入框就是这么跟丢的。用 ``InputRow``
    单独包一层：InputRow 自己 width:100%（填满 WelcomeView 宽度），组内只有
    唯一子控件 Input，align 单子控件时行为正常，真正把 Input 居中。
    """

    DEFAULT_CSS = f"""
    WelcomeView {{
        align: center middle;
        height: 1fr;
        background: {theme.BG};
        padding: 0 4;
    }}
    WelcomeView > #logo {{
        width: 100%;
        height: auto;
        content-align: center middle;
        margin: 0 0 1 0;
    }}
    WelcomeView > #subtitle {{
        width: 100%;
        text-align: center;
        color: {theme.ACCENT_BRAND};
        margin: 0 0 2 0;
    }}
    WelcomeView > InputRow {{
        width: 100%;
        height: auto;
        min-height: 3;
        align: center middle;
    }}
    WelcomeView > InputRow > #welcome-input {{
        width: 64;
        max-width: 90%;
        height: 3;
        border: tall {theme.BORDER};
        background: {theme.SURFACE};
        color: {theme.TEXT_PRIMARY};
        padding: 0 1;
    }}
    WelcomeView > InputRow > #welcome-input:focus {{
        border: tall {theme.BORDER_FOCUS};
    }}
    WelcomeView > #tip {{
        width: 100%;
        text-align: center;
        color: {theme.TEXT_DIM};
        margin: 1 0 0 0;
    }}
    """


class InputRow(Container):
    """欢迎页输入框的单独居中层——见 ``WelcomeView`` docstring 的对齐说明。"""


class ConversationInput(TextArea):
    """对话页输入框——多行 TextArea，支持粘贴多行内容不丢失。

    替代原生 ``Input``：``Input._on_paste`` 用 ``splitlines()[0]`` 截断多行粘贴，
    导致粘贴文章 / 代码片段只剩第一行，后续行被静默丢弃。TextArea 原生支持多行，
    ``_on_paste`` 完整保留，从根源消除内容丢失。

    交互（保持原 Input UX 同时支持多行编辑）：
    - ``Enter``：发送（与原 Input 一致，避免破坏既有使用习惯）。
    - ``Ctrl+Enter`` / ``Alt+Enter``：插入换行（终端能区分这两个组合键时）。
      覆盖 TextArea 默认的 ``Enter`` 换行行为——默认行为下用户敲 Enter 期望发送，
      而不是换行，必须改。

    高度策略（``_on_text_area_changed``）：按内容行数动态设高，封顶
    ``MAX_VISIBLE_LINES``。不能用 ``height: auto``——TextArea 是 ScrollView
    子类，``auto`` 会让它尝试渲染全部行（无封顶），长粘贴（30+ 行）触发持续
    layout 重算 + 容器级联 resize → 欢迎页 InputRow 错位 / 对话页输入卡死光标失效。
    """

    MAX_VISIBLE_LINES = 10
    """最大可见行数——超过此行数后 TextArea 内部滚动，避免撑高容器。"""

    BINDINGS = [
        Binding("ctrl+c", "copy", "Copy", show=False, priority=True),
        Binding("ctrl+z", "undo", "Undo", show=False, priority=True),
        Binding("ctrl+y", "redo", "Redo", show=False, priority=True),
    ]

    class Submitted(Message):
        """提交事件——对应原 ``Input.Submitted``，保持 App 层 handler 签名兼容。"""

        def __init__(self, input: "ConversationInput", value: str) -> None:
            self.input = input
            self.value = value
            super().__init__()

    DEFAULT_CSS = f"""
    ConversationInput {{
        height: 1;
        border: none;
        background: transparent;
        color: {theme.TEXT_PRIMARY};
        padding: 0;
    }}
    ConversationInput:focus {{
        border: none;
        background: transparent;
    }}
    """

    def _on_text_area_changed(self, event: TextArea.Changed) -> None:
        """按内容调整高度；欢迎页保留 tall 边框所需的三行最小高度。"""
        try:
            line_count = self.document.line_count
            minimum_height = 3 if self.id == "welcome-input" else 1
            self.styles.height = max(
                minimum_height,
                min(line_count, self.MAX_VISIBLE_LINES),
            )
        except Exception:
            pass

    async def _on_key(self, event: events.Key) -> None:
        # Enter（无修饰键）→ 发送。覆盖 TextArea 默认的 ``enter → 插入 \n``。
        #TextArea 默认在 ``_on_key`` 里把 ``enter`` 映射为 ``\n`` 插入，不走 BINDINGS，
        # 所以光在 BINDINGS 里加 binding 无效，必须在此拦截。
        if event.key == "enter":
            event.stop()
            event.prevent_default()
            self.post_message(self.Submitted(self, self.text))
            return
        # Ctrl+Enter / Alt+Enter → 手动换行。终端能区分这两个组合键时才触发——
        # 大多数现代终端（Windows Terminal / iTerm2 / kitty / alacritty）支持。
        if event.key in ("ctrl+enter", "alt+enter"):
            event.stop()
            event.prevent_default()
            self.insert("\n")
            return
        await super()._on_key(event)


class DraftSummary(Static):
    """长草稿折叠展示；完整文本由 ``DraftState`` 保存。"""

    can_focus = True
    BINDINGS = [
        Binding("enter", "submit_draft", "Send", show=False),
        Binding("ctrl+e", "edit_draft", "Edit", show=False),
        Binding("escape", "clear_draft", "Clear", show=False),
    ]

    DEFAULT_CSS = f"""
    DraftSummary {{
        height: auto;
        background: {theme.SURFACE};
        border-left: solid {theme.ACCENT_USER};
        padding: 1 2;
        color: {theme.TEXT_PRIMARY};
    }}
    DraftSummary:focus {{
        border-left: solid {theme.BORDER_FOCUS};
    }}
    """

    def __init__(self, draft: DraftState, *, id: str) -> None:
        super().__init__(self._format_draft(draft), id=id)
        self.draft = draft

    @staticmethod
    def _format_draft(draft: DraftState) -> str:
        preview = next((line.strip() for line in draft.text.splitlines() if line.strip()), "")
        if len(preview) > 72:
            preview = f"{preview[:72]}…"
        return (
            f"[Pasted · {draft.line_count} lines · {draft.character_count} chars]\n"
            f"{preview}\n"
            "Enter send full draft  ·  Ctrl+E edit  ·  Esc clear"
        )

    def action_submit_draft(self) -> None:
        self.post_message(ConversationInput.Submitted(self, self.draft.text))

    def action_edit_draft(self) -> None:
        self.app.open_draft_editor()

    def action_clear_draft(self) -> None:
        self.app.clear_draft()

    def refresh_draft(self, draft: DraftState) -> None:
        self.draft = draft
        self.update(self._format_draft(draft))


class DraftEditor(ModalScreen[str | None]):
    """长草稿完整编辑视图。"""

    BINDINGS = [
        Binding("escape", "cancel", "Cancel", show=False),
        Binding("ctrl+enter", "submit_draft", "Send", show=False),
    ]

    DEFAULT_CSS = f"""
    DraftEditor {{ align: center middle; background: $background 0%; }}
    DraftEditor > #draft-editor-box {{
        width: 90%; height: 80%; background: {theme.SURFACE};
        border: solid {theme.BORDER_FOCUS}; padding: 1 2;
    }}
    DraftEditor #draft-editor-input {{ height: 1fr; border: none; }}
    DraftEditor #draft-editor-actions {{ height: 3; align: right middle; }}
    """

    def __init__(self, text: str) -> None:
        super().__init__()
        self._text = text

    def compose(self) -> ComposeResult:
        with Container(id="draft-editor-box"):
            yield Static("Edit full draft · Ctrl+Enter send · Esc keep editing later")
            yield TextArea(self._text, soft_wrap=True, id="draft-editor-input")
            with Horizontal(id="draft-editor-actions"):
                yield Button("Keep draft", id="draft-cancel")
                yield Button("Send", variant="primary", id="draft-send")

    def on_mount(self) -> None:
        self.query_one("#draft-editor-input", TextArea).focus()

    def action_cancel(self) -> None:
        self.dismiss(self.query_one("#draft-editor-input", TextArea).text)

    def action_submit_draft(self) -> None:
        self.dismiss(self.query_one("#draft-editor-input", TextArea).text)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "draft-send":
            self.action_submit_draft()
        elif event.button.id == "draft-cancel":
            self.action_cancel()


class InputBox(Container):
    """底部深灰输入框——直角矩形 + 左细竖线，和历史消息卡片同一视觉语言，简约极客。

    内含输入行 + Build·模型 信息行，统一间距。上下留白全部用 InputBox 自身的
    padding 控制（不再额外给子控件加 margin），避免两套间距机制在窗口重绘/
    resize 时相互打架造成错位。
    """

    DEFAULT_CSS = f"""
    InputBox {{
        height: auto;
        background: {theme.SURFACE};
        border-left: solid {theme.ACCENT_USER};
        padding: 1 2 1 2;
        margin: 0 2 0 2;
    }}
    InputBox:focus-within {{
        border-left: solid {theme.BORDER_FOCUS};
    }}
    InputBox > #input-info {{
        height: 1;
        color: {theme.TEXT_SECONDARY};
        padding: 0;
        margin: 0;
    }}
    """


# ---- App ----


class Agent4MLTUI(App):
    """Agent4ML 全屏 TUI——双状态布局。"""

    # Textual 内置 ctrl+p 会打开系统自带的 command palette（textual.command
    # .CommandPalette），和自定义的 Ctrl+P 面板抢同一键位、优先级更高会直接
    # 吞掉按键——必须关掉内置的，才能让 action_toggle_command_palette 生效。
    ENABLE_COMMAND_PALETTE = False

    CSS = f"""
    Screen {{
        background: {theme.BG};
    }}

    /* 欢迎态：仅显示 WelcomeView */
    Agent4MLTUI.welcome-mode WelcomeView {{ display: block; }}
    Agent4MLTUI.welcome-mode #conv-body {{ display: none; }}

    /* 对话态：显示对话主体 */
    Agent4MLTUI.conversation-mode WelcomeView {{ display: none; }}
    Agent4MLTUI.conversation-mode #conv-body {{ display: block; }}
    Agent4MLTUI.welcome-mode #welcome-summary {{ display: none; }}
    Agent4MLTUI.welcome-mode.draft-summary #welcome-input {{ display: none; }}
    Agent4MLTUI.welcome-mode.draft-summary #welcome-summary {{ display: block; }}
    Agent4MLTUI.conversation-mode #conv-summary {{ display: none; }}
    Agent4MLTUI.conversation-mode.draft-summary #conv-input {{ display: none; }}
    Agent4MLTUI.conversation-mode.draft-summary #conv-summary {{ display: block; }}

    #conv-body {{
        layout: horizontal;
        height: 1fr;
        padding: 0 0 1 0;
    }}
    #main-col {{
        width: 1fr;
        height: 1fr;
    }}
    #side {{
        width: 44;
        display: none;
    }}
    /* 宽屏才展开右侧会话信息面板 */
    Agent4MLTUI.wide.conversation-mode #side {{ display: block; }}

    #exec-panel {{
        height: 1;
        color: {theme.TEXT_SECONDARY};
        background: {theme.SURFACE};
        padding: 0 2;
        margin: 0 2;
        display: none;
    }}
    Agent4MLTUI.conversation-mode.running #exec-panel {{ display: block; }}
    """

    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit", show=False),
        Binding("ctrl+l", "clear_screen", "Clear", show=False),
        Binding("escape", "cancel_run", "Cancel", show=False, priority=True),
        Binding("ctrl+p", "toggle_command_palette", "Commands", show=False),
        Binding("ctrl+n", "toggle_mcp_panel", "MCP", show=True, priority=True),
        Binding("ctrl+b", "toggle_settings", "Settings", show=True, priority=True),
    ]

    def __init__(self, runtime: Any, provider: str | None = None, model: str | None = None) -> None:
        super().__init__()
        self.runtime = runtime
        self.provider = provider
        self.model = model
        self._first_input = True
        self.draft = DraftState()
        self.cli_state: dict[str, Any] = {
            "pending_expert_mode": None,
            "pending_report": None,
            "mode": "expert" if runtime.config.runtime.expert_mode else "default",
            "model": self._resolve_model_name(),
            "provider": provider or "",
            "current_tokens": 0,
            "current_fraction": 0.0,
            "current_window": self._resolve_window(),
            "mcp_count": 0,
            "msg_count": 0,
            "sandbox_id": "",
            "mcp_servers": [],
            "_running": False,
        }
        self.registry = get_registry()

    def _resolve_model_name(self) -> str:
        """Build 信息行展示的模型名——取当前活跃 provider 的真实 model 名
        （如 deepseek-v4-flash），不是 provider 链名（如 openai,qwen,deepseek）。
        与 _resolve_window 用同一模型对象来源，保证两处展示互相一致。
        """
        try:
            model = self.runtime.capability_registry.get_model("researcher")
            from agent4ml.backend.agents.context_engineering.utilities import resolve_model_name
            name = resolve_model_name(model)
            if name:
                return name
        except Exception:
            pass
        try:
            from agent4ml.backend.agents.leader.agent import _resolve_actual_model_name
            return _resolve_actual_model_name(self.runtime.capability_registry)
        except Exception:
            return "?"

    def _resolve_window(self) -> int:
        """上下文容量：与治理策略同一来源——registry.get_model("researcher") →
        resolve_window_size 穿透 FallbackChatModel 取活跃 provider 真实窗口。
        首个 budget 事件回流后会用 governance 的 window 覆盖此初始值。"""
        try:
            model = self.runtime.capability_registry.get_model("researcher")
            from agent4ml.backend.agents.context_engineering.utilities import resolve_window_size
            w = resolve_window_size(model)
            if w > 0:
                return w
        except Exception:
            pass
        return 128000

    def _resolve_mcp_count(self) -> int:
        try:
            from agent4ml.backend.agents.agent_tools.available import get_available_tools
            from agent4ml.backend.agents.agent_tools.mcp_metadata import is_mcp_tool
            tools = get_available_tools(include_mcp=True)
            return sum(1 for t in tools if is_mcp_tool(t))
        except Exception:
            return 0

    def _input_info(self) -> str:
        model = self.cli_state["model"]
        provider = self.cli_state.get("provider") or ""
        base = f"■ Build · {model}"
        return f"{base}  {provider}" if provider else base

    def compose(self) -> ComposeResult:
        # 欢迎页（居中 logo + 副标题 + 输入框 + 提示）
        subtitle = f"v1.0.0 · {self.cli_state['mode']} · {self.cli_state['model']}"
        yield WelcomeView(
            Static(render_logo(), id="logo"),
            Static(subtitle, id="subtitle"),
            InputRow(
                ConversationInput(placeholder="Ask anything...  (输入 / 查看命令)", id="welcome-input"),
                DraftSummary(self.draft, id="welcome-summary"),
            ),
            Static("", id="tip"),
        )
        # 对话主体：左对话列 + 右信息面板
        yield Horizontal(
            Vertical(
                ConversationLog(),
                Static("", id="exec-panel"),
                InputBox(
                    ConversationInput(
                        placeholder="Ask anything...  (输入 / 查看命令 · Enter 发送 · Ctrl+Enter 换行)",
                        id="conv-input",
                    ),
                    DraftSummary(self.draft, id="conv-summary"),
                    Static("", id="input-info"),
                ),
                StatusBar(),
                id="main-col",
            ),
            SidePanel(id="side"),
            id="conv-body",
        )

    def on_mount(self) -> None:
        self._apply_markdown_theme()
        self.add_class("welcome-mode")
        self._apply_responsive(self.size.width)
        self._load_mcp_count()
        self.query_one("#welcome-input", ConversationInput).focus()

    def _apply_markdown_theme(self) -> None:
        """agent 输出的 Markdown 标题统一改成深橙→橙色（呼应参考设计的标题色），
        不用 Rich Markdown 默认的紫色系。``self.console`` 是 RichLog.write() 实际
        用来渲染的 Console（见 textual RichLog.write 内部用 ``self.app.console``），
        push_theme 覆盖 ``markdown.h1``~``h6`` 的 style 即可全局生效，不用逐处传参。
        """
        from rich.theme import Theme
        heading_style = f"bold {theme.ACCENT_THOUGHT}"
        self.console.push_theme(Theme({
            "markdown.h1": heading_style,
            "markdown.h2": heading_style,
            "markdown.h3": heading_style,
            "markdown.h4": heading_style,
            "markdown.h5": heading_style,
            "markdown.h6": theme.ACCENT_THOUGHT,
        }))

    def on_resize(self, event: Any) -> None:
        self._apply_responsive(event.size.width)

    def _apply_responsive(self, width: int) -> None:
        if width >= _WIDE_THRESHOLD:
            self.add_class("wide")
            self._refresh_side_panel()
        else:
            self.remove_class("wide")

    def _refresh_side_panel(self) -> None:
        try:
            self.query_one(SidePanel).update_state(self.cli_state)
        except Exception:
            pass

    def action_cancel_run(self) -> None:
        """Esc: single press = interrupt (resumable), double within 3s = rollback."""
        if not self.cli_state.get("_running"):
            return
        now = time.monotonic()
        last_esc = getattr(self, "_last_esc_time", 0)
        if now - last_esc < 3.0:
            self._cancel_run(action="rollback")
            self._last_esc_time = 0
        else:
            self._cancel_run(action="interrupt")
            self._last_esc_time = now

    def _cancel_run(self, action: str = "interrupt") -> None:
        """Cancel current run: stop stream + journal + UI cleanup."""
        from rich.text import Text
        conv = self.query_one(ConversationLog)
        label = "interrupted" if action == "interrupt" else "abandoned"
        conv.write(Text(f"⏹ Run {label} by user.", style=theme.ACCENT_WARN))
        self.cli_state["_running"] = False
        self.remove_class("running")
        self.query_one(StatusBar).update_state(self.cli_state)
        self._update_exec_panel(None)
        self._focus_conv_input()

    @work(thread=True)
    def _load_mcp_count(self) -> None:
        count = self._resolve_mcp_count()
        self.cli_state["mcp_count"] = count
        mcp_mgr = getattr(self.runtime, "mcp_manager", None) if self.runtime else None
        self.cli_state["mcp_servers"] = mcp_mgr.list_servers() if mcp_mgr else []
        self.call_from_thread(self._apply_mcp_count, count)

    def _apply_mcp_count(self, count: int) -> None:
        tip = self.query_one("#tip", Static)
        mcp_hint = f"{count} MCP tools" if count else "no MCP tools"
        tip.update(f"{mcp_hint}  ·  Shift+拖动选取复制  ·  /help 命令  ·  Ctrl+C 退出")
        self._refresh_side_panel()

    def action_clear_screen(self) -> None:
        if self.has_class("conversation-mode"):
            self.query_one(ConversationLog).clear()

    def action_toggle_command_palette(self) -> None:
        """Ctrl+P 弹出/收起命令面板（占位 UI，列表项暂不接实际执行）。"""
        if isinstance(self.screen, CommandPalette):
            self.pop_screen()
        else:
            self.push_screen(CommandPalette())

    def action_toggle_mcp_panel(self) -> None:
        """Ctrl+M 弹出/收起 MCP 管理面板（运行时加载 MCP server）。"""
        if isinstance(self.screen, (CommandPalette, McpPanel)):
            self.pop_screen()
        else:
            mcp_manager = getattr(self.runtime, "mcp_manager", None) if self.runtime else None
            self.push_screen(McpPanel(mcp_manager, self.runtime))

    def action_toggle_settings(self) -> None:
        """Ctrl+B 弹出/收起配置面板（编辑 API Key / Base URL，直写 .env）。"""
        from pathlib import Path
        from agent4ml.backend.app.tui.settings_screen import SettingsScreen

        if isinstance(self.screen, SettingsScreen):
            self.pop_screen()
        else:
            env_path = Path(__file__).parents[4] / ".env"
            if not env_path.exists():
                from rich.text import Text
                conv = self.query_one(ConversationLog)
                conv.write(Text(".env not found — run setup wizard first.", style="dim"))
                return
            self.push_screen(SettingsScreen(env_path))

    def _focus_conv_input(self) -> None:
        """聚焦对话输入框（切换布局/完成一轮后调用，确保可继续输入）。"""
        try:
            if self.has_class("conversation-mode"):
                self.query_one("#conv-input", ConversationInput).focus()
        except Exception:
            pass

    def _render_draft(self) -> None:
        for summary_id in ("#welcome-summary", "#conv-summary"):
            self.query_one(summary_id, DraftSummary).refresh_draft(self.draft)
        if self.draft.is_summarized:
            self.add_class("draft-summary")
            active_summary = "#conv-summary" if self.has_class("conversation-mode") else "#welcome-summary"
            self.query_one(active_summary, DraftSummary).focus()
        else:
            self.remove_class("draft-summary")

    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        if not isinstance(event.text_area, ConversationInput):
            return
        if event.text_area.text == self.draft.text:
            return
        self.draft.text = event.text_area.text
        self._render_draft()

    def clear_draft(self) -> None:
        self.draft.text = ""
        for input_id in ("#welcome-input", "#conv-input"):
            input_widget = self.query_one(input_id, ConversationInput)
            if input_widget.text:
                input_widget.text = ""
        self._render_draft()
        self._focus_conv_input()

    def open_draft_editor(self) -> None:
        self.push_screen(DraftEditor(self.draft.text), self._save_edited_draft)

    def _save_edited_draft(self, text: str | None) -> None:
        if text is None:
            return
        self.draft.text = text
        for input_id in ("#welcome-input", "#conv-input"):
            input_widget = self.query_one(input_id, ConversationInput)
            if input_widget.text != text:
                input_widget.text = text
        self._render_draft()

    async def on_conversation_input_submitted(self, event: ConversationInput.Submitted) -> None:
        text = event.value.strip()
        self.clear_draft()
        if not text:
            return

        # Steer: running 时用户输入不中断，排队等当前工具 batch 完成后注入
        if self.cli_state.get("_running") and not text.startswith("/"):
            self._queue_steer(text)
            return

        # 首次输入：从欢迎态切换到对话态
        if self._first_input:
            self._first_input = False
            self.remove_class("welcome-mode")
            self.remove_class("draft-summary")
            self.add_class("conversation-mode")
            self.query_one("#input-info", Static).update(self._input_info())
            self.query_one(StatusBar).update_state(self.cli_state)
            self._refresh_side_panel()
            # display 从 none 切到 block 后需等刷新完成再聚焦，否则焦点丢失
            self.call_after_refresh(self._focus_conv_input)

        # / 命令
        if text.startswith("/"):
            self._handle_command(text)
            return

        # 意图识别
        from agent4ml.backend.agents.intent import default_intent_tree
        intent_tree = default_intent_tree(report_handler=self._handle_report_intent)
        if intent_tree.detect_and_dispatch(text, self.runtime):
            return

        # 研究流
        self._run_research(text)

    def _handle_command(self, cmd: str) -> None:
        """处理 / 命令。"""
        from rich.console import Console
        from io import StringIO

        buf = StringIO()
        temp_console = Console(file=buf, force_terminal=False, width=100)
        conv = self.query_one(ConversationLog)

        class _MockRenderer:
            state = conv.state
            def expand_last_round(self):
                conv.expand_last_round()
            def _stop_spinner(self):
                pass

        should_exit = handle_command(
            cmd, temp_console, _MockRenderer(), self.cli_state, self.runtime,
        )

        output = buf.getvalue().strip()
        if output:
            from rich.text import Text
            conv.write(Text(output))

        if should_exit:
            self.exit()

        pending = self.cli_state.get("pending_expert_mode")
        if pending is not None:
            self.runtime = self.runtime.switch_expert_mode(expert_mode=pending)
            self.cli_state["pending_expert_mode"] = None
            self.cli_state["mode"] = "expert" if pending else "default"
            self.cli_state["model"] = self._resolve_model_name()
            self.query_one("#input-info", Static).update(self._input_info())
            self.query_one(StatusBar).update_state(self.cli_state)
            self._refresh_side_panel()

        pending_report = self.cli_state.get("pending_report")
        if pending_report is not None:
            self.cli_state["pending_report"] = None
            self._trigger_report(pending_report)

    def _handle_report_intent(self, intent: Any, rt: Any) -> bool:
        self.cli_state["pending_report"] = ""
        return True

    def _trigger_report(self, topic: str) -> None:
        conv = self.query_one(ConversationLog)
        from rich.text import Text
        conv.write(Text(f"Generating report{' on: ' + topic if topic else ''}...", style="cyan"))
        try:
            self.runtime.generate_report(topic=topic or None)
            conv.write(Text("Report generated.", style="green"))
        except Exception as exc:
            conv.write(Text(f"Report failed: {exc}", style="red"))

    @work(exclusive=True, group="research")
    async def _run_research(self, question: str) -> None:
        """流式研究——Agent4MLStreamClient → ConversationLog + StatusBar + SidePanel。"""
        conv = self.query_one(ConversationLog)
        status = self.query_one(StatusBar)

        conv.render_user_input(question)
        conv.state["round_t0"] = time.monotonic()
        conv.state["model"] = self.cli_state["model"]

        self.cli_state["_running"] = True
        status.update_state(self.cli_state)
        self.add_class("running")
        self._run_start_time = time.monotonic()

        try:
            ctx = self.runtime.run_manager.create_run(
                thread_id=self.runtime.thread_id,
                user_id="default-user",
                run_id=None,
                model_name=self.runtime.researcher_model_name,
                thread_dir=self.runtime.thread_dir,
            )
            self.runtime.run_manager.mark_running(ctx.run_id)
            config = self._build_stream_config(ctx)
            client = Agent4MLStreamClient(graph=self.runtime.leader_agent.graph, config=config)

            async for event in client.stream(question):
                budget = event.get("budget")
                if budget:
                    self.cli_state["current_tokens"] = budget.get("total", 0)
                    win = budget.get("window") or self.cli_state["current_window"]
                    self.cli_state["current_window"] = win
                    frac = budget.get("fraction")
                    if not frac and win:
                        frac = self.cli_state["current_tokens"] / win
                    self.cli_state["current_fraction"] = frac or 0.0
                    status.update_state(self.cli_state)
                    self._refresh_side_panel()
                if event.get("type") == "sandbox_update":
                    self.cli_state["sandbox_id"] = event.get("content", "")
                    self._refresh_side_panel()
                if event.get("type") in ("tool_start", "tool_end"):
                    self._update_exec_panel(event)
                conv.render_event(event)

            self.runtime.run_manager.mark_success(ctx.run_id)

        except Exception as exc:
            from rich.text import Text
            conv.write(Text(f"✗ {exc}", style=theme.ACCENT_WARN))
            try:
                self.runtime.run_manager.mark_failed(ctx.run_id, str(exc))
            except Exception:
                pass
        finally:
            self.cli_state["_running"] = False
            self.remove_class("running")
            status.update_state(self.cli_state)
            self._refresh_side_panel()
            self._update_exec_panel(None)
            # 每轮结束后重新聚焦输入框，保证用户可继续输入
            self._focus_conv_input()

    def _update_exec_panel(self, event: Any) -> None:
        """Update execution panel with current activity info."""
        try:
            panel = self.query_one("#exec-panel", Static)
        except Exception:
            return
        if event is None:
            panel.update("")
            return
        elapsed = time.monotonic() - getattr(self, "_run_start_time", time.monotonic())
        mins = int(elapsed // 60)
        secs = int(elapsed % 60)
        tool_name = event.get("tool_name") or ""
        if event.get("type") == "tool_start":
            panel.update(f"● {tool_name} · {mins}m{secs:02d}s · Esc cancel")
        elif event.get("type") == "tool_end":
            panel.update(f"✓ {tool_name} · {mins}m{secs:02d}s")
            self._flush_steer_queue()

    def _queue_steer(self, text: str) -> None:
        """Queue user input as steer text (non-interrupting injection)."""
        if not hasattr(self, "_steer_queue"):
            self._steer_queue: list[str] = []
        self._steer_queue.append(text)
        from rich.text import Text
        conv = self.query_one(ConversationLog)
        conv.write(Text(f"⚡ Steer queued: {text[:80]}", style="dim"))

    def _flush_steer_queue(self) -> None:
        """Display queued steer messages after tool batch completes."""
        if not hasattr(self, "_steer_queue") or not self._steer_queue:
            return
        from rich.text import Text
        conv = self.query_one(ConversationLog)
        for msg in self._steer_queue:
            conv.write(Text(f"  ↳ Steer applied: {msg[:80]}", style="dim"))
        self._steer_queue.clear()

    def _build_stream_config(self, ctx: Any) -> dict:
        from agent4ml.backend.app.cli.main import _build_stream_config
        return _build_stream_config(self.runtime, ctx)
