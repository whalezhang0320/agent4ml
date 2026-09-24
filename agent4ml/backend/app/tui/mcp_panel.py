"""McpPanel — Ctrl+N MCP 管理面板（ModalScreen）。

风格延续 CommandPalette：居中浮层 + 标题行 + 分组 + 右对齐 esc 提示。
结构化输入 server 配置，调 McpManager.add_server 加载，成功触发 reload_mcp_tools。

INVARIANT:
- Ctrl+N 推面板，Esc/Ctrl+N 关
- 标题行左"MCP 管理"右"esc"，对齐 CommandPalette
- 已加载 server 列表用 row_text 风格（name + transport + tools + status 对齐）
- 输入区分组：连接（Name/Transport/URL-Cmd）+ 参数（Args/Headers-Env）
- Transport 用 RadioSet（stdio/http/sse）防输错
- 校验失败显示状态行，不崩溃退出
- 成功：刷新列表 + 提示 + 触发 reload + dismiss
- 失败：保留输入 + 显示错误（脱敏后）+ 可改重试
"""
from __future__ import annotations

import asyncio
import shlex
import shutil
from typing import Any

from textual import events
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Select, Static

from rich.text import Text

from agent4ml.backend.app.tui import theme
from agent4ml.backend.agents.mcp import McpManager, McpServerConfig


class McpPanel(ModalScreen[None]):
    """Ctrl+N 弹出的 MCP 管理面板。"""

    BINDINGS = [
        Binding("escape", "dismiss_panel", "Close", show=False),
        Binding("ctrl+n", "dismiss_panel", "Close", show=False),
        Binding("up", "focus_previous", "Previous", show=False, priority=True),
        Binding("down", "focus_next", "Next", show=False, priority=True),
        Binding("left", "focus_previous", "Previous", show=False, priority=True),
        Binding("right", "focus_next", "Next", show=False, priority=True),
        Binding("ctrl+enter", "submit_form", "Load MCP", show=False, priority=True),
    ]

    DEFAULT_CSS = f"""
    McpPanel {{
        align: center middle;
        background: $background 0%;
    }}
    McpPanel > #mcp-box {{
        width: 76;
        max-width: 92%;
        height: 38;
        max-height: 92%;
        background: {theme.SURFACE};
        border: solid {theme.BORDER};
        padding: 1 2;
    }}
    McpPanel #mcp-title {{
        height: 2;
        padding: 0 1;
        background: {theme.SURFACE_ALT};
        margin: 0 0 1 0;
    }}
    McpPanel #mcp-title-left {{
        color: {theme.TEXT_PRIMARY};
    }}
    McpPanel #mcp-title-right {{
        width: auto;
        height: 1;
        border: solid {theme.BORDER};
        padding: 0 1;
        color: {theme.TEXT_SECONDARY};
    }}
    McpPanel .mcp-section-title {{
        height: 1;
        color: {theme.ACCENT_BRAND};
        margin: 1 0 1 0;
    }}
    McpPanel #mcp-server-list {{
        height: 4;
        min-height: 4;
        max-height: 4;
        background: {theme.SURFACE_ALT};
        padding: 0 1;
        margin: 0 0 1 0;
    }}
    McpPanel .mcp-server-row {{
        height: 1;
    }}
    McpPanel #mcp-form {{
        height: 21;
        padding: 1;
        border: none;
        background: {theme.SURFACE_ALT};
    }}
    McpPanel .mcp-input-row {{
        height: 4;
        margin: 0;
        align: left middle;
    }}
    McpPanel .mcp-input-label {{
        width: 15;
        height: 4;
        content-align: left middle;
        color: {theme.TEXT_SECONDARY};
        padding: 0 1 0 0;
    }}
    McpPanel .mcp-input-field {{
        height: 3;
        border: none;
        background: {theme.BG};
        color: {theme.TEXT_PRIMARY};
        padding: 0 1;
    }}
    McpPanel .mcp-input-field:focus {{
        border: none;
        background: {theme.BG};
    }}
    McpPanel #mcp-transport-select {{
        height: 3;
        width: 1fr;
        background: {theme.BG};
        border: none;
        color: {theme.TEXT_PRIMARY};
        padding: 0 1;
    }}
    McpPanel #mcp-transport-select:focus {{
        border: none;
        background: {theme.BG};
    }}
    McpPanel #mcp-transport-select > .select--label {{
        color: {theme.TEXT_PRIMARY};
    }}
    McpPanel #mcp-transport-select > .select--selection {{
        color: {theme.TEXT_PRIMARY};
    }}
    McpPanel #mcp-transport-select > .select--button {{
        color: {theme.TEXT_SECONDARY};
        background: {theme.BG};
    }}
    McpPanel RadioSet > RadioButton {{
        color: {theme.TEXT_SECONDARY};
        padding: 0 2 0 0;
    }}
    McpPanel .mcp-hint {{
        height: 1;
        color: {theme.TEXT_DIM};
        margin: 1 0 0 0;
    }}
    McpPanel .mcp-button-row {{
        height: 3;
        align: right middle;
        margin: 1 0 0 0;
    }}
    McpPanel #mcp-add-btn {{
        min-width: 18;
        margin: 0 1 0 0;
    }}
    McpPanel #mcp-cancel-btn {{
        min-width: 12;
    }}
    McpPanel #mcp-status {{
        height: 3;
        border: none;
        background: {theme.SURFACE_ALT};
        color: {theme.TEXT_DIM};
        padding: 0 1;
        margin: 1 0 0 0;
    }}
    """

    def __init__(self, mcp_manager: McpManager | None, runtime: Any = None) -> None:
        super().__init__()
        self._mcp_manager = mcp_manager
        self._runtime = runtime

    def compose(self) -> ComposeResult:
        with Container(id="mcp-box"):
            with Horizontal(id="mcp-title"):
                yield Static("MCP 管理", id="mcp-title-left")
                yield Static("esc", id="mcp-title-right")
            yield Static("已加载 server", classes="mcp-section-title")
            with Vertical(id="mcp-server-list"):
                pass
            yield Static("添加 server", classes="mcp-section-title")
            with Vertical(id="mcp-form"):
                with Horizontal(classes="mcp-input-row"):
                    yield Static("Name", classes="mcp-input-label")
                    yield Input(placeholder="唯一名称，例如 my_search", id="mcp-name", classes="mcp-input-field")
                with Horizontal(classes="mcp-input-row"):
                    yield Static("Transport", classes="mcp-input-label")
                    yield Select(
                        [("stdio  · 本地命令", "stdio"), ("http  · 远程服务", "http"), ("sse  · 事件流", "sse")],
                        value="stdio",
                        allow_blank=False,
                        id="mcp-transport-select",
                    )
                with Horizontal(classes="mcp-input-row"):
                    yield Static("URL / Command", classes="mcp-input-label")
                    yield Input(placeholder="https://server/mcp 或 npx", id="mcp-url-cmd", classes="mcp-input-field")
                with Horizontal(classes="mcp-input-row"):
                    yield Static("Args", classes="mcp-input-label")
                    yield Input(placeholder="仅 stdio：-y package-name", id="mcp-args", classes="mcp-input-field")
                with Horizontal(classes="mcp-input-row"):
                    yield Static("Headers / Env", classes="mcp-input-label")
                    yield Input(placeholder="key=value, key2=value2", id="mcp-headers-env", classes="mcp-input-field")
            yield Static("  ↑↓ 移动  ←→ 选择 transport  Tab 下一项  Ctrl+Enter 加载  Esc 关闭", classes="mcp-hint")
            with Horizontal(classes="mcp-button-row"):
                yield Button("加载并应用", id="mcp-add-btn", variant="primary")
                yield Button("关闭", id="mcp-cancel-btn")
            yield Static("状态  等待输入", id="mcp-status", classes="mcp-status")

    def on_mount(self) -> None:
        self._refresh_server_list()
        self.query_one("#mcp-name", Input).focus()

    def _server_row(self, s: dict) -> Text:
        """单行 server 信息，对齐 CommandPalette 的 row_text 风格。"""
        name = s.get("name", "?")
        transport = s.get("transport", "?")
        tool_count = s.get("tool_count", 0)
        health = s.get("health_state", "?")
        mark = "●" if health == "healthy" else "○"
        label = f"  {mark} {name}"
        hint = f"{transport}  {tool_count} tools  {health}"
        pad = max(2, 52 - len(label))
        t = Text()
        t.append(label, style=theme.TEXT_PRIMARY)
        t.append(" " * pad)
        t.append(hint, style=theme.TEXT_DIM)
        return t

    def _refresh_server_list(self) -> None:
        """刷新已加载 server 列表。"""
        try:
            server_list = self.query_one("#mcp-server-list", Vertical)
        except Exception:
            return
        server_list.remove_children()
        if self._mcp_manager is None:
            server_list.mount(Static("  MCP 未启用（设置 AGENT4ML_MCP_ENABLED=true）", classes="mcp-server-row"))
            return
        servers = self._mcp_manager.list_servers()
        if not servers:
            server_list.mount(Static("  （无已加载 server）", classes="mcp-server-row"))
            return
        for s in servers:
            server_list.mount(Static(self._server_row(s), classes="mcp-server-row"))

    def _set_status(self, msg: str) -> None:
        try:
            self.query_one("#mcp-status", Static).update(msg)
        except Exception:
            pass

    def _parse_headers_env(self, raw: str) -> dict[str, str]:
        """解析 key=val,key=val 格式。"""
        result: dict[str, str] = {}
        if not raw.strip():
            return result
        for pair in raw.split(","):
            pair = pair.strip()
            if "=" not in pair:
                continue
            k, _, v = pair.partition("=")
            result[k.strip()] = v.strip()
        return result

    def _get_transport(self) -> str:
        """从 Select 取选中的 transport。"""
        try:
            value = self.query_one("#mcp-transport-select", Select).value
            return value if value in ("stdio", "http", "sse") else "stdio"
        except Exception:
            return "stdio"

    def _validate_and_build_config(self) -> tuple[McpServerConfig | None, str | None]:
        """校验输入并构建 McpServerConfig。返 (config, error)。"""
        name = self.query_one("#mcp-name", Input).value.strip()
        transport = self._get_transport()
        url_cmd = self.query_one("#mcp-url-cmd", Input).value.strip()
        args_raw = self.query_one("#mcp-args", Input).value.strip()
        headers_env_raw = self.query_one("#mcp-headers-env", Input).value.strip()

        if not name:
            return None, "Name 不能为空"
        if not url_cmd:
            return None, "URL/Cmd 不能为空"

        # Name 重名检查
        if self._mcp_manager and name in self._mcp_manager._config.servers:
            return None, f"server 名 '{name}' 已存在，请改名"

        kv = self._parse_headers_env(headers_env_raw)

        if transport == "stdio":
            # Command 存在性仅警告
            cmd_base = url_cmd.split()[0] if " " in url_cmd else url_cmd
            if not shutil.which(cmd_base):
                self._set_status(f"⚠️ Command '{cmd_base}' 不在 PATH（径下有，允许提交）")
            args = shlex.split(args_raw) if args_raw else []
            return McpServerConfig(
                name=name, transport="stdio",
                command=url_cmd, args=args, env=kv,
            ), None
        else:
            # URL 格式阻断
            if not (url_cmd.startswith("http://") or url_cmd.startswith("https://")):
                return None, f"{transport} transport 需以 http:// 或 https:// 开头"
            return McpServerConfig(
                name=name, transport=transport,
                url=url_cmd, headers=kv,
            ), None

    async def _do_add(self) -> None:
        """执行 add_server + 反馈。"""
        try:
            config, error = self._validate_and_build_config()
        except Exception as exc:
            self._set_status(f"❌ 输入解析失败: {exc}")
            return
        if error:
            self._set_status(f"❌ {error}")
            return

        assert config is not None
        if self._mcp_manager is None:
            self._set_status("❌ MCP 未启用（设置 AGENT4ML_MCP_ENABLED=true）")
            return

        self._set_status("connecting...")
        try:
            success = await self._mcp_manager.add_server(config)
        except Exception as exc:
            self._set_status(f"❌ 加载失败: {exc}")
            return

        if success:
            tool_count = sum(
                1 for e in self._mcp_manager.registry._entries.values()
                if e.server_name == config.name
            )
            self._set_status(f"✅ 加载成功，{tool_count} 个工具可用，下轮生效")
            self._refresh_server_list()
            if self._runtime is not None:
                try:
                    self._runtime = self._runtime.reload_mcp_tools()
                except Exception as exc:
                    self._set_status(f"⚠️ 加载成功但 graph 重建失败: {exc}")
                    return
            self.set_timer(1.5, lambda: self.dismiss(None))
        else:
            self._set_status("❌ 加载失败（连接超时或 server 无响应），可修改后重试")

    def action_focus_next(self) -> None:
        """下箭头：按视觉顺序聚焦下一表单项。"""
        self.focus_next()

    def action_focus_previous(self) -> None:
        """上箭头：按视觉顺序聚焦上一表单项。"""
        self.focus_previous()

    async def action_submit_form(self) -> None:
        """Ctrl+Enter：快速提交表单。"""
        await self._do_add()

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "mcp-add-btn":
            await self._do_add()
        elif event.button.id == "mcp-cancel-btn":
            self.dismiss(None)

    def on_click(self, event: events.Click) -> None:
        if event.widget is self:
            self.dismiss(None)

    def action_dismiss_panel(self) -> None:
        self.dismiss(None)
