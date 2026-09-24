"""SettingsScreen — Ctrl+B 配置面板（Textual 模态 Screen）。

显示已配置 provider 的 API Key / Base URL / Model 输入框 + 功能开关。
保存时直写 .env 文件 + load_dotenv reload，变更下轮生效（与 /model 一致）。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from textual import events
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Vertical, VerticalScroll
from textual.screen import Screen
from textual.widgets import Button, Input, Static, Switch

from agent4ml.backend.app.tui import theme


class SettingsScreen(Screen):
    """配置面板 — Ctrl+B 打开，编辑 provider 配置直写 .env。"""

    BINDINGS = [
        Binding("escape", "app.pop_screen", "Close", show=False),
        Binding("ctrl+s", "save", "Save", show=True),
    ]

    DEFAULT_CSS = f"""
    SettingsScreen {{
        align: center middle;
    }}
    #settings-box {{
        width: 90;
        max-width: 90%;
        height: auto;
        max-height: 80%;
        background: {theme.SURFACE};
        border: tall {theme.BORDER};
        padding: 1 2;
    }}
    #settings-title {{
        color: {theme.ACCENT_BRAND};
        text-align: center;
        margin-bottom: 1;
    }}
    .provider-section {{
        margin-bottom: 1;
        padding: 0 1;
        border-bottom: solid {theme.BORDER};
    }}
    .provider-name {{
        color: {theme.ACCENT_ASSISTANT};
        margin-bottom: 0;
    }}
    .field-row {{
        height: 3;
        margin: 0 0 0 2;
    }}
    .field-label {{
        color: {theme.TEXT_SECONDARY};
        width: 12;
        height: 1;
    }}
    .field-input {{
        width: 1fr;
        height: 1;
        border: tall {theme.BORDER};
        background: {theme.BG};
    }}
    .toggle-row {{
        height: 1;
        margin: 0 0 0 2;
    }}
    #settings-actions {{
        height: 3;
        align-horizontal: right;
        margin-top: 1;
    }}
    #save-btn {{
        background: {theme.ACCENT_USER};
        color: white;
        border: none;
        height: 3;
        min-width: 16;
    }}
    #save-btn:hover {{
        background: {theme.BORDER_FOCUS};
    }}
    """

    def __init__(self, env_path: Path) -> None:
        super().__init__()
        self._env_path = env_path
        self._fields: dict[str, Input] = {}   # env_key → Input
        self._toggles: dict[str, Switch] = {}  # env_key → Switch

    def compose(self) -> ComposeResult:
        from agent4ml.backend.agents.config.provider_profile import PROVIDER_PROFILES
        import os

        with Container(id="settings-box"):
            yield Static("Settings  (Ctrl+S save · Esc close)", id="settings-title")
            with VerticalScroll():
                # Provider 配置区：只显示有 API key 或 no_key 的 provider
                for profile in PROVIDER_PROFILES:
                    if profile.name == "fake":
                        continue
                    api_key = os.environ.get(profile.env_key, "")
                    base_url = os.environ.get(profile.env_base_url, "") or (profile.default_base_url or "")
                    model = os.environ.get(profile.env_model, "") or profile.default_model
                    # 跳过未配置且非默认的 provider（减少噪声）
                    if not api_key and not profile.is_default and not profile.no_key_required:
                        continue

                    with Vertical(classes="provider-section"):
                        yield Static(f"  {profile.name}", classes="provider-name")
                        # API Key
                        yield Static(f"    API Key", classes="field-label")
                        key_input = Input(
                            value=api_key, password=True,
                            id=f"key-{profile.name}", classes="field-input",
                        )
                        self._fields[profile.env_key] = key_input
                        yield key_input
                        # Base URL
                        yield Static(f"    Base URL", classes="field-label")
                        url_input = Input(
                            value=base_url,
                            id=f"url-{profile.name}", classes="field-input",
                        )
                        self._fields[profile.env_base_url] = url_input
                        yield url_input
                        # Model
                        yield Static(f"    Model", classes="field-label")
                        model_input = Input(
                            value=model,
                            id=f"model-{profile.name}", classes="field-input",
                        )
                        self._fields[profile.env_model] = model_input
                        yield model_input

                # 功能开关区
                with Vertical(classes="provider-section"):
                    yield Static("  Features", classes="provider-name")
                    for label, env_key in [
                        ("Skill", "AGENT4ML_SKILL_ENABLED"),
                        ("MCP", "AGENT4ML_MCP_ENABLED"),
                    ]:
                        val = os.environ.get(env_key, "false").lower() == "true"
                        sw = Switch(value=val, id=f"toggle-{env_key}")
                        self._toggles[env_key] = sw
                        with Container(classes="toggle-row"):
                            yield Static(f"    {label}", classes="field-label")
                            yield sw

            with Container(id="settings-actions"):
                yield Button("Save (Ctrl+S)", id="save-btn", variant="primary")

    def on_mount(self) -> None:
        self.query_one("#settings-box", Container).focus()

    def on_click(self, event: events.Click) -> None:
        if event.widget is self:
            self.app.pop_screen()

    def action_save(self) -> None:
        self._do_save()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "save-btn":
            self._do_save()

    def _do_save(self) -> None:
        """收集输入框值 → 直写 .env → load_dotenv reload → 关闭面板。"""
        from dotenv import load_dotenv
        from agent4ml.backend.app.cli.setup_wizard import update_env_file

        overrides: dict[str, str] = {}
        for env_key, widget in self._fields.items():
            val = widget.value.strip()
            if val:
                overrides[env_key] = val
        for env_key, widget in self._toggles.items():
            overrides[env_key] = "true" if widget.value else "false"

        update_env_file(self._env_path, overrides)
        load_dotenv(self._env_path, override=True)

        self.app.pop_screen()
