"""HelpRequestScreen — modal panel for displaying help requests and receiving user guidance."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Static

from agent4ml.backend.app.tui import theme


class HelpRequestScreen(ModalScreen[str | None]):
    """Display SituationReport and receive user's choice or custom guidance."""

    BINDINGS = [
        Binding("escape", "abandon", "Abandon", show=False),
    ]

    DEFAULT_CSS = f"""
    HelpRequestScreen {{
        align: center middle;
        background: $background 0%;
    }}
    HelpRequestScreen > #help-box {{
        width: 80;
        max-width: 90%;
        height: auto;
        max-height: 80%;
        background: {theme.SURFACE};
        border: solid {theme.ACCENT_WARN};
        padding: 1 2;
    }}
    HelpRequestScreen #help-report {{
        height: auto;
        max-height: 20;
        color: {theme.TEXT_PRIMARY};
        margin: 0 0 1 0;
    }}
    HelpRequestScreen #help-input {{
        height: 3;
        border: tall {theme.BORDER};
        background: {theme.BG};
        color: {theme.TEXT_PRIMARY};
        padding: 0 1;
    }}
    HelpRequestScreen #help-input:focus {{
        border: tall {theme.BORDER_FOCUS};
    }}
    HelpRequestScreen #help-actions {{
        height: 3;
        align: right middle;
        margin: 1 0 0 0;
    }}
    """

    def __init__(self, report_text: str) -> None:
        super().__init__()
        self._report_text = report_text

    def compose(self) -> ComposeResult:
        with Container(id="help-box"):
            yield Static(self._report_text, id="help-report")
            yield Input(placeholder="Type option number or your guidance...", id="help-input")
            with Horizontal(id="help-actions"):
                yield Button("Abandon", id="help-abandon")
                yield Button("Send", variant="primary", id="help-send")

    def on_mount(self) -> None:
        self.query_one("#help-input", Input).focus()

    def action_abandon(self) -> None:
        self.dismiss(None)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "help-input":
            self._submit()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "help-send":
            self._submit()
        elif event.button.id == "help-abandon":
            self.action_abandon()

    def _submit(self) -> None:
        text = self.query_one("#help-input", Input).value.strip()
        self.dismiss(text if text else None)
