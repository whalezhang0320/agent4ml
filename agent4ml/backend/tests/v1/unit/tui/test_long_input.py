from __future__ import annotations

import asyncio

from textual.app import App
from textual.widgets import TextArea

from agent4ml.backend.app.tui.app import (
    ConversationInput,
    DraftEditor,
    DraftState,
    DraftSummary,
    InputRow,
)


class _WelcomeInputApp(App):
    CSS = """
    _WelcomeInputApp {
        align: center middle;
        height: 1fr;
        padding: 0 4;
    }
    InputRow {
        width: 100%;
        height: auto;
        min-height: 3;
        align: center middle;
    }
    #welcome-input {
        width: 64;
        height: 3;
        border: tall blue;
        padding: 0 1;
    }
    """

    def compose(self):
        yield InputRow(ConversationInput(id="welcome-input"))


def test_welcome_input_keeps_visible_viewport_after_typing() -> None:
    async def check() -> None:
        app = _WelcomeInputApp()
        async with app.run_test(size=(120, 40)) as pilot:
            input_widget = app.query_one("#welcome-input", ConversationInput)
            await pilot.press("h", "e", "l", "l", "o")
            await pilot.pause()
            assert input_widget.text == "hello"
            assert input_widget.size.height >= 1

    asyncio.run(check())


class _CopyAndUndoApp(App):
    BINDINGS = [("ctrl+c", "quit", "Quit")]

    def compose(self):
        yield ConversationInput("original", id="input")


def test_input_ctrl_z_and_ctrl_y_use_textarea_history() -> None:
    async def check() -> None:
        app = _CopyAndUndoApp()
        async with app.run_test() as pilot:
            input_widget = app.query_one("#input", ConversationInput)
            await pilot.press("end", "x", "ctrl+z")
            await pilot.pause()
            assert input_widget.text == "original"
            await pilot.press("ctrl+y")
            await pilot.pause()
            assert input_widget.text == "originalx"

    asyncio.run(check())


def test_input_ctrl_c_copies_selection_without_quitting() -> None:
    async def check() -> None:
        app = _CopyAndUndoApp()
        async with app.run_test() as pilot:
            input_widget = app.query_one("#input", ConversationInput)
            input_widget.select_all()
            assert input_widget.selected_text == "original"
            await pilot.press("ctrl+c")
            await pilot.pause()
            assert app.clipboard == "original"

    asyncio.run(check())


def test_draft_state_preserves_multiline_text_and_detects_summary_mode() -> None:
    draft = DraftState("\n".join(f"line {index}" for index in range(11)))

    assert draft.text.splitlines()[-1] == "line 10"
    assert draft.line_count == 11
    assert draft.character_count == len(draft.text)
    assert draft.is_summarized


def test_draft_state_summarizes_long_single_line() -> None:
    draft = DraftState("x" * 801)

    assert draft.line_count == 1
    assert draft.is_summarized


def test_draft_summary_displays_statistics_and_preview() -> None:
    draft = DraftState("First line\n" + "x" * 900)

    summary = DraftSummary(draft, id="summary")

    assert "2 lines" in str(summary.render())
    assert "911 chars" in str(summary.render())
    assert "First line" in str(summary.render())


def test_draft_editor_cancel_returns_current_text() -> None:
    async def check() -> None:
        result: list[str | None] = []
        app = App()
        async with app.run_test() as pilot:
            editor = DraftEditor("original")
            app.push_screen(editor, result.append)
            await pilot.pause()
            input_widget = editor.query_one("#draft-editor-input", TextArea)
            input_widget.text = "edited"
            editor.action_cancel()
            await pilot.pause()
            assert result == ["edited"]

    asyncio.run(check())
