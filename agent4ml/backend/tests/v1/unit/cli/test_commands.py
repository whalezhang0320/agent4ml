"""commands.py / 命令测试：/expert /default /report pending 标记。"""

from io import StringIO

from rich.console import Console

from agent4ml.backend.app.cli.commands import handle_command


def _make_console() -> Console:
    return Console(file=StringIO(), width=120)


def _make_state() -> dict:
    return {"pending_expert_mode": None, "pending_report": None}


def test_expert_sets_pending_true() -> None:
    state = _make_state()
    handle_command("/expert", _make_console(), None, state, runtime=None)
    assert state["pending_expert_mode"] is True


def test_default_sets_pending_false() -> None:
    state = _make_state()
    handle_command("/default", _make_console(), None, state, runtime=None)
    assert state["pending_expert_mode"] is False


def test_report_no_topic_sets_pending_empty_string() -> None:
    """/report 无 topic → pending_report=""（空字符串表无 topic，区别于 None 未设）。"""
    state = _make_state()
    handle_command("/report", _make_console(), None, state, runtime=None)
    assert state["pending_report"] == ""


def test_report_with_topic_sets_pending_topic() -> None:
    state = _make_state()
    handle_command("/report 天气调研", _make_console(), None, state, runtime=None)
    assert state["pending_report"] == "天气调研"


def test_help_does_not_set_pending() -> None:
    state = _make_state()
    handle_command("/help", _make_console(), None, state, runtime=None)
    assert state["pending_expert_mode"] is None
    assert state["pending_report"] is None


def test_exit_returns_true() -> None:
    state = _make_state()
    assert handle_command("/exit", _make_console(), None, state, runtime=None) is True


def test_quit_returns_true() -> None:
    state = _make_state()
    assert handle_command("/quit", _make_console(), None, state, runtime=None) is True


def test_unknown_command_returns_false() -> None:
    state = _make_state()
    assert handle_command("/unknown", _make_console(), None, state, runtime=None) is False
