"""S4 SlashCommandCompleter 单测 — /skill <name> 补全 + 子命令优先 + 命令名补全。"""
from __future__ import annotations

from prompt_toolkit.document import Document

from agent4ml.backend.app.cli.command_completer import SlashCommandCompleter
from agent4ml.backend.app.cli.commands import get_registry


def _completions(completer, text, cursor=None):
    if cursor is None:
        cursor = len(text)
    doc = Document(text, cursor_position=cursor)
    return [c.text for c in completer.get_completions(doc, None)]


def test_skill_arg_completes_skill_name():
    c = SlashCommandCompleter(get_registry(), skill_provider=lambda: ["source-verification", "english-help"])
    assert "source-verification" in _completions(c, "/skill so")
    assert "english-help" not in _completions(c, "/skill so")  # 不匹配 so 前缀


def test_skill_subcommand_priority_over_skill_name():
    """子命令命中时只补子命令，不补 skill 名（enable 优先于 english）。"""
    c = SlashCommandCompleter(get_registry(), skill_provider=lambda: ["english-help"])
    result = _completions(c, "/skill en")
    assert "enable" in result
    assert "english-help" not in result  # 子命令命中，不补 skill 名


def test_skill_off_matches_subcommand_only():
    c = SlashCommandCompleter(get_registry(), skill_provider=lambda: ["off-site-tool"])
    result = _completions(c, "/skill off")
    assert "off" in result
    assert "off-site-tool" not in result  # 子命令优先


def test_skill_empty_arg_shows_subcommands_not_skills():
    c = SlashCommandCompleter(get_registry(), skill_provider=lambda: ["source-verification"])
    result = _completions(c, "/skill ")
    # 空参 → 全 5 子命令命中，不补 skill 名
    assert set(result) == {"list", "off", "enable", "disable", "install"}
    assert "source-verification" not in result


def test_skill_no_provider_no_skill_completion():
    """无 skill_provider → /skill so 无子命令命中，无补全（不崩）。"""
    c = SlashCommandCompleter(get_registry())  # 无 provider
    assert _completions(c, "/skill so") == []
    # 子命令仍补
    assert "enable" in _completions(c, "/skill en")


def test_skill_provider_exception_silent():
    """provider 抛异常 → 静默返空 skill（不崩）。"""
    def _boom():
        raise RuntimeError("boom")
    c = SlashCommandCompleter(get_registry(), skill_provider=_boom)
    assert _completions(c, "/skill so") == []  # 不抛


def test_command_name_completion_unchanged():
    c = SlashCommandCompleter(get_registry(), skill_provider=lambda: ["x"])
    result = _completions(c, "/hel")
    assert "/help" in result


def test_non_skill_command_no_skill_completion():
    """非 /skill 命令的参数不触发 skill 名补全。"""
    c = SlashCommandCompleter(get_registry(), skill_provider=lambda: ["source-verification"])
    # /prompt list —— word="list" 非 /，但 stripped 不以 /skill 开头 → 无补全
    assert _completions(c, "/prompt list") == []


def test_skill_full_name_completes():
    c = SlashCommandCompleter(get_registry(), skill_provider=lambda: ["source-verification"])
    result = _completions(c, "/skill source-verification")
    assert "source-verification" in result
