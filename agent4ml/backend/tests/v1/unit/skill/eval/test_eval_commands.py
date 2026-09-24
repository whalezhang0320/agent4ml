"""L3-E9 单测：/skill health + /skill eval-history 命令。

验证命令解析 + 输出格式 + 无 skill manager 时降级。
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from agent4ml.backend.agents.skill.config import SkillConfig
from agent4ml.backend.agents.skill.eval.types import SkillJudgment
from agent4ml.backend.agents.skill.store import SQLiteSkillStore
from agent4ml.backend.agents.skill import SkillManager
from agent4ml.backend.agents.skill.types import SkillLineage, SkillRecord


@pytest.fixture()
def mgr(tmp_path: Path) -> SkillManager:
    cfg = SkillConfig(
        enabled=True,
        db_path=str(tmp_path / "test_cmd.db"),
        skill_dirs=(str(tmp_path / "skills"),),
        include_builtin=False,
    )
    (tmp_path / "skills").mkdir(exist_ok=True)
    m = SkillManager(cfg)
    # 注册一个测试 skill
    skill_md = tmp_path / "skills" / "test-skill" / "SKILL.md"
    skill_md.parent.mkdir(parents=True, exist_ok=True)
    skill_md.write_text("---\nname: test-skill\ndescription: test\n---\n# Test\nbody", encoding="utf-8")
    m.load_startup()
    return m


class _Ctx:
    """Mock CommandContext。"""
    def __init__(self, arg: str, runtime):
        self.arg = arg
        self.runtime = runtime
        self.console = MagicMock()


def _runtime_with_mgr(mgr):
    rt = MagicMock()
    rt.skill_manager = mgr
    return rt


# ── /skill health ──────────────────────────────────────

def test_health_specific_skill(mgr: SkillManager):
    from agent4ml.backend.app.cli.commands import _cmd_skill
    ctx = _Ctx("health test-skill", _runtime_with_mgr(mgr))
    _cmd_skill(ctx)
    output = " ".join(str(c) for c in ctx.console.print.call_args_list)
    assert "test-skill" in output


def test_health_all_skills(mgr: SkillManager):
    from agent4ml.backend.app.cli.commands import _cmd_skill
    ctx = _Ctx("health", _runtime_with_mgr(mgr))
    _cmd_skill(ctx)
    output = " ".join(str(c) for c in ctx.console.print.call_args_list)
    assert "test-skill" in output


def test_health_skill_not_found(mgr: SkillManager):
    from agent4ml.backend.app.cli.commands import _cmd_skill
    ctx = _Ctx("health nonexistent", _runtime_with_mgr(mgr))
    _cmd_skill(ctx)
    output = " ".join(str(c) for c in ctx.console.print.call_args_list)
    assert "not found" in output.lower()


def test_health_no_skill_manager():
    from agent4ml.backend.app.cli.commands import _cmd_skill
    rt = MagicMock()
    rt.skill_manager = None
    ctx = _Ctx("health", rt)
    _cmd_skill(ctx)
    output = " ".join(str(c) for c in ctx.console.print.call_args_list)
    assert "not enabled" in output.lower()


# ── /skill eval-history ────────────────────────────────

def test_eval_history_shows_judgments(mgr: SkillManager):
    """有 judgment → 显示。"""
    rec = mgr.store.get_active("test-skill")
    mgr.store.save_judgment(SkillJudgment(
        judgment_id="j1", skill_id=rec.skill_id, skill_name="test-skill",
        task_id="t1", skill_applied=True, deviation_note="ok",
        timestamp="2026-07-21T10:00:00Z",
    ))
    from agent4ml.backend.app.cli.commands import _cmd_skill
    ctx = _Ctx("eval-history test-skill", _runtime_with_mgr(mgr))
    _cmd_skill(ctx)
    output = " ".join(str(c) for c in ctx.console.print.call_args_list)
    assert "applied" in output.lower()
    assert "t1" in output


def test_eval_history_empty(mgr: SkillManager):
    """无 judgment → 提示。"""
    from agent4ml.backend.app.cli.commands import _cmd_skill
    ctx = _Ctx("eval-history test-skill", _runtime_with_mgr(mgr))
    _cmd_skill(ctx)
    output = " ".join(str(c) for c in ctx.console.print.call_args_list)
    assert "no eval history" in output.lower()


def test_eval_history_skill_not_found(mgr: SkillManager):
    from agent4ml.backend.app.cli.commands import _cmd_skill
    ctx = _Ctx("eval-history nonexistent", _runtime_with_mgr(mgr))
    _cmd_skill(ctx)
    output = " ".join(str(c) for c in ctx.console.print.call_args_list)
    assert "not found" in output.lower()


def test_eval_history_no_arg(mgr: SkillManager):
    from agent4ml.backend.app.cli.commands import _cmd_skill
    ctx = _Ctx("eval-history", _runtime_with_mgr(mgr))
    _cmd_skill(ctx)
    output = " ".join(str(c) for c in ctx.console.print.call_args_list)
    assert "usage" in output.lower()
