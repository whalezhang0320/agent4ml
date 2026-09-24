"""E9 /skill evolve|capture|history 命令 + config + SkillManager evolution_manager 单测。"""
from __future__ import annotations

from io import StringIO

from rich.console import Console

from agent4ml.backend.app.cli.commands import CommandContext, _cmd_skill, handle_command
from agent4ml.backend.agents.skill.config import SkillConfig, load_skill_config
from agent4ml.backend.agents.skill.evolution.types import EvolutionRecord


# ── config ──────────────────────────────────────────────


def test_skill_config_evolve_defaults():
    cfg = SkillConfig()
    assert cfg.evolve_enabled is False
    assert cfg.evolve_threshold == 0.3
    assert cfg.evolve_min_selections == 5
    assert cfg.evolve_cooldown_turns == 10
    assert cfg.evolve_mutate_budget == 20
    assert cfg.evolve_max_steps == 5


def test_load_skill_config_evolve_env(monkeypatch):
    monkeypatch.setenv("AGENT4ML_SKILL_EVOLVE_ENABLED", "true")
    monkeypatch.setenv("AGENT4ML_SKILL_EVOLVE_THRESHOLD", "0.4")
    monkeypatch.setenv("AGENT4ML_SKILL_EVOLVE_MAX_STEPS", "8")
    cfg = load_skill_config()
    assert cfg.evolve_enabled is True
    assert cfg.evolve_threshold == 0.4
    assert cfg.evolve_max_steps == 8


def test_load_skill_config_evolve_invalid_uses_default(monkeypatch):
    monkeypatch.setenv("AGENT4ML_SKILL_EVOLVE_THRESHOLD", "abc")
    monkeypatch.setenv("AGENT4ML_SKILL_EVOLVE_MAX_STEPS", "xyz")
    cfg = load_skill_config()
    assert cfg.evolve_threshold == 0.3
    assert cfg.evolve_max_steps == 5


# ── SkillManager evolution_manager ──────────────────────


def test_skill_manager_evolution_field_default_none():
    from agent4ml.backend.agents.skill import SkillManager
    mgr = SkillManager(SkillConfig())
    assert mgr.get_evolution_manager() is None


def test_skill_manager_set_evolution():
    from agent4ml.backend.agents.skill import SkillManager
    mgr = SkillManager(SkillConfig())
    sentinel = object()
    mgr.set_evolution_manager(sentinel)
    assert mgr.get_evolution_manager() is sentinel


# ── /skill 命令 ─────────────────────────────────────────


class _FakeEvolutionManager:
    def __init__(self, record=None, raises=None):
        self._record = record
        self._raises = raises

    def evolve_skill(self, name):
        if self._raises:
            raise self._raises
        return self._record or EvolutionRecord(
            evolution_id="e1", skill_name=name, evolution_type="FIX", trigger="METRIC",
            baseline_id=f"{name}__imp", candidate_id=f"{name}__cand",
            failure_focus="f", mutation_diff="d", eval_score=0.8,
            gate_decision="accept", created_version_id=f"{name}__cand",
        )

    def capture_skill(self, pattern, suggested_name):
        if self._raises:
            raise self._raises
        return self._record or EvolutionRecord(
            evolution_id="e2", skill_name=suggested_name, evolution_type="CAPTURED",
            trigger="CAPTURE", baseline_id=None, candidate_id=f"{suggested_name}__cand",
            failure_focus=pattern, mutation_diff="d", eval_score=0.75,
            gate_decision="accept", created_version_id=f"{suggested_name}__cand",
        )


class _FakeStore:
    def __init__(self, history=None):
        self._history = history or []

    def get_evolution_history(self, name, limit=20):
        return list(self._history)


class _FakeSkillManager:
    def __init__(self, evo=None, store=None):
        self._evo = evo
        self.store = store or _FakeStore()

    def get_evolution_manager(self):
        return self._evo


def _ctx(state, arg, runtime=None) -> CommandContext:
    return CommandContext(
        console=Console(file=StringIO(), force_terminal=False),
        renderer=None, state=state, runtime=runtime, arg=arg,
    )


def _runtime_with_evo(evo=None, store=None):
    mgr = _FakeSkillManager(evo=evo, store=store)
    return type("R", (), {"skill_manager": mgr})()


def _runtime_no_evo():
    mgr = _FakeSkillManager(evo=None)
    return type("R", (), {"skill_manager": mgr})()


# /skill evolve


def test_skill_evolve_success():
    rt = _runtime_with_evo(_FakeEvolutionManager())
    _cmd_skill(_ctx({}, "evolve sv", runtime=rt))  # 不抛


def test_skill_evolve_no_evo_manager():
    rt = _runtime_no_evo()
    _cmd_skill(_ctx({}, "evolve sv", runtime=rt))  # 提示 not enabled


def test_skill_evolve_not_found():
    rt = _runtime_with_evo(_FakeEvolutionManager(raises=ValueError("skill not found: sv")))
    _cmd_skill(_ctx({}, "evolve sv", runtime=rt))  # 提示 not found


def test_skill_evolve_no_arg_usage():
    rt = _runtime_with_evo(_FakeEvolutionManager())
    _cmd_skill(_ctx({}, "evolve", runtime=rt))  # usage


# /skill capture


def test_skill_capture_success():
    rt = _runtime_with_evo(_FakeEvolutionManager())
    _cmd_skill(_ctx({}, "capture 信源核验 new-skill", runtime=rt))  # 不抛


def test_skill_capture_no_evo_manager():
    rt = _runtime_no_evo()
    _cmd_skill(_ctx({}, "capture pattern new-skill", runtime=rt))  # 提示 not enabled


def test_skill_capture_missing_name_usage():
    rt = _runtime_with_evo(_FakeEvolutionManager())
    _cmd_skill(_ctx({}, "capture pattern", runtime=rt))  # usage（缺 name）


def test_skill_capture_no_arg_usage():
    rt = _runtime_with_evo(_FakeEvolutionManager())
    _cmd_skill(_ctx({}, "capture", runtime=rt))  # usage


# /skill history


def test_skill_history_shows():
    history = [
        {"evolution_type": "FIX", "trigger": "METRIC", "eval_score": 0.8,
         "gate_decision": "accept", "timestamp": "2026-07-17T10:00"},
    ]
    store = _FakeStore(history=history)
    rt = _runtime_with_evo(store=store)
    _cmd_skill(_ctx({}, "history sv", runtime=rt))  # 显示历史


def test_skill_history_empty():
    store = _FakeStore(history=[])
    rt = _runtime_with_evo(store=store)
    _cmd_skill(_ctx({}, "history sv", runtime=rt))  # 提示无历史


def test_skill_history_no_manager():
    rt = type("R", (), {"skill_manager": None})()
    _cmd_skill(_ctx({}, "history sv", runtime=rt))  # 提示 not enabled


# handle_command 分发


def test_handle_command_dispatches_evolve():
    rt = _runtime_with_evo(_FakeEvolutionManager())
    console = Console(file=StringIO(), force_terminal=False)
    handle_command("/skill evolve sv", console, None, {}, rt)  # 不抛


def test_handle_command_dispatches_capture():
    rt = _runtime_with_evo(_FakeEvolutionManager())
    console = Console(file=StringIO(), force_terminal=False)
    handle_command("/skill capture pattern new-skill", console, None, {}, rt)  # 不抛
