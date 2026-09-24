"""B11 /skill 命令单测 — 激活/清除/list/usage + handle_command 分发。"""
from __future__ import annotations

from io import StringIO

from rich.console import Console

from agent4ml.backend.app.cli.commands import CommandContext, _cmd_skill, handle_command


def _ctx(state: dict, arg: str, runtime=None) -> CommandContext:
    return CommandContext(
        console=Console(file=StringIO(), force_terminal=False),
        renderer=None,
        state=state,
        runtime=runtime,
        arg=arg,
    )


def test_skill_activate_sets_override():
    state: dict = {}
    _cmd_skill(_ctx(state, "source-verification"))
    assert state["skill_override"] == ["source-verification"]


def test_skill_off_clears_override():
    state = {"skill_override": ["source-verification"]}
    _cmd_skill(_ctx(state, "off"))
    assert state["skill_override"] == []


def test_skill_no_arg_shows_usage_no_crash():
    state = {"skill_override": ["x"]}
    _cmd_skill(_ctx(state, ""))  # 不抛，打印 usage + current


def test_skill_list_no_manager_no_crash():
    state: dict = {}
    rt = type("R", (), {"skill_manager": None})()
    _cmd_skill(_ctx(state, "list", runtime=rt))  # 打印 "not enabled"


def test_skill_list_with_manager_shows_skills():
    state: dict = {}
    mgr = type("M", (), {
        "list_skills": lambda self: [{
            "name": "source-verification",
            "effective_rate": 0.5,
            "total_selections": 4,
            "allowed_tools": ["web_search", "browse_page"],
        }],
    })()
    rt = type("R", (), {"skill_manager": mgr})()
    _cmd_skill(_ctx(state, "list", runtime=rt))  # 打印 skill，不抛


def test_handle_command_dispatches_skill_activate():
    state: dict = {}
    console = Console(file=StringIO(), force_terminal=False)
    handle_command("/skill my-skill", console, None, state, None)
    assert state["skill_override"] == ["my-skill"]


def test_handle_command_dispatches_skill_off():
    state = {"skill_override": ["x"]}
    console = Console(file=StringIO(), force_terminal=False)
    handle_command("/skill off", console, None, state, None)
    assert state["skill_override"] == []


def test_skill_override_persists_across_activations():
    state: dict = {}
    _cmd_skill(_ctx(state, "a"))
    assert state["skill_override"] == ["a"]
    _cmd_skill(_ctx(state, "b"))  # 覆盖前一个
    assert state["skill_override"] == ["b"]


# ── enable/disable 运行时持久 ─────────────────────────────


class _FakeStore:
    def __init__(self, records):
        self._records = {r.skill_id: r for r in records}
        self.enabled_calls: list[tuple[str, bool]] = []

    def get_active(self, name):
        for r in self._records.values():
            if r.name == name and r.is_active:
                return r
        return None

    def set_enabled(self, skill_id, enabled):
        if skill_id not in self._records:
            return False
        self._records[skill_id] = self._records[skill_id]  # id
        self.enabled_calls.append((skill_id, enabled))
        return True


class _FakeManager:
    def __init__(self, store):
        self.store = store

    def list_skills(self):
        return []


def _skill_record(name="source-verification", skill_id="sv__builtin"):
    from agent4ml.backend.agents.skill.types import SkillRecord, SkillLineage
    return SkillRecord(
        skill_id=skill_id, name=name, path="/p", content_hash="h",
        lineage=SkillLineage(generation=0, origin="BUILTIN"),
        description="d", allowed_tools=("web_search",),
    )


def _ctx_with_mgr(state, arg, records):
    mgr = _FakeManager(_FakeStore(records))
    rt = type("R", (), {"skill_manager": mgr})()
    return CommandContext(
        console=Console(file=StringIO(), force_terminal=False),
        renderer=None, state=state, runtime=rt, arg=arg,
    )


def test_skill_disable_persists():
    state: dict = {}
    rec = _skill_record()
    ctx = _ctx_with_mgr(state, "disable source-verification", [rec])
    _cmd_skill(ctx)
    assert ctx.runtime.skill_manager.store.enabled_calls == [("sv__builtin", False)]


def test_skill_enable_persists():
    state: dict = {}
    rec = _skill_record()
    ctx = _ctx_with_mgr(state, "enable source-verification", [rec])
    _cmd_skill(ctx)
    assert ctx.runtime.skill_manager.store.enabled_calls == [("sv__builtin", True)]


def test_skill_disable_not_found():
    state: dict = {}
    ctx = _ctx_with_mgr(state, "disable nonexistent", [])  # 空 store
    _cmd_skill(ctx)  # 提示 not found，不抛
    assert ctx.runtime.skill_manager.store.enabled_calls == []


def test_skill_off_does_not_disable():
    """off 清 override，不调 set_enabled（skill 仍 enabled）。"""
    state = {"skill_override": ["source-verification"]}
    rec = _skill_record()
    ctx = _ctx_with_mgr(state, "off", [rec])
    _cmd_skill(ctx)
    assert state["skill_override"] == []
    assert ctx.runtime.skill_manager.store.enabled_calls == []  # off 不禁用


def test_skill_no_manager_enable_silent():
    state: dict = {}
    rt = type("R", (), {"skill_manager": None})()
    ctx = CommandContext(
        console=Console(file=StringIO(), force_terminal=False),
        renderer=None, state=state, runtime=rt, arg="enable x",
    )
    _cmd_skill(ctx)  # 提示 not enabled，不抛


# ── install 装载新 skill ──────────────────────────────────


def _install_mgr():
    """带 load_startup 计数的 FakeManager。"""
    mgr = _FakeManager(_FakeStore([]))
    mgr.load_count = 0

    def _load(*a, **k):
        mgr.load_count += 1

    mgr.load_startup = _load
    return mgr


def test_skill_install_success(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    src = tmp_path / "my-src"
    src.mkdir()
    (src / "SKILL.md").write_text(
        "---\nname: my-skill\ndescription: d\n---\nbody", encoding="utf-8",
    )
    mgr = _install_mgr()
    rt = type("R", (), {"skill_manager": mgr})()
    ctx = CommandContext(
        console=Console(file=StringIO(), force_terminal=False),
        renderer=None, state={}, runtime=rt, arg=f"install {src} my-skill",
    )
    _cmd_skill(ctx)
    assert (tmp_path / "skills" / "my-skill" / "SKILL.md").exists()
    assert mgr.load_count == 1  # re-discover 触发


def test_skill_install_default_name_from_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    src = tmp_path / "auto-named"
    src.mkdir()
    (src / "SKILL.md").write_text("---\nname: auto-named\ndescription: d\n---\nb", encoding="utf-8")
    mgr = _install_mgr()
    rt = type("R", (), {"skill_manager": mgr})()
    ctx = CommandContext(
        console=Console(file=StringIO(), force_terminal=False),
        renderer=None, state={}, runtime=rt, arg=f"install {src}",  # 无 name
    )
    _cmd_skill(ctx)
    assert (tmp_path / "skills" / "auto-named" / "SKILL.md").exists()


def test_skill_install_invalid_name_rejected(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    src = tmp_path / "src"
    src.mkdir()
    (src / "SKILL.md").write_text("---\nname: x\ndescription: d\n---\nb", encoding="utf-8")
    mgr = _install_mgr()
    rt = type("R", (), {"skill_manager": mgr})()
    ctx = CommandContext(
        console=Console(file=StringIO(), force_terminal=False),
        renderer=None, state={}, runtime=rt, arg=f'install {src} "../escape"',
    )
    _cmd_skill(ctx)  # ValueError 捕获，提示
    assert not (tmp_path / "skills" / "escape").exists()  # 未拷贝
    assert not (tmp_path / "escape").exists()


def test_skill_install_path_not_found(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    mgr = _install_mgr()
    rt = type("R", (), {"skill_manager": mgr})()
    ctx = CommandContext(
        console=Console(file=StringIO(), force_terminal=False),
        renderer=None, state={}, runtime=rt, arg=f"install {tmp_path / 'nonexistent'}",
    )
    _cmd_skill(ctx)  # FileNotFoundError 捕获，提示
    assert mgr.load_count == 0  # 未触发 re-discover


def test_skill_install_no_arg_usage():
    state: dict = {}
    mgr = _install_mgr()
    rt = type("R", (), {"skill_manager": mgr})()
    ctx = CommandContext(
        console=Console(file=StringIO(), force_terminal=False),
        renderer=None, state=state, runtime=rt, arg="install",  # 无 path
    )
    _cmd_skill(ctx)  # 提示 usage，不抛
    assert mgr.load_count == 0
