"""AppRuntime.switch_expert_mode 测试：模式切换保留 thread 连续性 + 精准重建。"""

from pathlib import Path
from types import SimpleNamespace

from agent4ml.backend.app.bootstrap import AppRuntime


def _fake_runtime(tmp_path: Path) -> AppRuntime:
    """构造最小 AppRuntime 实例（fakes），用于测 switch_expert_mode 保留/重建逻辑。"""
    from agent4ml.backend.agents.config.loader import load_config

    config = load_config(cli_overrides={"logs_root": str(tmp_path / "logs")})
    journal = SimpleNamespace(events=[], append=lambda et, p: journal.events.append((et, p)))
    return AppRuntime(
        config=config,
        capability_registry=SimpleNamespace(),  # sentinel
        run_manager=SimpleNamespace(),
        researcher_model_name="fake-model",
        thread_id="thread-test-123",
        thread_dir=tmp_path,
        thread_journal=journal,
        leader_agent=SimpleNamespace(),  # sentinel 旧 leader
    )


def test_switch_expert_mode_preserves_thread_id(tmp_path, monkeypatch) -> None:
    runtime = _fake_runtime(tmp_path)

    captured = {}

    def fake_make_lead_agent(expert_mode=False, capability_registry=None, **kw):
        captured["expert_mode"] = expert_mode
        captured["registry"] = capability_registry
        return SimpleNamespace(new_leader=True)

    monkeypatch.setattr(
        "agent4ml.backend.app.bootstrap.make_lead_agent", fake_make_lead_agent
    )

    new_runtime = runtime.switch_expert_mode(expert_mode=True)

    assert new_runtime.thread_id == "thread-test-123"  # 保留
    assert new_runtime.thread_dir == tmp_path  # 保留
    assert new_runtime.thread_journal is runtime.thread_journal  # 保留同一实例
    assert new_runtime.capability_registry is runtime.capability_registry  # 保留
    assert new_runtime.researcher_model_name == "fake-model"  # 保留


def test_switch_expert_mode_rebuilds_leader_agent(tmp_path, monkeypatch) -> None:
    runtime = _fake_runtime(tmp_path)

    new_leader = SimpleNamespace(rebuilt=True)
    monkeypatch.setattr(
        "agent4ml.backend.app.bootstrap.make_lead_agent",
        lambda expert_mode=False, capability_registry=None, **kw: new_leader,
    )

    new_runtime = runtime.switch_expert_mode(expert_mode=True)

    assert new_runtime.leader_agent is new_leader  # 重建
    assert new_runtime.leader_agent is not runtime.leader_agent  # 非旧实例


def test_switch_expert_mode_rebuilds_config_and_run_manager(tmp_path, monkeypatch) -> None:
    runtime = _fake_runtime(tmp_path)
    assert runtime.config.runtime.expert_mode is False  # 初始 default

    monkeypatch.setattr(
        "agent4ml.backend.app.bootstrap.make_lead_agent",
        lambda expert_mode=False, capability_registry=None, **kw: SimpleNamespace(),
    )

    new_runtime = runtime.switch_expert_mode(expert_mode=True)

    assert new_runtime.config.runtime.expert_mode is True  # 重建 config
    assert new_runtime.run_manager is not runtime.run_manager  # 重建 run_manager


def test_switch_expert_mode_passes_expert_mode_to_factory(tmp_path, monkeypatch) -> None:
    runtime = _fake_runtime(tmp_path)

    captured = {}

    def fake_make_lead_agent(expert_mode=False, capability_registry=None, **kw):
        captured["expert_mode"] = expert_mode
        return SimpleNamespace()

    monkeypatch.setattr(
        "agent4ml.backend.app.bootstrap.make_lead_agent", fake_make_lead_agent
    )

    runtime.switch_expert_mode(expert_mode=True)
    assert captured["expert_mode"] is True

    runtime.switch_expert_mode(expert_mode=False)
    assert captured["expert_mode"] is False


def test_switch_expert_mode_logs_mode_switched_event(tmp_path, monkeypatch) -> None:
    runtime = _fake_runtime(tmp_path)
    monkeypatch.setattr(
        "agent4ml.backend.app.bootstrap.make_lead_agent",
        lambda expert_mode=False, capability_registry=None, **kw: SimpleNamespace(),
    )

    runtime.switch_expert_mode(expert_mode=True)

    events = runtime.thread_journal.events
    assert any(et == "mode.switched" for et, _ in events)
    switched_event = next(p for et, p in events if et == "mode.switched")
    assert switched_event["expert_mode"] is True
    assert switched_event["thread_id"] == "thread-test-123"


def test_switch_expert_mode_returns_new_instance(tmp_path, monkeypatch) -> None:
    runtime = _fake_runtime(tmp_path)
    monkeypatch.setattr(
        "agent4ml.backend.app.bootstrap.make_lead_agent",
        lambda expert_mode=False, capability_registry=None, **kw: SimpleNamespace(),
    )

    new_runtime = runtime.switch_expert_mode(expert_mode=True)

    assert new_runtime is not runtime  # 新实例


def test_switch_expert_mode_passes_context_governance(tmp_path, monkeypatch) -> None:
    """switch_expert_mode 必须把 context_governance 透传给 make_lead_agent——

    之前没传，_build_middlewares 看到 None 跳过整个治理层，StrategyMiddleware 不挂，
    切换 expert 后 budget/fraction/压缩全部失效（D12 同款故障）。
    """
    runtime = _fake_runtime(tmp_path)

    captured = {}

    def fake_make_lead_agent(expert_mode=False, capability_registry=None, **kw):
        captured["context_governance"] = kw.get("context_governance")
        return SimpleNamespace()

    monkeypatch.setattr(
        "agent4ml.backend.app.bootstrap.make_lead_agent", fake_make_lead_agent
    )

    runtime.switch_expert_mode(expert_mode=True)

    cg = captured.get("context_governance")
    assert cg is not None, "context_governance 必须透传给 make_lead_agent"
    # 切换后的 externalize_dir 应已被 _resolve_relative_paths 锚到项目根（绝对路径）
    ext_dir = cg.params.get("externalize_dir", "")
    assert Path(ext_dir).is_absolute(), f"切换后 externalize_dir 应为绝对路径，实际 {ext_dir}"
