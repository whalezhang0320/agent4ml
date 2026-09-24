"""bootstrap._resolve_relative_paths 单测：externalize_dir 相对路径锚到项目根。"""

from __future__ import annotations

from pathlib import Path

from agent4ml.backend.agents.config.loader import load_config
from agent4ml.backend.app.bootstrap import _PROJECT_ROOT, _resolve_relative_paths


def test_externalize_dir_relative_resolved_to_project_root() -> None:
    """默认 '.agent4ml/externalized' 相对路径 → 锚到 _PROJECT_ROOT 下绝对路径。"""
    config = load_config()
    resolved = _resolve_relative_paths(config)
    ext_dir = resolved.context_governance.params["externalize_dir"]
    p = Path(ext_dir)
    assert p.is_absolute(), f"应是绝对路径，实际 {ext_dir}"
    assert p == (_PROJECT_ROOT / ".agent4ml" / "externalized").resolve()
    # 原始 config 不被修改（frozen dataclass + 不可变语义）
    assert "externalize_dir" not in config.context_governance.params


def test_externalize_dir_absolute_preserved() -> None:
    """已是绝对路径的 externalize_dir 不被改写（用户自定义位置时尊重）。"""
    custom = Path("/tmp/my_externalized").resolve()
    config = load_config()
    # frozen dataclass，用 replace 构造带自定义 params 的 config
    from dataclasses import replace
    cfg_with_custom = replace(
        config,
        context_governance=replace(
            config.context_governance,
            params={"externalize_dir": str(custom)},
        ),
    )
    resolved = _resolve_relative_paths(cfg_with_custom)
    assert resolved.context_governance.params["externalize_dir"] == str(custom)


def test_other_params_preserved() -> None:
    """params 里其他键不被丢掉，仅 externalize_dir 被解析。"""
    from dataclasses import replace
    config = load_config()
    cfg = replace(
        config,
        context_governance=replace(
            config.context_governance,
            params={"summarize_model": "deepseek-chat", "extra_flag": True},
        ),
    )
    resolved = _resolve_relative_paths(cfg)
    assert resolved.context_governance.params["summarize_model"] == "deepseek-chat"
    assert resolved.context_governance.params["extra_flag"] is True
    assert "externalize_dir" in resolved.context_governance.params
