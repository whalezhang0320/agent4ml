"""test_skill_config — SkillConfig 默认值 + frozen + .env 覆盖。

覆盖 spec Scenario:
- 默认值: SkillConfig() 各字段
- frozen: 修改字段抛 FrozenInstanceError
- load_skill_config 无 env 时返默认值
- AGENT4ML_SKILL_ENABLED=false → enabled=False
- AGENT4ML_SKILL_DIRS="a,b" → skill_dirs==("a","b")
- AGENT4ML_SKILL_MAX_INJECT=5 → max_inject==5
- 非法 int → 用默认值不抛
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from agent4ml.backend.agents.skill.config import (
    SkillConfig,
    _anchor,
    load_skill_config,
)

# 所有 AGENT4ML_SKILL_* 环境变量名
_SKILL_ENV_KEYS = [
    "AGENT4ML_SKILL_ENABLED",
    "AGENT4ML_SKILL_DB_PATH",
    "AGENT4ML_SKILL_DIRS",
    "AGENT4ML_SKILL_MAX_INJECT",
    "AGENT4ML_SKILL_QUALITY_THRESHOLD",
    "AGENT4ML_SKILL_MIN_SELECTIONS",
]


def _clear_skill_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in _SKILL_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)


class TestSkillConfigDefaults:
    """Scenario: 默认值"""

    def test_default_values(self):
        cfg = SkillConfig()
        assert cfg.enabled is False
        assert cfg.db_path == ".agent4ml/skills.db"
        assert cfg.skill_dirs == ("skills/",)
        assert cfg.max_inject == 3
        assert cfg.quality_threshold == 0.3
        assert cfg.min_selections == 5


class TestSkillConfigFrozen:
    """Scenario: frozen"""

    def test_modify_enabled_raises_frozen(self):
        cfg = SkillConfig()
        with pytest.raises(dataclasses.FrozenInstanceError):
            cfg.enabled = False  # type: ignore[misc]

    def test_modify_db_path_raises_frozen(self):
        cfg = SkillConfig()
        with pytest.raises(dataclasses.FrozenInstanceError):
            cfg.db_path = "other.db"  # type: ignore[misc]

    def test_modify_max_inject_raises_frozen(self):
        cfg = SkillConfig()
        with pytest.raises(dataclasses.FrozenInstanceError):
            cfg.max_inject = 10  # type: ignore[misc]


class TestLoadSkillConfigNoEnv:
    """Scenario: 无 env 时返默认值"""

    def test_no_env_returns_defaults(self, monkeypatch):
        _clear_skill_env(monkeypatch)
        cfg = load_skill_config()
        assert cfg.enabled is False
        # load_skill_config 锚定相对路径到项目根（CWD 无关），dataclass 默认值仍相对
        assert cfg.db_path == _anchor(".agent4ml/skills.db")
        assert Path(cfg.db_path).is_absolute()
        assert cfg.skill_dirs == (_anchor("skills"),)
        assert all(Path(d).is_absolute() for d in cfg.skill_dirs)
        assert cfg.max_inject == 3
        assert cfg.quality_threshold == pytest.approx(0.3)
        assert cfg.min_selections == 5


class TestLoadSkillConfigEnabled:
    """Scenario: AGENT4ML_SKILL_ENABLED 覆盖"""

    def test_enabled_false(self, monkeypatch):
        _clear_skill_env(monkeypatch)
        monkeypatch.setenv("AGENT4ML_SKILL_ENABLED", "false")
        cfg = load_skill_config()
        assert cfg.enabled is False

    def test_enabled_true_uppercase(self, monkeypatch):
        _clear_skill_env(monkeypatch)
        monkeypatch.setenv("AGENT4ML_SKILL_ENABLED", "TRUE")
        cfg = load_skill_config()
        assert cfg.enabled is True

    def test_enabled_random_string_means_false(self, monkeypatch):
        _clear_skill_env(monkeypatch)
        monkeypatch.setenv("AGENT4ML_SKILL_ENABLED", "yes")
        cfg = load_skill_config()
        assert cfg.enabled is False


class TestLoadSkillConfigDirs:
    """Scenario: AGENT4ML_SKILL_DIRS 逗号分隔"""

    def test_dirs_two_entries(self, monkeypatch):
        _clear_skill_env(monkeypatch)
        monkeypatch.setenv("AGENT4ML_SKILL_DIRS", "a,b")
        cfg = load_skill_config()
        assert cfg.skill_dirs == (_anchor("a"), _anchor("b"))

    def test_dirs_with_spaces_stripped(self, monkeypatch):
        _clear_skill_env(monkeypatch)
        monkeypatch.setenv("AGENT4ML_SKILL_DIRS", " a , b , c ")
        cfg = load_skill_config()
        assert cfg.skill_dirs == (_anchor("a"), _anchor("b"), _anchor("c"))

    def test_dirs_empty_falls_back_default(self, monkeypatch):
        _clear_skill_env(monkeypatch)
        monkeypatch.setenv("AGENT4ML_SKILL_DIRS", "")
        cfg = load_skill_config()
        assert cfg.skill_dirs == (_anchor("skills"),)

    def test_dirs_absolute_passthrough(self, monkeypatch):
        """绝对路径原样返回，不重新锚定。"""
        _clear_skill_env(monkeypatch)
        abs_dir = str(Path(__file__).parent / "fixtures")
        monkeypatch.setenv("AGENT4ML_SKILL_DIRS", abs_dir)
        cfg = load_skill_config()
        assert cfg.skill_dirs == (abs_dir,)


class TestLoadSkillConfigMaxInject:
    """Scenario: AGENT4ML_SKILL_MAX_INJECT int 覆盖 + 非法值兜底"""

    def test_max_inject_override(self, monkeypatch):
        _clear_skill_env(monkeypatch)
        monkeypatch.setenv("AGENT4ML_SKILL_MAX_INJECT", "5")
        cfg = load_skill_config()
        assert cfg.max_inject == 5

    def test_max_inject_invalid_uses_default(self, monkeypatch):
        _clear_skill_env(monkeypatch)
        monkeypatch.setenv("AGENT4ML_SKILL_MAX_INJECT", "abc")
        cfg = load_skill_config()
        assert cfg.max_inject == 3


class TestLoadSkillConfigQualityThreshold:
    """Scenario: AGENT4ML_SKILL_QUALITY_THRESHOLD float 覆盖 + 非法值兜底"""

    def test_quality_threshold_override(self, monkeypatch):
        _clear_skill_env(monkeypatch)
        monkeypatch.setenv("AGENT4ML_SKILL_QUALITY_THRESHOLD", "0.5")
        cfg = load_skill_config()
        assert cfg.quality_threshold == pytest.approx(0.5)

    def test_quality_threshold_invalid_uses_default(self, monkeypatch):
        _clear_skill_env(monkeypatch)
        monkeypatch.setenv("AGENT4ML_SKILL_QUALITY_THRESHOLD", "not-a-float")
        cfg = load_skill_config()
        assert cfg.quality_threshold == pytest.approx(0.3)


class TestLoadSkillConfigMinSelections:
    """Scenario: AGENT4ML_SKILL_MIN_SELECTIONS int 覆盖 + 非法值兜底"""

    def test_min_selections_override(self, monkeypatch):
        _clear_skill_env(monkeypatch)
        monkeypatch.setenv("AGENT4ML_SKILL_MIN_SELECTIONS", "10")
        cfg = load_skill_config()
        assert cfg.min_selections == 10

    def test_min_selections_invalid_uses_default(self, monkeypatch):
        _clear_skill_env(monkeypatch)
        monkeypatch.setenv("AGENT4ML_SKILL_MIN_SELECTIONS", "xyz")
        cfg = load_skill_config()
        assert cfg.min_selections == 5


class TestLoadSkillConfigDbPath:
    """Scenario: AGENT4ML_SKILL_DB_PATH 覆盖"""

    def test_db_path_override(self, monkeypatch):
        _clear_skill_env(monkeypatch)
        monkeypatch.setenv("AGENT4ML_SKILL_DB_PATH", "custom/skills.db")
        cfg = load_skill_config()
        assert cfg.db_path == _anchor("custom/skills.db")
        assert Path(cfg.db_path).is_absolute()
