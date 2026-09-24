import pytest

from agent4ml.backend.agents.config.loader import ConfigError, load_config


def test_loads_default_profile() -> None:
    config = load_config()

    assert config.runtime.expert_mode is False
    assert config.runtime.plan_enabled is True
    assert config.reporting.save_artifact is True
    assert config.runtime.logs_root == ".agent4ml/logs"
    assert config.models.researcher_model == "fake-researcher"


def test_expert_mode_true_activates_profile() -> None:
    config = load_config(expert_mode=True)

    assert config.runtime.expert_mode is True
    assert config.runtime.reflection_enabled is True
    assert config.runtime.max_loop_steps == 8


def test_cli_override_can_change_expert_mode() -> None:
    config = load_config(expert_mode=True, cli_overrides={"expert_mode": False})

    assert config.runtime.expert_mode is False
    assert config.runtime.reflection_enabled is False


def test_missing_required_model_returns_clear_error() -> None:
    with pytest.raises(ConfigError, match="researcher_model is required"):
        load_config(cli_overrides={"researcher_model": ""})


def test_rejects_non_boolean_expert_mode() -> None:
    with pytest.raises(ConfigError, match="expert_mode must be a boolean"):
        load_config(cli_overrides={"expert_mode": "yes"})
