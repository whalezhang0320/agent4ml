"""PiCredentialProvider 单测 — 双轨凭证解析（P2，决策 3）。

验证（tasks.md 2.2）：
- test_config_api_key_priority: config 显式 api_key 最高优先级
- test_config_provider_env_var: config provider → 找对应 env var
- test_env_var_china_priority: 同时设 DEEPSEEK + ANTHROPIC，选 DEEPSEEK（国内优先）
- test_auth_file_fallback: env vars 都没设时读 auth.json
- test_pi_not_installed: pi 不在 PATH，返 None
- test_no_credential: pi 装了但无 API key，返 None
"""
from __future__ import annotations

import shutil
from pathlib import Path
from unittest.mock import patch

import pytest

from agent4ml.backend.agents.multiagent.credentials.pi_credential import (
    PiCredential,
    PiCredentialProvider,
)


def _mock_pi_installed(installed: bool = True):
    """mock shutil.which 返 pi 路径或 None。"""
    return patch(
        "agent4ml.backend.agents.multiagent.credentials.pi_credential.shutil.which",
        return_value="/usr/local/bin/pi" if installed else None,
    )


def test_config_api_key_priority(monkeypatch):
    """决策 3 优先级 1：config 显式 api_key 最高优先级（即使 env 也设了）。"""
    # 清除所有 auth env vars
    for var in ("DEEPSEEK_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-from-env")

    with _mock_pi_installed():
        provider = PiCredentialProvider(
            config_provider="anthropic",
            config_api_key="sk-from-config",
        )
        cred = provider.get_credential()

    assert cred is not None
    assert cred.api_key == "sk-from-config"  # config 优先
    assert cred.provider == "anthropic"


def test_config_api_key_default_provider(monkeypatch):
    """config_api_key 设了但 config_provider=None 时默认 anthropic。"""
    for var in ("DEEPSEEK_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(var, raising=False)

    with _mock_pi_installed():
        provider = PiCredentialProvider(
            config_provider=None,
            config_api_key="sk-direct",
        )
        cred = provider.get_credential()

    assert cred is not None
    assert cred.api_key == "sk-direct"
    assert cred.provider == "anthropic"  # 默认


def test_config_provider_env_var(monkeypatch):
    """决策 3 优先级 2：config provider → 找对应 env var。"""
    for var in ("DEEPSEEK_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-deepseek")

    with _mock_pi_installed():
        provider = PiCredentialProvider(
            config_provider="deepseek",
            config_api_key=None,
        )
        cred = provider.get_credential()

    assert cred is not None
    assert cred.provider == "deepseek"
    assert cred.api_key == "sk-deepseek"


def test_config_provider_env_var_not_set(monkeypatch):
    """config provider 设了但对应 env var 没设 → 降级到遍历 env vars。"""
    for var in ("DEEPSEEK_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    # config_provider=deepseek 但 DEEPSEEK_API_KEY 没设，KIMI 设了
    monkeypatch.setenv("KIMI_API_KEY", "sk-kimi")

    with _mock_pi_installed():
        provider = PiCredentialProvider(
            config_provider="deepseek",
            config_api_key=None,
        )
        cred = provider.get_credential()

    # 降级到遍历，找到 KIMI（国内优先顺序）
    assert cred is not None
    assert cred.provider == "kimi-coding"
    assert cred.api_key == "sk-kimi"


def test_env_var_china_priority(monkeypatch):
    """决策 3 优先级 3：同时设 DEEPSEEK + ANTHROPIC，选 DEEPSEEK（国内优先）。"""
    for var in ("KIMI_API_KEY", "MINIMAX_API_KEY", "XIAOMI_API_KEY", "ZAI_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-deepseek")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-anthropic")

    with _mock_pi_installed():
        provider = PiCredentialProvider()
        cred = provider.get_credential()

    assert cred is not None
    assert cred.provider == "deepseek"  # 国内优先
    assert cred.api_key == "sk-deepseek"


def test_env_var_kimi_before_anthropic(monkeypatch):
    """KIMI 优先级高于 ANTHROPIC（国内 provider 顺序）。"""
    for var in ("DEEPSEEK_API_KEY", "MINIMAX_API_KEY", "XIAOMI_API_KEY", "ZAI_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("KIMI_API_KEY", "sk-kimi")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-anthropic")

    with _mock_pi_installed():
        provider = PiCredentialProvider()
        cred = provider.get_credential()

    assert cred is not None
    assert cred.provider == "kimi-coding"


def test_auth_file_fallback(monkeypatch, tmp_path):
    """决策 3 优先级 4：env vars 都没设时读 auth.json。"""
    # 清除所有 auth env vars
    for var in (
        "DEEPSEEK_API_KEY", "KIMI_API_KEY", "MINIMAX_API_KEY", "XIAOMI_API_KEY",
        "ZAI_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY",
        "OPENROUTER_API_KEY", "GROQ_API_KEY", "XAI_API_KEY", "MISTRAL_API_KEY",
        "TOGETHER_API_KEY",
    ):
        monkeypatch.delenv(var, raising=False)

    # 创建 mock auth.json
    auth_file = tmp_path / "auth.json"
    auth_file.write_text('{"anthropic": {"type": "api_key", "key": "sk-from-file"}}')

    with _mock_pi_installed():
        provider = PiCredentialProvider()
        with patch.object(
            provider, "_resolve_auth_path", return_value=auth_file
        ):
            cred = provider.get_credential()

    assert cred is not None
    assert cred.auth_file == str(auth_file)
    assert cred.api_key is None  # auth_file 模式不解析 api_key


def test_pi_not_installed(monkeypatch):
    """pi 不在 PATH → 返 None（specialist 不注册）。"""
    for var in ("DEEPSEEK_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant")

    with _mock_pi_installed(False):  # pi 不在 PATH
        provider = PiCredentialProvider()
        cred = provider.get_credential()

    assert cred is None


def test_no_credential(monkeypatch, tmp_path):
    """pi 装了但无 API key + 无 auth.json → 返 None。"""
    for var in (
        "DEEPSEEK_API_KEY", "KIMI_API_KEY", "MINIMAX_API_KEY", "XIAOMI_API_KEY",
        "ZAI_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY",
        "OPENROUTER_API_KEY", "GROQ_API_KEY", "XAI_API_KEY", "MISTRAL_API_KEY",
        "TOGETHER_API_KEY",
    ):
        monkeypatch.delenv(var, raising=False)

    # auth.json 不存在
    nonexistent_auth = tmp_path / "nonexistent-auth.json"

    with _mock_pi_installed():
        provider = PiCredentialProvider()
        with patch.object(
            provider, "_resolve_auth_path", return_value=nonexistent_auth
        ):
            cred = provider.get_credential()

    assert cred is None


def test_pi_credential_kind():
    """PiCredential.kind 默认 "pi"。"""
    cred = PiCredential()
    assert cred.kind == "pi"


def test_pi_credential_frozen():
    """PiCredential frozen dataclass 不可变。"""
    cred = PiCredential(provider="deepseek", api_key="sk-")
    with pytest.raises(Exception):
        cred.api_key = "changed"  # type: ignore


def test_resolve_auth_path_default(monkeypatch):
    """默认 auth path 是 ~/.pi/agent/auth.json。"""
    for var in ("PI_CODING_AGENT_DIR",):
        monkeypatch.delenv(var, raising=False)
    with _mock_pi_installed():
        provider = PiCredentialProvider()
        path = provider._resolve_auth_path()
    # 验证路径以 .pi/agent/auth.json 结尾
    assert str(path).endswith(".pi" + chr(92) + "agent" + chr(92) + "auth.json") or \
           str(path).endswith(".pi/agent/auth.json")


def test_resolve_auth_path_custom_dir(monkeypatch):
    """PI_CODING_AGENT_DIR 覆盖默认 auth path。"""
    monkeypatch.setenv("PI_CODING_AGENT_DIR", "/custom/pi/dir")
    with _mock_pi_installed():
        provider = PiCredentialProvider()
        path = provider._resolve_auth_path()
    assert str(path) == str(Path("/custom/pi/dir/auth.json"))
