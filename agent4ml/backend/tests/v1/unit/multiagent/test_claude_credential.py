"""ClaudeCredentialProvider 单测 — env 覆盖 + OAuth 解析 + 过期检测 + 缺失。"""
from __future__ import annotations

import json
import time
import dataclasses
from pathlib import Path

import pytest

from agent4ml.backend.agents.multiagent.credentials.claude_credential import (
    ClaudeCredential,
    ClaudeCredentialProvider,
)


def _write_json(path: Path, data: dict) -> Path:
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def _oauth_data(access_token="tok", refresh_token="rt", expires_at=0, **extra):
    oauth = {
        "accessToken": access_token,
        "refreshToken": refresh_token,
        "expiresAt": expires_at,
    }
    oauth.update(extra)
    return {"claudeAiOauth": oauth}


def test_claude_credential_frozen():
    c = ClaudeCredential(kind="claude", access_token="tok")
    with pytest.raises(dataclasses.FrozenInstanceError):
        c.access_token = "x"


def test_claude_credential_defaults():
    c = ClaudeCredential(kind="claude", access_token="tok")
    assert c.refresh_token == ""
    assert c.expires_at == 0


def test_is_expired_no_expiry():
    """expires_at=0 不做过期检测（无过期信息）。"""
    c = ClaudeCredential(kind="claude", access_token="tok", expires_at=0)
    assert c.is_expired is False


def test_is_expired_future():
    future = int((time.time() + 3600) * 1000)
    c = ClaudeCredential(kind="claude", access_token="tok", expires_at=future)
    assert c.is_expired is False


def test_is_expired_past():
    past = int((time.time() - 3600) * 1000)
    c = ClaudeCredential(kind="claude", access_token="tok", expires_at=past)
    assert c.is_expired is True


def test_resolve_path_env_override(monkeypatch, tmp_path):
    custom = tmp_path / "custom_creds.json"
    monkeypatch.setenv("CLAUDE_CODE_CREDENTIALS_PATH", str(custom))
    provider = ClaudeCredentialProvider()
    assert provider._resolve_path() == custom


def test_resolve_path_default(monkeypatch):
    monkeypatch.delenv("CLAUDE_CODE_CREDENTIALS_PATH", raising=False)
    monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    provider = ClaudeCredentialProvider()
    path = provider._resolve_path()
    assert path == Path.home() / ".claude" / ".credentials.json"


def test_direct_token_from_env(monkeypatch):
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "env_token_123")
    monkeypatch.delenv("CLAUDE_CODE_CREDENTIALS_PATH", raising=False)

    cred = ClaudeCredentialProvider().get_credential()

    assert cred is not None
    assert cred.kind == "claude"
    assert cred.access_token == "env_token_123"
    assert cred.expires_at == 0


def test_direct_token_from_anthropic_auth_fallback(monkeypatch):
    monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "anthropic_token")

    cred = ClaudeCredentialProvider().get_credential()

    assert cred is not None
    assert cred.access_token == "anthropic_token"


def test_oauth_file_parsing(monkeypatch, tmp_path):
    cred_path = _write_json(tmp_path / "creds.json", _oauth_data(
        access_token="file_tok",
        refresh_token="file_rt",
        expires_at=0,
    ))
    monkeypatch.setenv("CLAUDE_CODE_CREDENTIALS_PATH", str(cred_path))
    monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)

    cred = ClaudeCredentialProvider().get_credential()

    assert cred is not None
    assert cred.access_token == "file_tok"
    assert cred.refresh_token == "file_rt"
    assert cred.expires_at == 0


def test_oauth_file_future_not_expired(monkeypatch, tmp_path):
    future = int((time.time() + 3600) * 1000)
    cred_path = _write_json(tmp_path / "creds.json", _oauth_data(expires_at=future))
    monkeypatch.setenv("CLAUDE_CODE_CREDENTIALS_PATH", str(cred_path))
    monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)

    cred = ClaudeCredentialProvider().get_credential()

    assert cred is not None
    assert cred.access_token == "tok"


def test_oauth_file_expired_returns_none(monkeypatch, tmp_path):
    past = int((time.time() - 3600) * 1000)
    cred_path = _write_json(tmp_path / "creds.json", _oauth_data(expires_at=past))
    monkeypatch.setenv("CLAUDE_CODE_CREDENTIALS_PATH", str(cred_path))
    monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)

    cred = ClaudeCredentialProvider().get_credential()

    assert cred is None


def test_file_not_found_returns_none(monkeypatch, tmp_path):
    monkeypatch.setenv("CLAUDE_CODE_CREDENTIALS_PATH", str(tmp_path / "nonexistent.json"))
    monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    assert ClaudeCredentialProvider().get_credential() is None


def test_no_access_token_returns_none(monkeypatch, tmp_path):
    cred_path = _write_json(tmp_path / "creds.json", {"claudeAiOauth": {"refreshToken": "rt"}})
    monkeypatch.setenv("CLAUDE_CODE_CREDENTIALS_PATH", str(cred_path))
    monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
    assert ClaudeCredentialProvider().get_credential() is None


def test_empty_access_token_returns_none(monkeypatch, tmp_path):
    cred_path = _write_json(tmp_path / "creds.json", _oauth_data(access_token=""))
    monkeypatch.setenv("CLAUDE_CODE_CREDENTIALS_PATH", str(cred_path))
    monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
    assert ClaudeCredentialProvider().get_credential() is None


def test_invalid_json_returns_none(monkeypatch, tmp_path):
    bad = tmp_path / "creds.json"
    bad.write_text("{invalid", encoding="utf-8")
    monkeypatch.setenv("CLAUDE_CODE_CREDENTIALS_PATH", str(bad))
    monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
    assert ClaudeCredentialProvider().get_credential() is None


def test_direct_token_takes_precedence_over_file(monkeypatch, tmp_path):
    """CLAUDE_CODE_OAUTH_TOKEN 优先于 credentials 文件。"""
    cred_path = _write_json(tmp_path / "creds.json", _oauth_data(access_token="file_tok"))
    monkeypatch.setenv("CLAUDE_CODE_CREDENTIALS_PATH", str(cred_path))
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "env_tok")

    cred = ClaudeCredentialProvider().get_credential()

    assert cred is not None
    assert cred.access_token == "env_tok"


def test_oauth_non_dict_returns_none(monkeypatch, tmp_path):
    """claudeAiOauth 不是 dict 时返 None。"""
    cred_path = _write_json(tmp_path / "creds.json", {"claudeAiOauth": "not_a_dict"})
    monkeypatch.setenv("CLAUDE_CODE_CREDENTIALS_PATH", str(cred_path))
    monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
    assert ClaudeCredentialProvider().get_credential() is None
