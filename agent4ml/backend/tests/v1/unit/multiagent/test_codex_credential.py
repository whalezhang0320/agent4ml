"""CodexCredentialProvider 单测 — 默认路径 + env 覆盖 + legacy/nested 格式 + 缺失。"""
from __future__ import annotations

import json
import dataclasses
from pathlib import Path

import pytest

from agent4ml.backend.agents.multiagent.credentials.codex_credential import (
    CodexCredential,
    CodexCredentialProvider,
)


def _write_json(path: Path, data: dict) -> Path:
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_codex_credential_frozen():
    c = CodexCredential(kind="codex", access_token="tok")
    with pytest.raises(dataclasses.FrozenInstanceError):
        c.access_token = "x"


def test_codex_credential_defaults():
    c = CodexCredential(kind="codex", access_token="tok")
    assert c.account_id == ""


def test_resolve_path_env_override(monkeypatch, tmp_path):
    custom = tmp_path / "custom_auth.json"
    monkeypatch.setenv("CODEX_AUTH_PATH", str(custom))
    provider = CodexCredentialProvider()
    assert provider._resolve_path() == custom


def test_resolve_path_default(monkeypatch):
    monkeypatch.delenv("CODEX_AUTH_PATH", raising=False)
    provider = CodexCredentialProvider()
    path = provider._resolve_path()
    assert path == Path.home() / ".codex" / "auth.json"


def test_legacy_format(monkeypatch, tmp_path):
    auth_path = _write_json(tmp_path / "auth.json", {
        "access_token": "tok_legacy",
        "account_id": "acc1",
    })
    monkeypatch.setenv("CODEX_AUTH_PATH", str(auth_path))

    cred = CodexCredentialProvider().get_credential()

    assert cred is not None
    assert cred.kind == "codex"
    assert cred.access_token == "tok_legacy"
    assert cred.account_id == "acc1"


def test_nested_format(monkeypatch, tmp_path):
    auth_path = _write_json(tmp_path / "auth.json", {
        "tokens": {
            "access_token": "tok_nested",
            "account_id": "acc2",
        },
    })
    monkeypatch.setenv("CODEX_AUTH_PATH", str(auth_path))

    cred = CodexCredentialProvider().get_credential()

    assert cred is not None
    assert cred.access_token == "tok_nested"
    assert cred.account_id == "acc2"


def test_legacy_token_field_fallback(monkeypatch, tmp_path):
    """data.get('token') 作为 access_token 的 fallback。"""
    auth_path = _write_json(tmp_path / "auth.json", {"token": "tok_field"})
    monkeypatch.setenv("CODEX_AUTH_PATH", str(auth_path))

    cred = CodexCredentialProvider().get_credential()

    assert cred is not None
    assert cred.access_token == "tok_field"


def test_file_not_found_returns_none(monkeypatch, tmp_path):
    monkeypatch.setenv("CODEX_AUTH_PATH", str(tmp_path / "nonexistent.json"))
    assert CodexCredentialProvider().get_credential() is None


def test_no_access_token_returns_none(monkeypatch, tmp_path):
    auth_path = _write_json(tmp_path / "auth.json", {"account_id": "acc"})
    monkeypatch.setenv("CODEX_AUTH_PATH", str(auth_path))
    assert CodexCredentialProvider().get_credential() is None


def test_empty_access_token_returns_none(monkeypatch, tmp_path):
    auth_path = _write_json(tmp_path / "auth.json", {"access_token": "", "account_id": "acc"})
    monkeypatch.setenv("CODEX_AUTH_PATH", str(auth_path))
    assert CodexCredentialProvider().get_credential() is None


def test_invalid_json_returns_none(monkeypatch, tmp_path):
    bad = tmp_path / "auth.json"
    bad.write_text("{not valid json", encoding="utf-8")
    monkeypatch.setenv("CODEX_AUTH_PATH", str(bad))
    assert CodexCredentialProvider().get_credential() is None


def test_directory_path_returns_none(monkeypatch, tmp_path):
    monkeypatch.setenv("CODEX_AUTH_PATH", str(tmp_path))
    assert CodexCredentialProvider().get_credential() is None


def test_nested_with_legacy_precedence(monkeypatch, tmp_path):
    """legacy top-level access_token 优先于 nested tokens.access_token。"""
    auth_path = _write_json(tmp_path / "auth.json", {
        "access_token": "top_level",
        "tokens": {"access_token": "nested_one"},
    })
    monkeypatch.setenv("CODEX_AUTH_PATH", str(auth_path))

    cred = CodexCredentialProvider().get_credential()

    assert cred is not None
    assert cred.access_token == "top_level"
