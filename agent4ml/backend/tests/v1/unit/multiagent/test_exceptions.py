"""Multi-Agent exceptions 单测 — 异常层次 + details 展开 + isinstance。"""
from __future__ import annotations

import pytest

from agent4ml.backend.agents.multiagent.exceptions import (
    SpecialistCrashError,
    SpecialistCredentialError,
    SpecialistError,
    SpecialistNotFoundError,
    SpecialistStartupError,
    SpecialistTimeoutError,
    SubagentError,
    SubagentMaxStepsError,
    SubagentTimeoutError,
)


def test_specialist_error_str_no_details():
    e = SpecialistError("fail")
    assert str(e) == "fail"


def test_specialist_error_str_with_details():
    e = SpecialistError("fail", details={"key": "val"})
    assert str(e) == "fail (key='val')"


def test_specialist_error_str_multiple_details():
    e = SpecialistError("fail", details={"a": 1, "b": "x"})
    s = str(e)
    assert s.startswith("fail (")
    assert "a=1" in s
    assert "b='x'" in s


def test_specialist_error_message_attr():
    e = SpecialistError("msg", details={"k": "v"})
    assert e.message == "msg"
    assert e.details == {"k": "v"}


def test_specialist_timeout_isinstance():
    e = SpecialistTimeoutError(timeout_seconds=30.0)
    assert isinstance(e, SpecialistError)
    assert e.details == {"timeout_seconds": 30.0}


def test_specialist_crash_exit_code():
    e = SpecialistCrashError(exit_code=137)
    assert isinstance(e, SpecialistError)
    assert e.details == {"exit_code": 137}


def test_specialist_startup_isinstance():
    e = SpecialistStartupError("handshake failed")
    assert isinstance(e, SpecialistError)


def test_specialist_credential_isinstance():
    e = SpecialistCredentialError("missing")
    assert isinstance(e, SpecialistError)


def test_specialist_not_found_with_name():
    e = SpecialistNotFoundError("codex")
    assert isinstance(e, SpecialistError)
    assert e.details == {"name": "codex"}
    assert "codex" in str(e)


def test_subagent_error_str_with_details():
    e = SubagentError("fail", details={"k": "v"})
    assert str(e) == "fail (k='v')"


def test_subagent_error_not_specialist_error():
    """SubagentError 独立层次，不属于 SpecialistError。"""
    e = SubagentError("fail")
    assert not isinstance(e, SpecialistError)
    assert isinstance(e, Exception)


def test_subagent_timeout_isinstance():
    e = SubagentTimeoutError("slow")
    assert isinstance(e, SubagentError)


def test_subagent_max_steps_with_max_steps():
    e = SubagentMaxStepsError(max_steps=20)
    assert isinstance(e, SubagentError)
    assert e.details == {"max_steps": 20}


def test_all_specialist_errors_raise():
    """四子类 + NotFound 都可 raise + catch as base。"""
    with pytest.raises(SpecialistError):
        raise SpecialistTimeoutError()
    with pytest.raises(SpecialistError):
        raise SpecialistCrashError(exit_code=1)
    with pytest.raises(SpecialistError):
        raise SpecialistStartupError()
    with pytest.raises(SpecialistError):
        raise SpecialistCredentialError()
    with pytest.raises(SpecialistError):
        raise SpecialistNotFoundError("x")


def test_all_subagent_errors_raise():
    with pytest.raises(SubagentError):
        raise SubagentTimeoutError()
    with pytest.raises(SubagentError):
        raise SubagentMaxStepsError(max_steps=10)
