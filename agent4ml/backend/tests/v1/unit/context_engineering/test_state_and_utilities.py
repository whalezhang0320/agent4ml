"""merge_governance reducer + token utility 单测。"""

from __future__ import annotations

from types import SimpleNamespace

from agent4ml.backend.agents.state.reducers import merge_governance
from agent4ml.backend.agents.context_engineering.utilities import (
    _char_estimate,
    resolve_window_size,
    token_counter,
)


# ---------- merge_governance ----------


def test_merge_governance_incoming_none_preserves_current() -> None:
    current = {"vol.refs": {"a": "p1"}}
    assert merge_governance(current, None) is current


def test_merge_governance_current_none_returns_copy() -> None:
    incoming = {"vol.watermark": 5}
    result = merge_governance(None, incoming)
    assert result == incoming
    assert result is not incoming  # copy


def test_merge_governance_deep_merge_nested_dict() -> None:
    current = {"vol.refs": {"a": "p1"}}
    incoming = {"vol.refs": {"b": "p2"}}
    result = merge_governance(current, incoming)
    assert result == {"vol.refs": {"a": "p1", "b": "p2"}}


def test_merge_governance_last_write_wins_scalar() -> None:
    current = {"vol.watermark": 5}
    incoming = {"vol.watermark": 10}
    assert merge_governance(current, incoming) == {"vol.watermark": 10}


def test_merge_governance_different_keys_coexist() -> None:
    current = {"vol.refs": {"a": "p1"}}
    incoming = {"vol.watermark": 5}
    assert merge_governance(current, incoming) == {
        "vol.refs": {"a": "p1"},
        "vol.watermark": 5,
    }


# ---------- token_counter ----------


def test_token_counter_empty() -> None:
    assert token_counter([]) == 0


def test_token_counter_cjk_real_count() -> None:
    # tiktoken 可用时返回实际计数（非 fallback）；至少 > 0
    count = token_counter(["你好世界"])
    assert count > 0


def test_char_estimate_cjk_aware() -> None:
    # 4 CJK + 4 latin("test") → 4 + 4//4 = 5
    assert _char_estimate(["你好世界test"]) == 5


def test_char_estimate_empty() -> None:
    assert _char_estimate([]) == 0


# ---------- resolve_window_size ----------


def test_resolve_window_size_attr() -> None:
    """model 属性 max_input_tokens 优先。"""
    model = SimpleNamespace(max_input_tokens=8000)
    assert resolve_window_size(model) == 8000


def test_resolve_window_size_model_name_map() -> None:
    """无属性 → model_name 映射表精确匹配。"""
    model = SimpleNamespace(model_name="deepseek-chat")
    assert resolve_window_size(model) == 64_000


def test_resolve_window_size_prefix_match() -> None:
    """model_name 前缀匹配（带日期后缀）。"""
    model = SimpleNamespace(model_name="gpt-4o-2024-08-06")
    assert resolve_window_size(model) == 128_000


def test_resolve_window_size_fallback() -> None:
    """无属性无 model_name → default 128000。"""
    model = SimpleNamespace()
    assert resolve_window_size(model) == 128_000
