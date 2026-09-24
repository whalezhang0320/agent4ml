"""Batch B9 self-check: externalizer tool_metadata linkage."""
from langchain_core.messages import ToolMessage
from agent4ml.backend.agents.context_engineering.strategies.default.externalizer import (
    ExternalizerExecutor,
)


def test_threshold_no_metadata():
    """无元数据 → 走 min_chars 默认 500。"""
    e = ExternalizerExecutor(min_chars=500)
    assert e._get_threshold("unknown") == 500
    assert e._get_threshold(None) == 500
    print("PASS: no metadata → min_chars default")


def test_threshold_with_metadata():
    """有元数据 → max(min_chars, typical_tokens * 4)。"""
    e = ExternalizerExecutor(
        min_chars=500,
        tool_metadata={"bash": {"typical_output_tokens": 2000}},
    )
    # bash 2000 tokens → 8000 chars
    assert e._get_threshold("bash") == 8000
    print("PASS: with metadata → max(min_chars, tokens*4)")


def test_threshold_small_metadata_uses_min():
    """元数据 typical_tokens 小 → max 仍取 min_chars。"""
    e = ExternalizerExecutor(
        min_chars=500,
        tool_metadata={"small_tool": {"typical_output_tokens": 50}},
    )
    # 50 tokens * 4 = 200 < 500 → max(500, 200) = 500
    assert e._get_threshold("small_tool") == 500
    print("PASS: small metadata → max picks min_chars")


def test_externalize_if_needed_metadata_threshold():
    """externalize_if_needed 按工具元数据调阈值。"""
    e = ExternalizerExecutor(
        min_chars=500,
        tool_metadata={"bash": {"typical_output_tokens": 2000}},
    )
    # bash 3000 chars < 8000 threshold → 不外化
    msg = ToolMessage(content="x" * 3000, tool_call_id="c1", name="bash")
    result = e.externalize_if_needed(msg)
    assert result is None, "bash 3000 < 8000 threshold, should not externalize"
    print("PASS: bash 3000 < 8000 → no externalize")


def test_externalize_if_needed_metadata_exceeds():
    """bash 9000 chars > 8000 threshold → 外化。"""
    e = ExternalizerExecutor(
        min_chars=500,
        tool_metadata={"bash": {"typical_output_tokens": 2000}},
    )
    msg = ToolMessage(content="x" * 9000, tool_call_id="c2", name="bash")
    result = e.externalize_if_needed(msg)
    assert result is not None, "bash 9000 > 8000 threshold, should externalize"
    print("PASS: bash 9000 > 8000 → externalize")


def test_externalize_if_needed_no_metadata_uses_min():
    """无元数据工具 → 走 min_chars。"""
    e = ExternalizerExecutor(
        min_chars=500,
        tool_metadata={"bash": {"typical_output_tokens": 2000}},
    )
    # unknown 工具 600 chars > 500 min_chars → 外化
    msg = ToolMessage(content="x" * 600, tool_call_id="c3", name="unknown_tool")
    result = e.externalize_if_needed(msg)
    assert result is not None
    print("PASS: unknown tool 600 > 500 min_chars → externalize")


def test_externalize_if_needed_no_metadata_below_min():
    """无元数据工具 400 chars < 500 → 不外化。"""
    e = ExternalizerExecutor(
        min_chars=500,
        tool_metadata={"bash": {"typical_output_tokens": 2000}},
    )
    msg = ToolMessage(content="x" * 400, tool_call_id="c4", name="unknown_tool")
    result = e.externalize_if_needed(msg)
    assert result is None
    print("PASS: unknown tool 400 < 500 → no externalize")


if __name__ == "__main__":
    test_threshold_no_metadata()
    test_threshold_with_metadata()
    test_threshold_small_metadata_uses_min()
    test_externalize_if_needed_metadata_threshold()
    test_externalize_if_needed_metadata_exceeds()
    test_externalize_if_needed_no_metadata_uses_min()
    test_externalize_if_needed_no_metadata_below_min()
    print("\nAll B9 self-checks passed.")
