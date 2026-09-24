"""SkillSource Protocol + SkillMeta 单测（H1）。

验证：
- SkillMeta frozen dataclass 结构
- SkillSource Protocol 契约（mock 实现可被接受）
- search/fetch/preview 方法签名
"""
from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from agent4ml.backend.agents.skill.hub.source import SkillMeta, SkillSource


def test_skill_meta_basic_fields():
    """SkillMeta 含基本字段。"""
    meta = SkillMeta(
        name="frontend-design",
        description="frontend UI design skill",
        category="creative",
        source="builtin",
        identifier="builtin:frontend-design",
    )
    assert meta.name == "frontend-design"
    assert meta.description == "frontend UI design skill"
    assert meta.category == "creative"
    assert meta.source == "builtin"
    assert meta.identifier == "builtin:frontend-design"


def test_skill_meta_defaults():
    """SkillMeta 默认值。"""
    meta = SkillMeta(
        name="test",
        description="desc",
        category="core",
        source="builtin",
        identifier="test",
    )
    assert meta.install_path is None
    assert meta.preview_url is None
    assert meta.is_installed is False


def test_skill_meta_frozen():
    """SkillMeta frozen 不可变。"""
    meta = SkillMeta(
        name="test", description="d", category="c", source="s", identifier="i"
    )
    with pytest.raises(FrozenInstanceError):
        meta.name = "changed"  # type: ignore


def test_skill_source_protocol_mock_implementation():
    """mock 类实现 SkillSource 方法签名可被接受（结构性类型）。"""

    class _MockSource:
        name = "mock"

        def search(self, query: str, limit: int = 10) -> list[SkillMeta]:
            return [SkillMeta(
                name=query, description="mock", category="test",
                source="mock", identifier=f"mock:{query}",
            )]

        def fetch(self, identifier: str, dest_dir: Path) -> Path:
            return dest_dir / identifier

        def preview(self, identifier: str) -> str | None:
            return f"# {identifier}\nmock content"

    # mock 实现可被赋给 SkillSource 类型注解的变量（Protocol 结构性类型）
    source: SkillSource = _MockSource()  # type: ignore[assignment]

    # 验证方法可调用
    results = source.search("test")
    assert len(results) == 1
    assert results[0].name == "test"

    fetched = source.fetch("mock:test", Path("/tmp"))
    assert fetched == Path("/tmp/mock:test")

    preview = source.preview("mock:test")
    assert preview is not None
    assert "mock content" in preview


def test_skill_meta_to_dict_roundtrip():
    """SkillMeta 字段可作为 dict 序列化（供 JSON 返回）。"""
    meta = SkillMeta(
        name="tdd",
        description="test driven development",
        category="core",
        source="builtin",
        identifier="builtin:tdd",
        is_installed=True,
        install_path="/path/to/tdd",
    )
    d = {
        "name": meta.name,
        "description": meta.description,
        "category": meta.category,
        "source": meta.source,
        "identifier": meta.identifier,
        "is_installed": meta.is_installed,
        "install_path": meta.install_path,
        "preview_url": meta.preview_url,
    }
    assert d["name"] == "tdd"
    assert d["is_installed"] is True
    assert d["install_path"] == "/path/to/tdd"
