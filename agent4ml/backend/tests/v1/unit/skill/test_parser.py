"""parser 单测（B4）— .skill_id sidecar + frontmatter 解析 + install。"""
from __future__ import annotations

import re

import pytest

from agent4ml.backend.agents.skill.parser import (
    install,
    parse_skill_file,
    read_or_create_skill_id,
)

_SKILL_MD_TEMPLATE = """\
---
name: {name}
description: {description}{allowed_tools}{enabled}
---

# {title}

Body text.
"""

_IMP_ID_RE = re.compile(r"^[a-z0-9-]+__imp_[0-9a-f]{8}$")
_EVOLVED_ID_RE = re.compile(r"^[a-z0-9-]+__v\d+_[0-9a-f]{8}$")


def _write_skill_md(
    path, name="source-verification", description="验证信源可信度",
    allowed_tools=None, enabled=None,
):
    parts = []
    if allowed_tools is not None:
        lines = "\n".join(f"  - {t}" for t in allowed_tools)
        parts.append(f"\nallowed-tools:\n{lines}")
    if enabled is not None:
        parts.append(f"\nenabled: {enabled}")
    content = _SKILL_MD_TEMPLATE.format(
        name=name, description=description,
        allowed_tools="".join(parts), enabled="", title=name.replace("-", " ").title(),
    )
    path.write_text(content, encoding="utf-8")


# ── read_or_create_skill_id ──────────────────────────────────────────


class TestReadOrCreateSkillId:
    def test_first_time_generates_and_writes_sidecar(self, tmp_path):
        skill_dir = tmp_path / "my-skill"
        skill_dir.mkdir()
        sid = read_or_create_skill_id(skill_dir, "my-skill")
        assert _IMP_ID_RE.match(sid)
        sidecar = skill_dir / ".skill_id"
        assert sidecar.exists()
        assert sidecar.read_text(encoding="utf-8").strip() == sid

    def test_existing_sidecar_not_regenerated(self, tmp_path):
        skill_dir = tmp_path / "my-skill"
        skill_dir.mkdir()
        fixed = "my-skill__imp_aabbccdd"
        (skill_dir / ".skill_id").write_text(fixed, encoding="utf-8")
        sid = read_or_create_skill_id(skill_dir, "my-skill")
        assert sid == fixed

    def test_dir_rename_id_unchanged(self, tmp_path):
        skill_dir = tmp_path / "original"
        skill_dir.mkdir()
        sid = read_or_create_skill_id(skill_dir, "original")
        new_dir = tmp_path / "renamed"
        skill_dir.rename(new_dir)
        sid2 = read_or_create_skill_id(new_dir, "original")
        assert sid == sid2

    def test_evolved_origin_format(self, tmp_path):
        skill_dir = tmp_path / "evolved"
        skill_dir.mkdir()
        sid = read_or_create_skill_id(skill_dir, "evolved", origin="FIXED", generation=2)
        assert _EVOLVED_ID_RE.match(sid)

    def test_builtin_origin_deterministic_no_sidecar(self, tmp_path):
        skill_dir = tmp_path / "builtin"
        skill_dir.mkdir()
        sid = read_or_create_skill_id(skill_dir, "source-verification", origin="BUILTIN")
        assert sid == "source-verification__builtin"
        # 确定性：二次调用同 id
        sid2 = read_or_create_skill_id(skill_dir, "source-verification", origin="BUILTIN")
        assert sid2 == sid
        # 无 sidecar 文件
        assert not (skill_dir / ".skill_id").exists()


# ── parse_skill_file ─────────────────────────────────────────────────


class TestParseSkillFile:
    def test_full_frontmatter(self, tmp_path):
        skill_dir = tmp_path / "source-verification"
        skill_dir.mkdir()
        _write_skill_md(
            skill_dir / "SKILL.md",
            allowed_tools=["web_search", "browse_page"],
        )
        rec = parse_skill_file(skill_dir / "SKILL.md")
        assert rec.name == "source-verification"
        assert rec.description == "验证信源可信度"
        assert rec.allowed_tools == ("web_search", "browse_page")
        assert rec.enabled is True
        assert rec.lineage.origin == "IMPORTED"
        assert _IMP_ID_RE.match(rec.skill_id)
        assert len(rec.content_hash) == 16
        assert rec.path == str(skill_dir / "SKILL.md")

    def test_allowed_tools_default_empty(self, tmp_path):
        skill_dir = tmp_path / "guidance"
        skill_dir.mkdir()
        _write_skill_md(skill_dir / "SKILL.md")
        rec = parse_skill_file(skill_dir / "SKILL.md")
        assert rec.allowed_tools == ()

    def test_enabled_default_true(self, tmp_path):
        skill_dir = tmp_path / "guidance"
        skill_dir.mkdir()
        _write_skill_md(skill_dir / "SKILL.md")
        rec = parse_skill_file(skill_dir / "SKILL.md")
        assert rec.enabled is True

    def test_enabled_false(self, tmp_path):
        skill_dir = tmp_path / "disabled"
        skill_dir.mkdir()
        _write_skill_md(skill_dir / "SKILL.md", enabled=False)
        rec = parse_skill_file(skill_dir / "SKILL.md")
        assert rec.enabled is False

    def test_missing_name_raises(self, tmp_path):
        skill_dir = tmp_path / "bad"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            "---\ndescription: no name\n---\nbody", encoding="utf-8"
        )
        with pytest.raises(ValueError, match="name"):
            parse_skill_file(skill_dir / "SKILL.md")

    def test_missing_description_raises(self, tmp_path):
        skill_dir = tmp_path / "bad"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            "---\nname: no-desc\n---\nbody", encoding="utf-8"
        )
        with pytest.raises(ValueError, match="description"):
            parse_skill_file(skill_dir / "SKILL.md")

    def test_missing_frontmatter_raises(self, tmp_path):
        skill_dir = tmp_path / "nofm"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("# Just body\nno frontmatter", encoding="utf-8")
        with pytest.raises(ValueError, match="frontmatter"):
            parse_skill_file(skill_dir / "SKILL.md")

    def test_content_hash_deterministic(self, tmp_path):
        skill_dir = tmp_path / "hash-test"
        skill_dir.mkdir()
        _write_skill_md(skill_dir / "SKILL.md")
        rec1 = parse_skill_file(skill_dir / "SKILL.md")
        rec2 = parse_skill_file(skill_dir / "SKILL.md")
        assert rec1.content_hash == rec2.content_hash

    def test_builtin_origin_deterministic_id_no_sidecar(self, tmp_path):
        skill_dir = tmp_path / "source-verification"
        skill_dir.mkdir()
        _write_skill_md(skill_dir / "SKILL.md")
        rec = parse_skill_file(skill_dir / "SKILL.md", origin="BUILTIN")
        assert rec.skill_id == "source-verification__builtin"
        assert rec.lineage.origin == "BUILTIN"
        # BUILTIN 不写 sidecar
        assert not (skill_dir / ".skill_id").exists()


# ── install ──────────────────────────────────────────────────────────


class TestInstall:
    def test_install_copies_and_returns_skill_id(self, tmp_path):
        source = tmp_path / "source-skill"
        source.mkdir()
        _write_skill_md(source / "SKILL.md")
        dest_root = tmp_path / "dest"
        dest_root.mkdir()

        sid = install(source, "source-verification", dest_root)

        assert _IMP_ID_RE.match(sid)
        assert (dest_root / "source-verification" / "SKILL.md").exists()
        assert (dest_root / "source-verification" / ".skill_id").exists()
        assert (dest_root / "source-verification" / ".skill_id").read_text(encoding="utf-8").strip() == sid

    def test_install_overwrites_existing(self, tmp_path):
        source = tmp_path / "src"
        source.mkdir()
        _write_skill_md(source / "SKILL.md")
        dest_root = tmp_path / "dest"
        dest_root.mkdir()
        # pre-existing stale dir
        (dest_root / "source-verification").mkdir()
        (dest_root / "source-verification" / "stale.txt").write_text("old", encoding="utf-8")

        sid = install(source, "source-verification", dest_root)

        assert not (dest_root / "source-verification" / "stale.txt").exists()
        assert _IMP_ID_RE.match(sid)

    def test_install_invalid_name_path_traversal_raises(self, tmp_path):
        """install 名含 ../ 逃逸 → ValueError。"""
        source = tmp_path / "src"
        source.mkdir()
        _write_skill_md(source / "SKILL.md")
        dest_root = tmp_path / "dest"
        dest_root.mkdir()
        with pytest.raises(ValueError, match="invalid skill name"):
            install(source, "../evil", dest_root)

    def test_install_invalid_name_uppercase_raises(self, tmp_path):
        """install 名含大写/空格 → ValueError。"""
        source = tmp_path / "src"
        source.mkdir()
        _write_skill_md(source / "SKILL.md")
        dest_root = tmp_path / "dest"
        dest_root.mkdir()
        with pytest.raises(ValueError, match="invalid skill name"):
            install(source, "Evil Name", dest_root)

    def test_parse_skill_file_crlf_frontmatter(self, tmp_path):
        """CRLF 换行的 frontmatter 也能解析。"""
        skill_dir = tmp_path / "crlf-skill"
        skill_dir.mkdir()
        content = "---\r\nname: crlf-test\r\ndescription: CRLF test\r\n---\r\n\r\nbody\r\n"
        (skill_dir / "SKILL.md").write_text(content, encoding="utf-8")
        rec = parse_skill_file(skill_dir / "SKILL.md")
        assert rec.name == "crlf-test"
        assert rec.description == "CRLF test"
