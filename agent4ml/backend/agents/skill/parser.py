"""SKILL.md YAML frontmatter 解析 + .skill_id sidecar + install。

INVARIANT:
- .skill_id sidecar 持久 id（仅 IMPORTED/EVOLVED）：首次生成写文件，已存在则读（目录改名 id 不变）
- IMPORTED: {name}__imp_{uuid8}；EVOLVED: {name}__v{generation}_{uuid8}
- BUILTIN: {name}__builtin（确定性，无 sidecar，核心 skill 随包 id 可复现）
- frontmatter 必需 name + description，缺则 ValueError
- allowed-tools YAML list → tuple；缺省 ()
- enabled 缺省 True
- content_hash = sha256(SKILL.md 全文)[:16]
- SkillRecord.lineage.origin 由 parse_skill_file origin 参数定（IMPORTED/BUILTIN；版本演进走 store.create_version）
"""
from __future__ import annotations

import hashlib
import re
import shutil
import uuid
from pathlib import Path

import yaml

from agent4ml.backend.agents.skill.types import SkillLineage, SkillRecord

_SKILL_ID_FILE = ".skill_id"
_FRONTMATTER_RE = re.compile(r"^---\r?\n(.*?)\r?\n---\r?\n(.*)", re.DOTALL)


def _generate_skill_id(name: str, origin: str, generation: int) -> str:
    """按 origin 生成 skill_id。

    BUILTIN → {name}__builtin（确定性）；IMPORTED → {name}__imp_{uuid8}；
    EVOLVED → {name}__v{gen}_{uuid8}。
    """
    if origin == "BUILTIN":
        return f"{name}__builtin"
    short = uuid.uuid4().hex[:8]
    if origin == "IMPORTED":
        return f"{name}__imp_{short}"
    return f"{name}__v{generation}_{short}"


def read_or_create_skill_id(
    skill_dir: Path, name: str, origin: str = "IMPORTED", generation: int = 0
) -> str:
    """读 .skill_id sidecar；不存在则生成并写。

    BUILTIN origin：确定性 id `{name}__builtin`，不读不写 sidecar（核心 skill 随包，id 可复现）。
    IMPORTED/EVOLVED：sidecar 持久（目录改名 id 不变）。
    """
    if origin == "BUILTIN":
        return _generate_skill_id(name, origin, generation)
    sidecar = skill_dir / _SKILL_ID_FILE
    if sidecar.exists():
        return sidecar.read_text(encoding="utf-8").strip()
    skill_id = _generate_skill_id(name, origin, generation)
    sidecar.write_text(skill_id, encoding="utf-8")
    return skill_id


def parse_skill_file(skill_file: Path, origin: str = "IMPORTED") -> SkillRecord:
    """解析 SKILL.md → SkillRecord。

    frontmatter: ---\\n{yaml}\\n---\\n{body}
    必需 name + description；可选 allowed-tools / enabled / related-skills。
    origin: IMPORTED（用户 skill，sidecar）| BUILTIN（核心 skill，确定性 id）。
    """
    content = skill_file.read_text(encoding="utf-8")
    match = _FRONTMATTER_RE.match(content)
    if not match:
        raise ValueError(
            f"SKILL.md {skill_file} missing YAML frontmatter (expected '---\\n...\\n---\\n')"
        )
    fm_raw, _body = match.group(1), match.group(2)
    try:
        fm = yaml.safe_load(fm_raw) or {}
    except yaml.YAMLError as exc:
        raise ValueError(f"SKILL.md {skill_file} frontmatter YAML parse error: {exc}") from exc
    if not isinstance(fm, dict):
        raise ValueError(f"SKILL.md {skill_file} frontmatter must be a mapping, got {type(fm).__name__}")

    name = fm.get("name")
    description = fm.get("description")
    if not name:
        raise ValueError(f"SKILL.md {skill_file} frontmatter missing required field 'name'")
    if not description:
        raise ValueError(f"SKILL.md {skill_file} frontmatter missing required field 'description'")

    allowed_tools_raw = fm.get("allowed-tools") or []
    allowed_tools = tuple(allowed_tools_raw) if allowed_tools_raw else ()
    enabled = bool(fm.get("enabled", True))

    skill_dir = skill_file.parent
    skill_id = read_or_create_skill_id(skill_dir, name, origin=origin)
    content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]

    return SkillRecord(
        skill_id=skill_id,
        name=name,
        path=str(skill_file),
        content_hash=content_hash,
        description=description,
        allowed_tools=allowed_tools,
        enabled=enabled,
        lineage=SkillLineage(origin=origin),
    )


def install(source_dir: Path, name: str, dest_root: Path) -> str:
    """拷 source_dir → dest_root/{name}/，解析 SKILL.md 注册，返回 skill_id。

    name 只允许 [a-z0-9-]+，拒绝 `../` 或绝对路径逃逸。
    """
    if not re.fullmatch(r"[a-z0-9-]+", name):
        raise ValueError(f"invalid skill name: {name!r}")
    dest_dir = dest_root / name
    if dest_dir.exists():
        shutil.rmtree(dest_dir)
    shutil.copytree(source_dir, dest_dir)
    skill_file = dest_dir / "SKILL.md"
    if not skill_file.exists():
        raise FileNotFoundError(f"installed skill dir {dest_dir} has no SKILL.md")
    record = parse_skill_file(skill_file)
    return record.skill_id
