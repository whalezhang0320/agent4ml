"""Skill 注入文本构建 — markdown block，从 SKILL.md 文件读 body（去 frontmatter）。

INVARIANT:
- 内容/索引分离：SkillRecord 只存 path，body 从文件读（source of truth 在文件）
- frontmatter 剥离：---\n{yaml}\n---\n{body} → 取 body
- 文件读失败返空 body（不抛，注入 header 即可）
"""
from __future__ import annotations

from pathlib import Path

from agent4ml.backend.agents.skill.types import SkillRecord


def build_injection_text(skills: list[SkillRecord]) -> str:
    """构建 active skills 的 markdown 注入块。无 skill 返空串。"""
    if not skills:
        return ""
    lines: list[str] = ["# Active Skills", ""]
    for rec in skills:
        body = _read_body(rec.path)
        lines.append(f"### Skill: {rec.name}")
        lines.append(f"**Path**: {rec.path}")
        lines.append("")
        lines.append(body.strip())
        lines.append("")
        lines.append("---")
        lines.append("")
    return "\n".join(lines)


def _read_body(path: str) -> str:
    """读 SKILL.md，剥离 frontmatter 返 body。失败返空。"""
    try:
        content = Path(path).read_text(encoding="utf-8")
    except Exception:
        return ""
    if content.startswith("---"):
        parts = content.split("---", 2)
        if len(parts) >= 3:
            return parts[2].lstrip("\r\n")
    return content
