"""Skill eval 确定性检查函数 — 纯函数，无类，无循环依赖。

被 ResponseContractChecker 和 ProgrammaticEvalBridge（facade）共同复用。
单一真相源——check 逻辑只在此模块定义。
"""
from __future__ import annotations

import re
from pathlib import Path

# hard failure modes（关键失败，触发即 reject 倾向）
HARD_MODES = ("nonempty", "json_parseable")

# 指令性词（semantic_density，借鉴 SkillOpt）
DIRECTIVE_WORDS = ("MUST", "ALWAYS", "NEVER", "SHOULD", "MUST NOT", "REQUIRED", "FORBIDDEN")
UNFOUNDED_WORDS = ("绝对", "一定", "必然", "毫无疑问", "absolutely", "definitely", "certainly")
CONCLUSION_WORDS = ("结论", "总结", "核心", "要点", "conclusion", "summary", "key")
CITE_PATTERN = re.compile(r"(https?://|@[\w-]+|来源|引用|source|cite)", re.IGNORECASE)
YAML_FRONTMATTER = re.compile(r"^---\r?\n(.*?)\r?\n---\r?\n", re.DOTALL)

PARAGRAPH_LIMIT = 20
SEMANTIC_DENSITY_MIN = 0.005
SEMANTIC_DENSITY_MAX = 0.15


def read_content(record) -> str:
    """读 SKILL.md 全文。record.path 文件。失败返空。"""
    try:
        return Path(record.path).read_text(encoding="utf-8")
    except Exception:
        return ""


def split_body(content: str) -> str:
    """去 frontmatter，返 body。"""
    m = YAML_FRONTMATTER.match(content)
    if m:
        return content[m.end():]
    return content


def check_nonempty(content: str) -> bool:
    return len(split_body(content).strip()) > 0


def check_json_parseable(content: str) -> bool:
    """frontmatter YAML 可解析。无 frontmatter 也算 pass。"""
    m = YAML_FRONTMATTER.match(content)
    if not m:
        return True
    try:
        import yaml
        yaml.safe_load(m.group(1))
        return True
    except Exception:
        return False


def check_must_cite(content: str) -> bool:
    return bool(CITE_PATTERN.search(content))


def check_paragraph_limit(content: str) -> bool:
    body = split_body(content)
    paragraphs = [p for p in body.split("\n\n") if p.strip()]
    return len(paragraphs) <= PARAGRAPH_LIMIT


def check_lead_with_conclusion(content: str) -> bool:
    body = split_body(content).strip()
    if not body:
        return False
    paras = body.split("\n\n")[:3]
    head = "\n\n".join(paras)
    return any(w.lower() in head.lower() for w in CONCLUSION_WORDS)


def check_no_unfounded_claims(content: str) -> bool:
    return not any(w in content for w in UNFOUNDED_WORDS)


def semantic_density(content: str) -> float:
    """指令性词密度 = 指令词出现次数 / 总词数。"""
    if not content:
        return 0.0
    words = re.findall(r"\w+", content)
    if not words:
        return 0.0
    count = sum(content.upper().count(w) for w in DIRECTIVE_WORDS)
    return count / len(words)
