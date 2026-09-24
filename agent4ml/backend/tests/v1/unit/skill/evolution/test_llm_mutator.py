"""E5 LLMMutator 单测 — FIX 编辑 + CAPTURED 生成 + budget 截断 + frontmatter 保留。"""
from __future__ import annotations

import pytest

from agent4ml.backend.agents.skill.evolution.mutators.llm_mutator import LLMMutator
from agent4ml.backend.agents.skill.evolution.types import EvolutionContext
from agent4ml.backend.agents.skill.types import SkillLineage, SkillRecord


class _FakeLLM:
    def __init__(self, content: str):
        self._content = content

    def invoke(self, messages):
        return type("R", (), {"content": self._content})()


def _write_skill(tmp_path, name="sv", body="# Steps\n1. old step\n2. verify\n") -> str:
    d = tmp_path / name
    d.mkdir(parents=True, exist_ok=True)
    p = d / "SKILL.md"
    p.write_text(
        f"---\nname: {name}\ndescription: verify sources\nallowed-tools:\n  - web_search\n---\n{body}",
        encoding="utf-8",
    )
    return str(p)


def _baseline(path: str, name: str = "sv") -> SkillRecord:
    return SkillRecord(
        skill_id=f"{name}__imp_a1b2", name=name, path=path, content_hash="h_orig",
        lineage=SkillLineage(generation=0, origin="IMPORTED"),
        description="verify sources", allowed_tools=("web_search",),
    )


def _fix_ctx(baseline: SkillRecord, direction: str = "步骤1措辞不清") -> EvolutionContext:
    return EvolutionContext(
        trigger="METRIC", evolution_type="FIX",
        target_skill=baseline, fix_direction=direction,
    )


# ── FIX ─────────────────────────────────────────────────


def test_fix_edits_body_preserves_frontmatter(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = _write_skill(tmp_path)
    baseline = _baseline(path)
    ctx = _fix_ctx(baseline)
    # LLM 返编辑后 body（改步骤1）
    mutator = LLMMutator(max_changed_lines=20, llm=_FakeLLM("# Steps\n1. new improved step\n2. verify\n"))
    candidate, diff = mutator.mutate(ctx)

    # candidate is_active=False（champion 隔离）
    assert candidate.is_active is False
    assert candidate.lineage.origin == "FIXED"
    assert candidate.lineage.parent_skill_ids == (baseline.skill_id,)
    assert candidate.lineage.generation == 1
    # frontmatter 保留（name/allowed-tools 不变）
    content = open(candidate.path, encoding="utf-8").read()
    assert "name: sv" in content
    assert "web_search" in content
    # body 改了
    assert "new improved step" in content
    assert "old step" not in content
    # diff 非空
    assert diff != ""
    assert "new improved step" in diff


def test_fix_no_llm_returns_unchanged(tmp_path, monkeypatch):
    """LLM=None → 不变异（返 baseline body 作 candidate，diff 空）。"""
    monkeypatch.chdir(tmp_path)
    path = _write_skill(tmp_path)
    baseline = _baseline(path)
    ctx = _fix_ctx(baseline)
    mutator = LLMMutator(llm=None)
    candidate, diff = mutator.mutate(ctx)
    # body 未变
    content = open(candidate.path, encoding="utf-8").read()
    assert "old step" in content
    assert diff == ""


# ── budget 截断 ─────────────────────────────────────────


def test_enforce_budget_truncates():
    """超 budget → partial apply 前 N 改动，超出回退原。"""
    orig = "a\nb\nc\nd\ne"
    new = "a\nX\nY\nd\ne"  # replace b→X + insert Y = 2 改动
    enforced = LLMMutator._enforce_budget(orig, new, budget=1)
    # 只 apply 第 1 改动（b→X），drop +Y，保留 c
    assert enforced == "a\nX\nc\nd\ne"


def test_enforce_budget_within_limit_unchanged():
    """未超 budget → 原 new_body。"""
    orig = "a\nb\nc"
    new = "a\nX\nc"  # 1 改动
    assert LLMMutator._enforce_budget(orig, new, budget=5) == new


def test_fix_budget_truncates_large_edit(tmp_path, monkeypatch):
    """LLM 大改超 budget → 截断。"""
    monkeypatch.chdir(tmp_path)
    path = _write_skill(tmp_path, body="l0\nl1\nl2\nl3\nl4\n")
    baseline = _baseline(path)
    ctx = _fix_ctx(baseline)
    # LLM 改 3 行（超 budget=1）
    mutator = LLMMutator(max_changed_lines=1, llm=_FakeLLM("l0\nX1\nX2\nl3\nl4\n"))
    candidate, diff = mutator.mutate(ctx)
    content = open(candidate.path, encoding="utf-8").read()
    # 只 apply 1 改动（l1→X1），l2 保留
    assert "X1" in content
    assert "l2" in content  # 未被改


# ── CAPTURED ────────────────────────────────────────────


def test_capture_generates_new_skill(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ctx = EvolutionContext(
        trigger="CAPTURE", evolution_type="CAPTURED", target_skill=None,
        capture_pattern="信源交叉核验", suggested_name="source-cross-check",
    )
    new_skill_md = "---\nname: source-cross-check\ndescription: 交叉核验信源\n---\n# Steps\n1. 找2源\n"
    mutator = LLMMutator(llm=_FakeLLM(new_skill_md))
    candidate, diff = mutator.mutate(ctx)

    assert candidate.is_active is False
    assert candidate.lineage.origin == "CAPTURED"
    assert candidate.lineage.parent_skill_ids == ()  # 无 parent
    assert candidate.lineage.generation == 0
    assert candidate.name == "source-cross-check"
    content = open(candidate.path, encoding="utf-8").read()
    assert "交叉核验信源" in content
    assert "CAPTURED" in diff


def test_capture_no_llm_raises(tmp_path, monkeypatch):
    """CAPTURED 必须 LLM 生成，无 LLM 抛错。"""
    monkeypatch.chdir(tmp_path)
    ctx = EvolutionContext(
        trigger="CAPTURE", evolution_type="CAPTURED", target_skill=None,
        capture_pattern="模式", suggested_name="new-skill",
    )
    mutator = LLMMutator(llm=None)
    with pytest.raises(ValueError, match="CAPTURED 需要 LLM"):
        mutator.mutate(ctx)


def test_capture_invalid_name_raises(tmp_path, monkeypatch):
    """LLM 产非法 name → 抛。"""
    monkeypatch.chdir(tmp_path)
    ctx = EvolutionContext(
        trigger="CAPTURE", evolution_type="CAPTURED", target_skill=None,
        capture_pattern="模式", suggested_name="bad",
    )
    bad_md = "---\nname: ../escape\ndescription: d\n---\nbody\n"
    mutator = LLMMutator(llm=_FakeLLM(bad_md))
    with pytest.raises(ValueError, match="invalid skill name"):
        mutator.mutate(ctx)


# ── 不支持类型 ──────────────────────────────────────────


def test_unsupported_evolution_type_raises(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = _write_skill(tmp_path)
    baseline = _baseline(path)
    ctx = EvolutionContext(
        trigger="METRIC", evolution_type="DERIVED", target_skill=baseline,
    )
    mutator = LLMMutator(llm=_FakeLLM("x"))
    with pytest.raises(ValueError, match="unsupported evolution_type"):
        mutator.mutate(ctx)


# ── Mutator Protocol ────────────────────────────────────


def test_llm_mutator_is_mutator_protocol():
    from agent4ml.backend.agents.skill.evolution.protocols import Mutator
    assert isinstance(LLMMutator(), Mutator)
