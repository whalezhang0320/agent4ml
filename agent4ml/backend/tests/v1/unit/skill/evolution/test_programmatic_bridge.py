"""E6 ProgrammaticEvalBridge 单测 — 7 mode + semantic_density + score + hard_failures。"""
from __future__ import annotations

from pathlib import Path

from agent4ml.backend.agents.skill.evolution.eval.programmatic_bridge import (
    ProgrammaticEvalBridge,
)
from agent4ml.backend.agents.skill.evolution.protocols import EvalBridge
from agent4ml.backend.agents.skill.evolution.types import EvalContext
from agent4ml.backend.agents.skill.types import SkillLineage, SkillRecord


_GOOD_SKILL = """---
name: sv
description: 验证信源
---

## 核心要点
验证信源可信度。来源：https://example.com。

## 步骤
1. 找 2 源
2. 比对

MUST 交叉核验。NEVER 单源采信。
"""

_BAD_SKILL_EMPTY = """---
name: bad
description: d
---
"""

_BAD_SKILL_UNPARSEABLE = """---
"unclosed string
---
body
"""

_BAD_SKILL_UNFOUNDED = """---
name: bad
description: d
---
绝对正确。一定有效。毫无疑问。
"""


def _rec(content: str, tmp_path, name: str = "sv") -> SkillRecord:
    p = tmp_path / f"{name}.md"
    p.write_text(content, encoding="utf-8")
    return SkillRecord(
        skill_id=f"{name}__imp", name=name, path=str(p), content_hash="h",
        lineage=SkillLineage(generation=0, origin="IMPORTED"),
    )


def _ctx(baseline_content: str, candidate_content: str, tmp_path) -> EvalContext:
    baseline = _rec(baseline_content, tmp_path, "baseline")
    candidate = _rec(candidate_content, tmp_path, "candidate")
    return EvalContext(baseline=baseline, candidate=candidate)


# ── Protocol ────────────────────────────────────────────


def test_is_eval_bridge_protocol():
    assert isinstance(ProgrammaticEvalBridge(), EvalBridge)


# ── 7 mode + score ──────────────────────────────────────


def test_good_skill_high_score(tmp_path):
    ctx = _ctx(_GOOD_SKILL, _GOOD_SKILL, tmp_path)
    result = ProgrammaticEvalBridge().evaluate(ctx)
    assert result.score > 0.5
    assert "nonempty" not in result.hard_failures
    assert "json_parseable" not in result.hard_failures
    assert result.recommendation == "accept"


def test_empty_body_hard_failure(tmp_path):
    """空 body → nonempty hard_failure → reject。"""
    ctx = _ctx(_GOOD_SKILL, _BAD_SKILL_EMPTY, tmp_path)
    result = ProgrammaticEvalBridge().evaluate(ctx)
    assert "nonempty" in result.hard_failures
    assert result.recommendation == "reject"


def test_unparseable_frontmatter_hard_failure(tmp_path):
    """frontmatter 不可解析 → json_parseable hard_failure → reject。"""
    ctx = _ctx(_GOOD_SKILL, _BAD_SKILL_UNPARSEABLE, tmp_path)
    result = ProgrammaticEvalBridge().evaluate(ctx)
    assert "json_parseable" in result.hard_failures
    assert result.recommendation == "reject"


def test_unfounded_claims_fails_soft(tmp_path):
    """绝对化声明 → no_unfounded_claims fail（软，非 hard）。"""
    ctx = _ctx(_GOOD_SKILL, _BAD_SKILL_UNFOUNDED, tmp_path)
    result = ProgrammaticEvalBridge().evaluate(ctx)
    # no_unfounded_claims 不在 hard_failures
    assert "no_unfounded_claims" not in result.hard_failures
    # 但该 mode fail
    evidence = {e.rule_name: e.candidate_pass for e in result.evidence}
    assert evidence["no_unfounded_claims"] is False


# ── 各 mode ─────────────────────────────────────────────


def test_check_nonempty():
    assert ProgrammaticEvalBridge._check_nonempty("---\nn: x\n---\nbody")
    assert not ProgrammaticEvalBridge._check_nonempty("---\nn: x\n---\n")


def test_check_json_parseable():
    assert ProgrammaticEvalBridge._check_json_parseable("---\nname: x\n---\nbody")
    assert ProgrammaticEvalBridge._check_json_parseable("no frontmatter body")  # 无 frontmatter pass
    # 无效 YAML（未闭合引号）→ fail
    assert not ProgrammaticEvalBridge._check_json_parseable('---\n"unclosed\n---\nbody')


def test_check_must_cite():
    assert ProgrammaticEvalBridge._check_must_cite("来源：https://example.com")
    assert ProgrammaticEvalBridge._check_must_cite("cite @author")
    assert not ProgrammaticEvalBridge._check_must_cite("普通文本无任何标记")


def test_check_paragraph_limit():
    long_body = "---\nn: x\n---\n" + "\n\n".join(f"para {i}" for i in range(30))
    assert not ProgrammaticEvalBridge._check_paragraph_limit(long_body)
    assert ProgrammaticEvalBridge._check_paragraph_limit("---\nn: x\n---\n短 body")


def test_check_lead_with_conclusion():
    assert ProgrammaticEvalBridge._check_lead_with_conclusion("---\nn: x\n---\n## 核心要点\n正文")
    assert ProgrammaticEvalBridge._check_lead_with_conclusion("---\nn: x\n---\nconclusion: x")
    assert not ProgrammaticEvalBridge._check_lead_with_conclusion("---\nn: x\n---\n## 步骤\n1. x")


def test_check_no_unfounded_claims():
    assert ProgrammaticEvalBridge._check_no_unfounded_claims("正常文本")
    assert not ProgrammaticEvalBridge._check_no_unfounded_claims("绝对正确")
    assert not ProgrammaticEvalBridge._check_no_unfounded_claims("definitely works")


# ── semantic_density ────────────────────────────────────


def test_semantic_density_zero():
    assert ProgrammaticEvalBridge._semantic_density("") == 0.0
    assert ProgrammaticEvalBridge._semantic_density("普通文本无指令词") == 0.0


def test_semantic_density_nonzero():
    density = ProgrammaticEvalBridge._semantic_density("you MUST verify. NEVER trust one source.")
    assert density > 0.0


# ── 零 LLM ──────────────────────────────────────────────


def test_zero_llm(tmp_path):
    """evaluate 无 LLM 调用（纯正则/解析）。"""
    ctx = _ctx(_GOOD_SKILL, _GOOD_SKILL, tmp_path)
    # 无 LLM 注入，evaluate 不调用任何 LLM
    result = ProgrammaticEvalBridge().evaluate(ctx)
    assert result.score >= 0.0  # 不抛即零 LLM


# ── score 计算 ──────────────────────────────────────────


def test_score_is_pass_ratio(tmp_path):
    """score = passed / total（contract-aware，规则数随 skill 文本变化）。"""
    ctx = _ctx(_GOOD_SKILL, _GOOD_SKILL, tmp_path)
    result = ProgrammaticEvalBridge().evaluate(ctx)
    total = len(result.evidence)
    passed = sum(1 for e in result.evidence if e.candidate_pass)
    assert total > 0
    assert abs(result.score - passed / total) < 0.01


def test_evidence_has_baseline_and_candidate(tmp_path):
    ctx = _ctx(_GOOD_SKILL, _GOOD_SKILL, tmp_path)
    result = ProgrammaticEvalBridge().evaluate(ctx)
    for e in result.evidence:
        assert hasattr(e, "baseline_pass")
        assert hasattr(e, "candidate_pass")
        assert e.kind == "programmatic_rule"
