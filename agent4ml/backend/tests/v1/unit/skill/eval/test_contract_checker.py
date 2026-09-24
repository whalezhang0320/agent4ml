"""L3-E3 单测：ContractCompiler + ResponseContractChecker。

验证：
- contract-aware：skill 没提"引用"就不编译 must_cite
- 硬失败 short-circuit
- 外部 skill 零改造（不需 frontmatter eval-contract）
- score 计算
- 复用 L2a check 函数
"""
from __future__ import annotations

import pytest

from agent4ml.backend.agents.skill.eval.analyzers.contract_compiler import ContractCompiler
from agent4ml.backend.agents.skill.eval.analyzers.response_contract_checker import (
    ResponseContractChecker,
)


# ── ContractCompiler ────────────────────────────────────

def test_compiler_always_has_global_hard_rules():
    rules = ContractCompiler().compile("# Some skill\n\nbody")
    rule_ids = [r.rule_id for r in rules]
    assert "nonempty" in rule_ids
    assert "json_parseable" in rule_ids
    assert rules[0].hard is True
    assert rules[1].hard is True


def test_compiler_always_has_soft_rules():
    rules = ContractCompiler().compile("# Some skill\n\nbody")
    rule_ids = [r.rule_id for r in rules]
    assert "no_unfounded_claims" in rule_ids
    assert "semantic_density" in rule_ids


def test_compiler_no_cite_when_skill_doesnt_mention():
    """skill 没提引用 → 不编译 must_cite。"""
    rules = ContractCompiler().compile("# Question Decomposition\n\n分解研究问题")
    rule_ids = [r.rule_id for r in rules]
    assert "must_cite" not in rule_ids


def test_compiler_compiles_must_cite_when_skill_mentions():
    """skill 提到"引用来源" → 编译 must_cite。"""
    rules = ContractCompiler().compile("# Source Verification\n\n验证信源，需引用来源。")
    rule_ids = [r.rule_id for r in rules]
    assert "must_cite" in rule_ids


def test_compiler_no_conclusion_when_not_mentioned():
    rules = ContractCompiler().compile("# Some skill\n\nbody text")
    rule_ids = [r.rule_id for r in rules]
    assert "lead_with_conclusion" not in rule_ids


def test_compiler_compiles_conclusion_when_mentioned():
    rules = ContractCompiler().compile("# Report Skill\n\n先给结论再展开。")
    rule_ids = [r.rule_id for r in rules]
    assert "lead_with_conclusion" in rule_ids


def test_compiler_paragraph_limit_extraction():
    rules = ContractCompiler().compile("# Skill\n\n输出不超过3段。")
    para_rules = [r for r in rules if r.rule_id == "paragraph_limit"]
    assert len(para_rules) == 1
    assert para_rules[0].params["max"] == 3


def test_compiler_paragraph_limit_english():
    rules = ContractCompiler().compile("# Skill\n\nwithin 5 paragraphs")
    para_rules = [r for r in rules if r.rule_id == "paragraph_limit"]
    assert len(para_rules) == 1
    assert para_rules[0].params["max"] == 5


def test_compiler_no_paragraph_limit_when_not_mentioned():
    rules = ContractCompiler().compile("# Skill\n\n普通文本无段落数限制")
    rule_ids = [r.rule_id for r in rules]
    assert "paragraph_limit" not in rule_ids


def test_compiler_external_skill_zero_modification():
    """外部 skill 无 eval-contract frontmatter，纯文本扫描。"""
    content = """---
name: external-skill
description: An external skill
---
# External Skill

This skill guides research. Always cite sources for claims.
Lead with the conclusion first.
"""
    rules = ContractCompiler().compile(content)
    rule_ids = [r.rule_id for r in rules]
    assert "must_cite" in rule_ids  # "cite sources" in text
    assert "lead_with_conclusion" in rule_ids  # "Lead with the conclusion" in text


# ── ResponseContractChecker ─────────────────────────────

def test_checker_basic_pass():
    """正常 skill content → 全 pass，recommendation=accept。"""
    content = "# Skill\n\n这是 skill body，有内容。"
    result = ResponseContractChecker().check(content, "")
    assert result.hard_failures == ()
    assert result.recommendation == "accept"
    assert result.score > 0.0


def test_checker_nonempty_hard_failure():
    """空 body → nonempty 硬失败 → reject。"""
    content = "---\nname: empty\n---\n\n"
    result = ResponseContractChecker().check(content, "")
    assert "nonempty" in result.hard_failures
    assert result.recommendation == "reject"


def test_checker_frontmatter_hard_failure():
    """坏 frontmatter → json_parseable 硬失败 → reject。"""
    content = '---\nname: "unclosed\n---\n\nbody'
    result = ResponseContractChecker().check(content, "")
    assert "json_parseable" in result.hard_failures
    assert result.recommendation == "reject"


def test_checker_contract_aware_no_must_cite():
    """skill 没提引用 → 不查 must_cite → 不因缺引用扣分。"""
    content = "# Question Decomposition\n\n分解研究问题为子问题"
    result = ResponseContractChecker().check(content, "")
    rule_names = [e.rule_name for e in result.evidence]
    assert "must_cite" not in rule_names


def test_checker_contract_aware_with_must_cite():
    """skill 提了引用 → 查 must_cite。"""
    content = "# Source Verification\n\n验证信源需引用来源。\n来源：https://example.com"
    result = ResponseContractChecker().check(content, "")
    rule_names = [e.rule_name for e in result.evidence]
    assert "must_cite" in rule_names


def test_checker_score_is_pass_ratio():
    """score = passed / total。"""
    content = "# Skill\n\nbody"
    result = ResponseContractChecker().check(content, "")
    total = len(result.evidence)
    passed = sum(1 for e in result.evidence if e.candidate_pass)
    assert result.score == pytest.approx(passed / total)


def test_checker_evidence_has_baseline_pass():
    """baseline_content 非空时 evidence 含 baseline_pass。"""
    baseline = "# Skill\n\nold body"
    candidate = "# Skill\n\nnew body"
    result = ResponseContractChecker().check(candidate, baseline)
    for e in result.evidence:
        assert hasattr(e, "baseline_pass")


def test_checker_external_skill_zero_modification():
    """外部 skill（无 eval-contract）→ contract-aware 编译，不误判。"""
    content = """---
name: external-research
description: Research skill
---
# Research Skill

Guide the agent to research systematically.
"""
    result = ResponseContractChecker().check(content, "")
    # 没提引用/结论/段落数 → 不编译那些规则 → 不因缺引用/结论扣分
    assert "must_cite" not in [e.rule_name for e in result.evidence]
    assert "lead_with_conclusion" not in [e.rule_name for e in result.evidence]
    # 但全局硬规则仍跑
    assert "nonempty" in [e.rule_name for e in result.evidence]
    assert result.hard_failures == ()
