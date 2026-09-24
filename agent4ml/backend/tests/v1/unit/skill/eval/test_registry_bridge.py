"""L3-E4 单测：EvalRegistry + RegistryEvalBridge + facade 改造。

验证：
- RegistryEvalBridge 实现 EvalBridge Protocol
- fail-closed（checker 异常 → reject）
- facade 委托 ResponseContractChecker（contract-aware）
- L2a 既有测试不回归
"""
from __future__ import annotations

from pathlib import Path

import pytest

from agent4ml.backend.agents.skill.eval.analyzers.contract_compiler import ContractCompiler
from agent4ml.backend.agents.skill.eval.analyzers.response_contract_checker import (
    ResponseContractChecker,
)
from agent4ml.backend.agents.skill.eval.registry import (
    EvalRegistry,
    RegistryEvalBridge,
    build_default_registry,
)
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

MUST 交叉核验。NEVER 单源采信。
"""

_BAD_SKILL_EMPTY = """---
name: bad
description: d
---
"""


def _rec(content: str, tmp_path: Path, name: str = "sv") -> SkillRecord:
    p = tmp_path / f"{name}.md"
    p.write_text(content, encoding="utf-8")
    return SkillRecord(
        skill_id=f"{name}__imp", name=name, path=str(p), content_hash="h",
        lineage=SkillLineage(generation=0, origin="IMPORTED"),
    )


def _ctx(baseline: str, candidate: str, tmp_path: Path) -> EvalContext:
    return EvalContext(
        baseline=_rec(baseline, tmp_path, "baseline"),
        candidate=_rec(candidate, tmp_path, "candidate"),
    )


# ── EvalRegistry ────────────────────────────────────────

def test_registry_holds_contract_checker():
    checker = ResponseContractChecker(ContractCompiler())
    reg = EvalRegistry(checker)
    assert reg.get_contract_checker() is checker


def test_build_default_registry():
    reg = build_default_registry()
    assert isinstance(reg.get_contract_checker(), ResponseContractChecker)


# ── RegistryEvalBridge ──────────────────────────────────

def test_registry_bridge_is_eval_bridge_protocol():
    reg = build_default_registry()
    bridge = RegistryEvalBridge(reg)
    assert isinstance(bridge, EvalBridge)


def test_registry_bridge_evaluate_pass(tmp_path: Path):
    reg = build_default_registry()
    bridge = RegistryEvalBridge(reg)
    ctx = _ctx(_GOOD_SKILL, _GOOD_SKILL, tmp_path)
    result = bridge.evaluate(ctx)
    assert result.hard_failures == ()
    assert result.recommendation == "accept"
    assert result.score > 0.0


def test_registry_bridge_evaluate_hard_failure(tmp_path: Path):
    reg = build_default_registry()
    bridge = RegistryEvalBridge(reg)
    ctx = _ctx(_GOOD_SKILL, _BAD_SKILL_EMPTY, tmp_path)
    result = bridge.evaluate(ctx)
    assert "nonempty" in result.hard_failures
    assert result.recommendation == "reject"


def test_registry_bridge_fail_closed_on_exception():
    """checker 异常 → fail-closed reject，不裸跑。"""

    class ExplodingChecker(ResponseContractChecker):
        def check(self, candidate_content, baseline_content):
            raise RuntimeError("boom")

    reg = EvalRegistry(ExplodingChecker())
    bridge = RegistryEvalBridge(reg)

    # 构造一个最小 ctx（path 不存在也会被 _read_content 返空，但 checker 爆炸）
    from agent4ml.backend.agents.skill.evolution.types import EvalContext
    fake_rec = SkillRecord(
        skill_id="x", name="x", path="/nonexistent", content_hash="h",
    )
    ctx = EvalContext(baseline=fake_rec, candidate=fake_rec)
    result = bridge.evaluate(ctx)
    assert result.recommendation == "reject"
    assert "eval_exception" in result.hard_failures
    assert result.score == 0.0


def test_registry_bridge_uses_contract_aware(tmp_path: Path):
    """RegistryEvalBridge 走 contract-aware（skill 没提引用就不查 must_cite）。"""
    no_cite_skill = "# Question Decomposition\n\n分解研究问题"
    reg = build_default_registry()
    bridge = RegistryEvalBridge(reg)
    ctx = _ctx(no_cite_skill, no_cite_skill, tmp_path)
    result = bridge.evaluate(ctx)
    rule_names = [e.rule_name for e in result.evidence]
    assert "must_cite" not in rule_names  # contract-aware：没提引用不查


# ── facade（ProgrammaticEvalBridge）──────────────────────

def test_facade_is_eval_bridge_protocol():
    assert isinstance(ProgrammaticEvalBridge(), EvalBridge)


def test_facade_delegates_to_contract_aware(tmp_path: Path):
    """facade evaluate() 走 contract-aware ResponseContractChecker。"""
    no_cite_skill = "# Some Skill\n\n普通 skill 无引用声明"
    bridge = ProgrammaticEvalBridge()
    ctx = _ctx(no_cite_skill, no_cite_skill, tmp_path)
    result = bridge.evaluate(ctx)
    rule_names = [e.rule_name for e in result.evidence]
    assert "must_cite" not in rule_names  # contract-aware


def test_facade_hard_failure_nonempty(tmp_path: Path):
    bridge = ProgrammaticEvalBridge()
    ctx = _ctx(_GOOD_SKILL, _BAD_SKILL_EMPTY, tmp_path)
    result = bridge.evaluate(ctx)
    assert "nonempty" in result.hard_failures
    assert result.recommendation == "reject"


def test_facade_good_skill_accept(tmp_path: Path):
    bridge = ProgrammaticEvalBridge()
    ctx = _ctx(_GOOD_SKILL, _GOOD_SKILL, tmp_path)
    result = bridge.evaluate(ctx)
    assert result.hard_failures == ()
    assert result.recommendation == "accept"
    assert result.score > 0.5
