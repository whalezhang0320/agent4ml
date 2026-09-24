"""Skill 自进化层值对象 — frozen dataclass + Literal 枚举。

INVARIANT:
- 全部 frozen（不可变值对象）
- EvolutionContext.target_skill: FIX/DERIVED 有；CAPTURED 无（新 skill）
- EvalResult.score [0,1]；hard_failures 标关键失败（非空/JSON 烂等）
- GateDecision.recommendation: accept_new_best/accept/reject/pending_human
- EvolutionRecord: 实验记录，写 store.skill_evolutions 表，可审计
- 进化逻辑唯一 LLM 推理（RL/计算式不适用，见 37.md D-L2-3）

承接 37.md §4.1 接口签名。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from agent4ml.backend.agents.skill.types import SkillMetrics, SkillRecord

# ── 枚举（Literal） ──────────────────────────────────────
TriggerType = Literal["METRIC", "CAPTURE", "PERIODIC", "TOOL_DEGRADATION", "ANALYSIS"]
EvolutionType = Literal["FIX", "DERIVED", "CAPTURED"]
FailureClass = Literal["FUNDAMENTAL", "IMPLEMENTATION"]
GateRecommendation = Literal["accept_new_best", "accept", "reject", "pending_human"]
EvalMetric = Literal["hard", "soft", "mixed"]
EvalEvidenceKind = Literal["programmatic_rule", "longitudinal_pair", "judge_reason"]


@dataclass(frozen=True)
class FailureEvidence:
    """聚焦后的单条失败证据（IVEFocuser 产出）。"""

    turn_index: int | None              # 失败 turn（PERIODIC/METRIC 为 None）
    tool_name: str | None               # 失败工具
    failure_class: FailureClass         # IVE 分类
    description: str                    # 失败描述
    impl_fail_count: int = 0            # implementation 累计次数（≥3 升级 fundamental）


@dataclass(frozen=True)
class EvolutionContext:
    """自进化统一上下文。Trigger 产出，Focuser 增强，Mutator 消费。

    FIX/DERIVED: target_skill 有值（待进化的现有 skill）
    CAPTURED: target_skill=None（生成新 skill），capture_pattern 描述模式
    """

    trigger: TriggerType
    evolution_type: EvolutionType
    target_skill: SkillRecord | None
    failure_evidence: tuple[FailureEvidence, ...] = ()
    fix_direction: str = ""             # IVE 第 5 问给的修复方向
    capture_pattern: str = ""           # CAPTURED：可复用模式描述
    suggested_name: str = ""            # CAPTURED：建议 skill name
    recent_analyses: tuple[str, ...] = ()  # 历史分析摘要（防重复，借鉴 OpenSpace _ANALYSIS_CONTEXT_MAX）


@dataclass(frozen=True)
class EvalContext:
    """eval 调用上下文。第二层传给 EvalBridge。"""

    baseline: SkillRecord               # 当前激活版（CAPTURED 时为 None 占位，用 placeholder）
    candidate: SkillRecord              # 变异后的候选版（is_active=False）
    metrics_baseline: SkillMetrics | None = None  # baseline 的 quality metrics
    replay_samples: tuple[Any, ...] = ()  # MVP 空（longitudinal 重放留 L3）
    task_domain: str | None = None


@dataclass(frozen=True)
class EvalEvidence:
    """eval 证据链。2a 仅 programmatic_rule；longitudinal_pair 留 L3。"""

    kind: EvalEvidenceKind
    rule_name: str                      # 如 "nonempty" / "semantic_density"
    baseline_pass: bool
    candidate_pass: bool
    detail: str = ""


@dataclass(frozen=True)
class EvalResult:
    """eval 结果。EvalBridge 返回，PromotionGate 消费。

    score: [0, 1] programmatic 通过率
    hard_failures: 触发 hard fail 的 rule 名（非空/JSON 烂等关键失败）
    confidence: programmatic ~0.7（L3 LLM-judge ~0.46 不可靠）
    """

    score: float
    metric: EvalMetric = "hard"
    hard_failures: tuple[str, ...] = ()
    evidence: tuple[EvalEvidence, ...] = ()
    confidence: float = 0.7
    recommendation: GateRecommendation = "reject"


@dataclass(frozen=True)
class GateDecision:
    """PromotionGate 决策。"""

    recommendation: GateRecommendation
    reason: str
    new_version_id: str | None = None   # accept 时的新 skill_id


@dataclass(frozen=True)
class EvolutionRecord:
    """实验记录。写 store.skill_evolutions 表，可审计。

    baseline_id: FIX/DERIVED 有；CAPTURED 无（新 skill）
    created_version_id: accept 时的新 skill_id；reject/pending 时 None
    """

    evolution_id: str
    skill_name: str
    evolution_type: EvolutionType
    trigger: TriggerType
    baseline_id: str | None
    candidate_id: str
    failure_focus: str                  # IVE 诊断摘要 / CAPTURED 模式描述
    mutation_diff: str                  # Mutator 产出的 diff
    eval_score: float
    gate_decision: GateRecommendation
    created_version_id: str | None = None
    timestamp: str = ""
