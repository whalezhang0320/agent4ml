"""Skill eval 评估层值对象 — frozen dataclass + Literal 枚举。

INVARIANT:
- 全部 frozen（不可变值对象）
- SkillJudgment: per-skill per-task 执行层判断（OpenSpace SkillJudgment 模型）
- TaskQualityScore: 任务层 4 维加权评分（SkillClaw session_judge 模型，权重 0.50/0.35/0.05/0.10）
- ContractRule: 响应层从 skill 文本编译的规则（AutoSkill EvalCompiler 模型）
- SkillHealthReport: 趋势层健康报告（RuntimeTracker 产出）
- EvalRun: 三层 eval 统一审计记录
- EvolutionSuggestion: 执行层 LLM 分析产出的进化建议，喂给 L2 trigger

承接 38-skill-eval-layer-design.md §4.1 接口签名。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from agent4ml.backend.agents.skill.evolution.types import EvolutionType

# ── 枚举（Literal） ──────────────────────────────────────
EvalLayer = Literal["execution", "task", "response"]
Trend = Literal["improving", "stable", "degrading", "insufficient_data"]
ContractRuleKind = Literal["programmatic", "llm_binary"]


@dataclass(frozen=True)
class SkillJudgment:
    """执行层：per-skill per-task 有效性判断（OpenSpace SkillJudgment 模型）。

    skill_applied: agent 是否实际应用了 skill 指导（LLM 判断，比 middleware 粗粒度打点更准）
    deviation_note: 偏差记录（如"跳过了 validate quality 步骤"）
    """

    judgment_id: str
    skill_id: str
    skill_name: str
    task_id: str
    skill_applied: bool
    deviation_note: str = ""
    timestamp: str = ""


@dataclass(frozen=True)
class EvolutionSuggestion:
    """执行层 LLM 分析产出的进化建议（OpenSpace EvolutionSuggestion 模型）。

    SkillJudgmentAnalyzer 一次 LLM 调用同时产 SkillJudgment + EvolutionSuggestion。
    suggestion 喂给 L2 trigger 作补充信号。
    """

    evolution_type: EvolutionType          # FIX / DERIVED / CAPTURED
    target_skill_ids: tuple[str, ...] = ()  # FIX/DERIVED 有；CAPTURED 无（新 skill）
    direction: str = ""                    # 自由文本：该修什么 / 该捕获什么


@dataclass(frozen=True)
class TaskQualityScore:
    """任务层：4 维加权评分（SkillClaw session_judge 模型）。

    权重 D-L3-13: 0.50*completion + 0.35*quality + 0.05*efficiency + 0.10*tool
    """

    score_id: str
    task_id: str
    task_completion: float          # [0, 1]
    response_quality: float         # [0, 1]
    efficiency: float               # [0, 1]
    tool_usage: float               # [0, 1]
    overall_score: float
    rationale: str = ""
    timestamp: str = ""


@dataclass(frozen=True)
class ContractRule:
    """响应层：从 skill 文本编译的 contract 规则（AutoSkill EvalCompiler 模型）。

    hard: True → 失败触发 hard_failure（reject 倾向）
    kind: programmatic（确定性检查）| llm_binary（LLM 判断）
    """

    rule_id: str                     # "nonempty" / "must_cite" / "json_parseable" / ...
    kind: ContractRuleKind
    hard: bool
    description: str = ""
    params: dict = field(default_factory=dict)


@dataclass(frozen=True)
class SkillHealthReport:
    """趋势层：skill 健康报告（RuntimeTracker 产出）。

    trend: improving/stable/degrading/insufficient_data
    recent_judgments: 近 N 条 SkillJudgment（偏差记录）
    advice: 文字建议
    """

    skill_id: str
    skill_name: str
    window_selections: int
    applied_rate: float
    completion_rate: float
    effective_rate: float
    fallback_rate: float
    trend: Trend
    recent_judgments: tuple[SkillJudgment, ...] = ()
    advice: str = ""


@dataclass(frozen=True)
class EvalRun:
    """一次 eval 运行的审计记录。写 skill_eval_runs 表。"""

    eval_run_id: str
    eval_layer: EvalLayer            # "execution" / "task" / "response"
    skill_ids: tuple[str, ...]
    candidate_id: str | None = None  # response 层有
    baseline_id: str | None = None   # response 层有
    result_json: str = ""
    timestamp: str = ""
