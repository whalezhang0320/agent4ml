"""L3 eval 类型层 — EvalContext + SpecialistHealthReport + DecisionLogRecord.

设计（43 文档 §4.1 + §4.4 + §4.6 + spec.md types Requirement）:
- EvalContext: L2 调 L3 evaluate 时传的上下文（candidate/baseline/task_sample/eval_method_hint/profile/metadata）
- SpecialistHealthReport: specialist 健康报告（pattern 复用 skill SkillHealthReport，字段不同：L3 有 avg_cost_usd/avg_latency_seconds）
- DecisionLogRecord: 跨 run decision log 记录（log_id/specialist_name/task_id/goal/success_criteria/failure_category/success_criteria_met/lesson_text/timestamp）
- Trend: Literal 4 值（improving/stable/degrading/insufficient_data，复用 skill Trend pattern）
- 复用 L2 自建 EvalTask / EvalResult（import from evolution/promotion_gate.py，不 import skill）
- EvolutionArtifact 复用 L2 Protocol（from evolution/types.py）
- FailureCategory 复用 L2 enum（from evolution/types.py，4 类 GOAL_UNCLEAR/CONTEXT_INSUFFICIENT/ABILITY_INSUFFICIENT/SANDBOX_ISSUE）
- frozen dataclass 不可变（Agent4ML 值对象原则）
- metadata 用 field(default_factory=dict) 避免 mutable default 共享
"""


from dataclasses import dataclass, field
from typing import Any, Literal

from agent4ml.backend.agents.multiagent.evolution.promotion_gate import EvalTask
from agent4ml.backend.agents.multiagent.evolution.types import EvolutionArtifact, FailureCategory

Trend = Literal["improving", "stable", "degrading", "insufficient_data"]


@dataclass(frozen=True)
class EvalContext:
    """L2 调 L3 evaluate 时传的上下文（L3 自建，L2 无此类型）.

    candidate/baseline: L2 演化产物（EvolutionArtifact Protocol）.
    task_sample: L2 抽样的历史 specialist 调用记录（EvalTask tuple）.
    eval_method_hint: L2 建议的评估方法（非强制，Bridge 可覆盖）.
    profile: specialist profile 名（如 "codex"/"claude"）.
    metadata: 扩展字段（如 task_type="open_ended" 触发 llm_judge）.
    """

    candidate: EvolutionArtifact
    baseline: EvolutionArtifact
    task_sample: tuple[EvalTask, ...]
    eval_method_hint: str | None = None
    profile: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SpecialistHealthReport:
    """specialist 健康报告（pattern 复用 skill SkillHealthReport，字段不同）.

    L3 自建（不实现 skill RuntimeTracker Protocol——返回类型 SkillHealthReport 字段不兼容）.
    SpecialistRuntimeTracker.health_report() 产出，L2 cron 周期读 metrics 算趋势.

    trend: improving/stable/degrading/insufficient_data（前后两半比较 completion_rate delta）.
    window_invoked: 窗口内调用次数（< 4 时 trend=insufficient_data）.
    """

    specialist_name: str
    window_invoked: int
    completion_rate: float
    avg_cost_usd: float
    avg_latency_seconds: float
    fallback_rate: float
    trend: Trend
    advice: str = ""


@dataclass(frozen=True)
class DecisionLogRecord:
    """跨 run decision log 记录（43 文档 §4.6 + spec.md DecisionLog Requirement）.

    L1 tool handler 调 specialist 后异步写（fire-and-forget，不阻塞 L1 turn）.
    L2 EvolutionMutator 演化时读最近 N 条 lesson 作为输入样本（不进 prompt，类似 failure cases）.
    保留 90 天 + 超期归档到 specialist_decision_log_archive 表（不删除）.

    failure_category: L2 FailureCategory enum（4 类），None 表示成功调用无失败分类.
    success_criteria_met: 0/1/None（None 表示未评估，对应 SQLite INTEGER nullable）.
    """

    log_id: str
    specialist_name: str
    task_id: str
    goal: str
    success_criteria: str
    failure_category: FailureCategory | None = None
    success_criteria_met: int | None = None
    lesson_text: str | None = None
    timestamp: str = ""
