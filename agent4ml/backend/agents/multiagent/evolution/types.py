"""L2 演化产物 + 类型层 — frozen dataclass + Protocol。

设计（42 文档 §2.3 + §7.1 + spec.md EvolutionArtifact Requirement）:
- EvolutionArtifact: @runtime_checkable Protocol（version / template_id / artifact_hash 属性）
- ContextSummaryTemplate: W2 演化产物（extractors / filters / max_tokens / prompt_skeleton）
- SkillInjectionTemplate: W4 演化产物（skill_selector / injection_format / max_skills=3）
- artifact_hash 由 payload 计算得出（@property，防环用，INV-7/INV-27）
- 演化产物形态 = 结构化 dataclass（D-F=b，可 version DAG / diff / 回滚，INV-9）
- SpecialistCandidate 不含 capability_match（R6.2 修正，LLM 自决，INV-35）
- FailureCategory 枚举 4 类（GOAL_UNCLEAR / CONTEXT_INSUFFICIENT / ABILITY_INSUFFICIENT / SANDBOX_ISSUE）
- 所有 dataclass frozen 不可变
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class EvolutionArtifact(Protocol):
    """演化产物 Protocol（runtime_checkable）。

    不含字段实现，仅声明属性接口。子类用 frozen dataclass 各自定义字段。
    artifact_hash 由 payload 计算得出（防环用，INV-7/INV-27）。
    """

    @property
    def version(self) -> str: ...

    @property
    def template_id(self) -> str: ...

    @property
    def artifact_hash(self) -> str: ...


class ContextExtractor(Protocol):
    """从 ThreadState 提取 context 片段（per-specialist 实现）。

    extractors 顺序执行，输出拼接后进 filters。
    """

    def extract(self, state: dict, goal: str) -> str: ...


class ContextFilter(Protocol):
    """过滤 / 截断 context（per-specialist 实现）。

    filters 顺序执行，前一个输出作后一个输入。
    """

    def filter(self, content: str, max_tokens: int) -> str: ...


class SkillSelector(Protocol):
    """根据 goal 选 skill（per-specialist 实现）。

    select 返回 tuple[Skill, ...]，按 max_skills 截断。
    """

    def select(self, goal: str, available_skills: list[Any]) -> tuple[Any, ...]: ...


@dataclass(frozen=True)
class ContextSummaryTemplate:
    """W2: ContextSummarizer 用的模板，控制 specialist 调用前 context 如何生成。

    演化产物：L2 EvolutionMutator 演化此模板（加 extractor / 调 max_tokens / 改 prompt 骨架）。
    不进 system prompt cache prefix（per-call 产物，hot swap 不破 cache，INV-6）。
    artifact_hash 由 payload（extractors + filters + max_tokens + prompt_skeleton）计算。
    """

    version: str
    template_id: str
    extractors: tuple[ContextExtractor, ...]
    filters: tuple[ContextFilter, ...]
    max_tokens: int
    prompt_skeleton: str

    @property
    def artifact_hash(self) -> str:
        payload = json.dumps({
            "version": self.version,
            "template_id": self.template_id,
            "extractors": [type(e).__name__ for e in self.extractors],
            "filters": [type(f).__name__ for f in self.filters],
            "max_tokens": self.max_tokens,
            "prompt_skeleton": self.prompt_skeleton,
        }, sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class SkillInjectionTemplate:
    """W4: SpecialistRequest.skill_injection 生成模板。

    演化产物：L2 EvolutionMutator 演化此模板（换 selector / 加 max_skills / 改注入格式）。
    不进 system prompt cache prefix（per-call 产物，hot swap 不破 cache，INV-6）。
    artifact_hash 由 payload（skill_selector + injection_format + max_skills）计算。
    """

    version: str
    template_id: str
    skill_selector: SkillSelector
    injection_format: str
    max_skills: int = 3

    @property
    def artifact_hash(self) -> str:
        payload = json.dumps({
            "version": self.version,
            "template_id": self.template_id,
            "skill_selector": type(self.skill_selector).__name__,
            "injection_format": self.injection_format,
            "max_skills": self.max_skills,
        }, sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


class FailureCategory(Enum):
    """L1 ResultSummarizer 输出的失败分类（L2 FailureFocuser 读取，D-7=c）。

    CONTEXT_INSUFFICIENT / ABILITY_INSUFFICIENT 可演化（→ W2 / W4）。
    GOAL_UNCLEAR / SANDBOX_ISSUE 不演化（转告警，不进 L2 流程）。
    """

    GOAL_UNCLEAR = "goal_unclear"
    CONTEXT_INSUFFICIENT = "context_insufficient"
    ABILITY_INSUFFICIENT = "ability_insufficient"
    SANDBOX_ISSUE = "sandbox_issue"


@dataclass(frozen=True)
class FailureRecord:
    """单条失败记录（L2 FailureFocuser 聚类取 top 样本用）。

    severity 用于聚类排序（越大越优先取）。
    """

    specialist_name: str
    goal: str
    success_criteria: str
    failure_category: FailureCategory
    raw_output_tail: str = ""
    severity: float = 0.0
    timestamp: str = ""


@dataclass(frozen=True)
class FailureStats:
    """失败聚焦统计（FailureFocuser.analyze 输出，喂给 EvolutionMutator）。

    dominant_category: 占比最高的可演化类别（GOAL_UNCLEAR / SANDBOX_ISSUE 不作主导）。
    sample_failures: 每类 top 2 样本（上限 5，INV-17）。
    """

    by_category: dict[FailureCategory, int]
    dominant_category: FailureCategory | None
    sample_failures: dict[FailureCategory, list[FailureRecord]]


@dataclass(frozen=True)
class SpecialistCandidate:
    """specialist 候选 metadata（IntentEngineStrengthened 生成，供 ContextSummarizer 渲染）。

    R6.2 修正：不含 capability_match（LLM 自决，INV-35）。
    historical_success_rate: 过去 N 次（N=20）success_criteria_met=true 比例。
    sample_size < 20 时 LLM 可判断可信度。
    """

    name: str
    historical_success_rate: float
    avg_cost_usd: float
    avg_latency_seconds: float
    sample_size: int


@dataclass(frozen=True)
class CostRecord:
    """单次 specialist 调用成本（BudgetGuard 记账用，R5.2 三维度）。

    cost_usd 由 token × model price 计算（MVP 用 config 默认 price）。
    """

    tokens: int = 0
    cost_usd: float = 0.0
    calls: int = 1


@dataclass(frozen=True)
class BudgetRemaining:
    """budget 剩余量（BudgetCheckResult.remaining 用）。

    三维度：tokens / cost_usd / calls，per-day UTC 0 点重置（R5.3）。
    """

    tokens: int = 0
    cost_usd: float = 0.0
    calls: int = 0


@dataclass(frozen=True)
class BudgetCheckResult:
    """BudgetGuard.check_and_record 返回（R5.4 超限 fallback lead）。

    allowed=False 时 reason 非空（"daily_cost_exceeded" / "daily_tokens_exceeded" / "daily_calls_exceeded"）。
    fallback_target 固定 "lead"（不 fallback 另一 specialist，INV-10）。
    """

    allowed: bool
    specialist_name: str
    reason: str | None
    remaining: BudgetRemaining | None
    fallback_target: str = "lead"


class TriggerSource(Enum):
    """L2 触发源（TriggerManager 四源 + 节流，D-1）。

    PERIODIC: 6h cron 兜底（R4.1）。
    FAILURE_FOCUSED: 24h 窗口内某 failure_category ≥ 5 次（R4.4a）。
    SPECIALIST_DEGRADED: invoked ≥ 5 + completion_rate < 0.4（R4.4b）。
    COST_ALERT: 单次 cost > $1（R4.4c）。
    LATENCY_ALERT: 单次 latency > 5min（R4.4c）。
    """

    PERIODIC = "periodic"
    FAILURE_FOCUSED = "failure_focused"
    SPECIALIST_DEGRADED = "specialist_degraded"
    COST_ALERT = "cost_alert"
    LATENCY_ALERT = "latency_alert"


class PromotionDecision(Enum):
    """PromotionGate 决策（D-4 hash 防环 + 95% CI）。

    ACCEPT: candidate CI 下界 > baseline CI 上界（INV-24）。
    REJECT: CI 重叠 / hash 命中近 5 版（防环，INV-7）。
    FAILED: 演化或 eval 失败（保持旧 is_active，INV-13）。
    """

    ACCEPT = "accept"
    REJECT = "reject"
    FAILED = "failed"


@dataclass(frozen=True)
class EvolutionTask:
    """L2 演化任务（L2TriggerMiddleware enqueue → cron queue → L2EvolutionWorker 消费）。

    profile: 演化 profile（per-profile 串行锁 key，INV-5）。
    trigger_source + trigger_detail: 触发源 + 详情（写 OrchestrationMetricsL2）。
    """

    task_id: str
    profile: str
    trigger_source: TriggerSource
    trigger_detail: str = ""
    artifact_type: str = "context_summary"
    timestamp: str = ""


@dataclass(frozen=True)
class EvolutionResult:
    """L2EvolutionWorker.run 返回（编排闭环结果）。

    decision=ACCEPT 时 new_artifact_id 非空（VersionDAG commit 后的 id）。
    decision=REJECT/FAILED 时 new_artifact_id=None（保持旧 is_active，INV-13）。
    """

    task_id: str
    decision: PromotionDecision
    new_artifact_id: str | None = None
    rationale: str = ""
    error: str | None = None
