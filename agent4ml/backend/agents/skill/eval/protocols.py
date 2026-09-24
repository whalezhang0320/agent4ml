"""Skill eval 评估层 Protocol 抽象 — 三层 eval + 趋势层 + 持久化契约。

5 Protocol 定义 L3 评估层的契约：
- SkillJudgmentAnalyzer：执行层 eval（post-execution LLM 分析，产 SkillJudgment + EvolutionSuggestion）
- TaskQualityJudge：任务层 eval（post-execution LLM 4 维评分）
- ResponseContractChecker：响应层 eval（pre-promotion，编译规则检查 candidate）
- RuntimeTracker：趋势层 eval（从历史数据产健康报告 + 退化检测）
- EvalRunStore：eval 结果持久化

L3 impl = 实现 Protocol + 注入 bootstrap，不动 L2 核心（零侵入）。
"""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from agent4ml.backend.agents.skill.evolution.types import EvalResult
from agent4ml.backend.agents.skill.eval.types import (
    EvolutionSuggestion,
    SkillHealthReport,
    SkillJudgment,
    TaskQualityScore,
)


@runtime_checkable
class SkillJudgmentAnalyzer(Protocol):
    """执行层 eval：每次任务后 LLM 判断 per-skill 是否被应用。

    D-L3-21: 同时产出 EvolutionSuggestion（FIX/DERIVED/CAPTURED + direction）。
    D-L3-3: 更新 L1 4 计数器（analyzer 内调 store.record_outcome）。
    D-L3-19: 异步 fire-and-forget，不阻塞用户。
    """

    async def analyze_execution(
        self,
        task_id: str,
        journal_events: list[dict],
        messages_summary: str,
        injected_skills: list[dict],
    ) -> tuple[list[SkillJudgment], list[EvolutionSuggestion]]: ...


@runtime_checkable
class TaskQualityJudge(Protocol):
    """任务层 eval：LLM 4 维加权评分（D-L3-13 权重 0.50/0.35/0.05/0.10）。"""

    async def judge_task(
        self,
        task_id: str,
        execution_trace: str,
        final_output: str,
    ) -> TaskQualityScore: ...


@runtime_checkable
class ResponseContractChecker(Protocol):
    """响应层 eval：从 skill 文本编译规则，检查 candidate SKILL.md。

    D-L3-4: ContractCompiler 从 skill 文本自动编译（contract-aware），外部 skill 零改造。
    走 EvalBridge Protocol 给 L2 ScoreDeltaGate。
    """

    def check(
        self,
        candidate_content: str,
        baseline_content: str,
    ) -> EvalResult: ...


@runtime_checkable
class RuntimeTracker(Protocol):
    """趋势层 eval：从历史数据产出健康报告 + 退化检测。

    D-L3-16: after_agent 写数据 + 命令时算趋势 + GitRatchet 按需调 degraded_skills。
    """

    def health_report(
        self, skill_id: str, window: int = 20,
    ) -> SkillHealthReport: ...

    def degraded_skills(
        self, threshold: float = 0.15,
    ) -> list[str]: ...


@runtime_checkable
class EvalRunStore(Protocol):
    """eval 结果持久化。复用 skills.db v2→v3（D-L3-10）。"""

    def save_judgment(self, judgment: SkillJudgment) -> str: ...
    def save_task_score(self, score: TaskQualityScore) -> str: ...
    def save_eval_run(self, run: Any) -> str: ...
    def get_judgments(self, skill_id: str, limit: int = 20) -> list[SkillJudgment]: ...
    def get_task_scores(self, task_id: str) -> TaskQualityScore | None: ...
