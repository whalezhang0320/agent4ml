"""IntentEngineStrengthened — ContextSummarizer 输入源（非 middleware，R6.5）。

设计（42 文档 §7.10 + spec.md IntentEngineStrengthened Requirement + R6）:
- 作为 ContextSummarizer 输入源（不作为 before_model middleware，不注入 system prompt，R6.5）
- 分层：IntentTreeStrategy（薄包装既有 ReportIntentStrategy.match）优先 + LLMIntentStrategy（MVP 仅接口，数据触发后实现）兜底
- IntentTreeStrategy 不调 detect_and_dispatch（CLI 路由入口返 bool 会触发 report action 副作用）
- get_candidate_metadata 生成 SpecialistCandidate（不含 capability_match，LLM 自决）
- intent_llm_enabled 默认 false（数据触发后启用：委派率 < 20% 或 ABILITY_INSUFFICIENT > 50%）
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from agent4ml.backend.agents.intent.engine import ReportIntentStrategy
from agent4ml.backend.agents.multiagent.evolution.metrics_view import MetricsView
from agent4ml.backend.agents.multiagent.evolution.types import SpecialistCandidate


@dataclass(frozen=True)
class IntentAnalysis:
    """意图识别结果（IntentTreeStrategy / LLMIntentStrategy 返回）.

    confidence >= 0.7 视为命中，< 0.7 视为 miss.
    candidates 为空列表（LLM 兜底启用后才生成 specialist 候选）.
    """

    confidence: float
    candidates: list[str]
    intent_type: str = "unknown"


class IntentStrategy(Protocol):
    """意图识别策略 Protocol（IntentTreeStrategy / LLMIntentStrategy 实现）."""

    def analyze(
        self,
        user_message: str,
        available_specialists: list[str],
    ) -> IntentAnalysis: ...


class IntentTreeStrategy:
    """薄包装既有 ReportIntentStrategy.match（非 detect_and_dispatch）.

    既有 IntentTree 是单层 root→Report leaf，无 specialist 分类能力.
    IntentTreeStrategy 主要起"confidence 判定 + 零 LLM 成本"作用.
    LLM 兜底才是 specialist 候选生成主力（数据触发后才启用）.

    不调 detect_and_dispatch（CLI 路由入口返 bool 会触发 report action 副作用）.
    """

    def __init__(self, confidence_threshold: float = 0.7) -> None:
        self._strategy = ReportIntentStrategy()
        self._threshold = confidence_threshold

    def analyze(
        self,
        user_message: str,
        available_specialists: list[str],
    ) -> IntentAnalysis:
        """调 ReportIntentStrategy.match（纯匹配返 MatchResult），适配为 IntentAnalysis.

        confidence>=0.7 视为命中，< 0.7 视为 miss.
        candidates 始终空（IntentTree 无 specialist 分类能力）.
        """
        result = self._strategy.match(user_message)
        if result.matched and result.confidence >= self._threshold:
            # payload["type"] 是 IntentType enum，取 .value 转 str
            raw_type = result.payload.get("type")
            intent_type = raw_type.value if hasattr(raw_type, "value") else str(raw_type)
            return IntentAnalysis(
                confidence=result.confidence,
                candidates=[],
                intent_type=intent_type,
            )
        return IntentAnalysis(confidence=0.0, candidates=[], intent_type="unknown")


class LLMIntentStrategy:
    """LLM 兜底（MVP 仅接口签名，不实现）.

    数据触发后才实现（委派率 < 20% 或 ABILITY_INSUFFICIENT > 50%）.
    MVP 阶段 intent_llm_enabled=false，LLMIntentStrategy 不实例化.
    """

    def analyze(
        self,
        user_message: str,
        available_specialists: list[str],
    ) -> IntentAnalysis:
        """MVP 不实现——抛 NotImplementedError 或返 IntentTree 结果.

        数据触发后才实现（R6.3 启用阈值）.
        """
        raise NotImplementedError(
            "LLMIntentStrategy not implemented yet - enable when delegate_rate < 20% "
            "or ABILITY_INSUFFICIENT > 50%"
        )


class IntentEngineStrengthened:
    """分层意图识别（IntentTree 优先 + LLM 兜底）+ candidate metadata 生成.

    R6.5 架构修正：作为 ContextSummarizer 输入源，不作为 before_model middleware.
    candidate metadata 通过 ContextSummarizer 渲染进 context_summary（per-call 产物）.

    MVP: intent_llm_enabled=false → llm=None → analyze 始终返 tree 结果.
    """

    def __init__(
        self,
        tree: IntentTreeStrategy,
        llm: LLMIntentStrategy | None = None,
    ) -> None:
        self._tree = tree
        self._llm = llm  # None = LLM 兜底未启用

    def analyze(
        self,
        user_message: str,
        available_specialists: list[str],
    ) -> IntentAnalysis:
        """分层：IntentTree 优先 + confidence>=0.7 或 llm=None 时返 tree + 否则调 llm 兜底 + 失败 fallback tree."""
        result = self._tree.analyze(user_message, available_specialists)
        if result.confidence >= 0.7 or self._llm is None:
            return result
        # IntentTree miss + LLM 兜底启用 → 调 LLM
        try:
            return self._llm.analyze(user_message, available_specialists)
        except NotImplementedError:
            # LLM 未实现 → fallback tree（不让意图识别失败阻塞 turn）
            return result
        except Exception:
            # LLM 调用失败 → fallback tree
            return result

    def get_candidate_metadata(
        self,
        goal: str,
        available_specialists: list[str],
        metrics_view: MetricsView,
    ) -> list[SpecialistCandidate]:
        """生成 candidate metadata（不含 capability_match，LLM 自决，INV-35）.

        供 ContextSummarizer 渲染进 context_summary（per-call 产物）.
        数据来源：metrics_view.get_specialist_metrics 查询历史成功率/成本/延迟.
        """
        candidates: list[SpecialistCandidate] = []
        for name in available_specialists:
            snap = metrics_view.get_specialist_metrics(name)
            if snap is None:
                # 无记录的 specialist 用默认值（sample_size=0，LLM 可判断可信度）
                candidates.append(SpecialistCandidate(
                    name=name,
                    historical_success_rate=0.0,
                    avg_cost_usd=0.0,
                    avg_latency_seconds=0.0,
                    sample_size=0,
                ))
                continue
            candidates.append(SpecialistCandidate(
                name=name,
                historical_success_rate=snap["completion_rate"],
                avg_cost_usd=snap["avg_cost_usd"],
                avg_latency_seconds=snap["avg_latency_seconds"],
                sample_size=snap["sample_size"],
            ))
        return candidates
