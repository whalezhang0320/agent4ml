"""EvolutionMutator 单测 — LLM 成功 + JSON parse 失败重试 + schema 不匹配重试 + 字段非法重试 + 重试 2 仍失败保持旧 + 多字段自由演化 + mock LLM 返不同 candidate.

设计（spec.md EvolutionMutator Requirement + 42 文档 §7.7 + R2）:
- 单次 LLM + 结构化 JSON 输出 + rationale 字段（R2.1）
- 重试 2 次（含首次共 3 次），失败保持旧 is_active（INV-14/INV-15）
- 多字段自由演化（R2.4）
- lead 同 model（R2.5，INV-18）
- LLM 调用失败/JSON parse/schema/字段非法 → 重试 1 次（R2.2）
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import pytest

from agent4ml.backend.agents.multiagent.evolution.evolution_mutator import (
    EvolutionMutator,
    EvolutionResult,
    LLMCaller,
)
from agent4ml.backend.agents.multiagent.evolution.types import (
    ContextSummaryTemplate,
    FailureCategory,
    FailureStats,
    SkillInjectionTemplate,
)


class _FakeLLMCaller:
    """测试用 LLMCaller（可配置返回值队列）."""

    def __init__(self, outputs: list[str]) -> None:
        self._outputs = list(outputs)
        self._call_count = 0
        self.calls: list[str] = []

    def call(self, prompt: str) -> str:
        self.calls.append(prompt)
        idx = min(self._call_count, len(self._outputs) - 1)
        self._call_count += 1
        return self._outputs[idx]


def _make_stats(
    dominant: FailureCategory = FailureCategory.CONTEXT_INSUFFICIENT,
    by_category: dict | None = None,
) -> FailureStats:
    if by_category is None:
        by_category = {dominant: 5} if dominant else {}
    return FailureStats(
        by_category=by_category,
        dominant_category=dominant,
        sample_failures={dominant: [
            __import__("agent4ml.backend.agents.multiagent.evolution.types", fromlist=["FailureRecord"]).FailureRecord(
                specialist_name="codex", goal="g", success_criteria="sc",
                failure_category=dominant, severity=0.9,
            )
        ]} if dominant else {},
    )


def _make_template(version="v1", skeleton="s1") -> ContextSummaryTemplate:
    return ContextSummaryTemplate(
        version=version, template_id="default",
        extractors=(), filters=(), max_tokens=2000, prompt_skeleton=skeleton,
    )


def _make_skill_template(version="v1") -> SkillInjectionTemplate:
    return SkillInjectionTemplate(
        version=version, template_id="default",
        skill_selector=_DummySelector(), injection_format="fmt",
    )


class _DummySelector:
    def select(self, goal, available_skills):
        return ()


def _ctx_json(skeleton="s2", max_tokens=3000, extractors=None) -> str:
    """构造合法 LLM JSON 输出（context_summary）."""
    return json.dumps({
        "version": "v2",
        "template_id": "default",
        "rationale": "add error logs",
        "extractors": extractors or ["LastNMessages", "ErrorLogsFromMessages"],
        "filters": ["TruncateAtTokens"],
        "max_tokens": max_tokens,
        "prompt_skeleton": skeleton,
    })


def _skill_json(selector="KeywordMatch", fmt="Skills: {skills}", max_skills=5) -> str:
    """构造合法 LLM JSON 输出（skill_injection）."""
    return json.dumps({
        "version": "v2",
        "template_id": "default",
        "rationale": "upgrade to LLMClassify",
        "skill_selector": selector,
        "injection_format": fmt,
        "max_skills": max_skills,
    })


# ── 单次 LLM 成功 ─────────────────────────────────────────────────────────────


def test_evolve_context_summary_success():
    """LLM 返合法 JSON → 解析为 ContextSummaryTemplate，含 rationale."""
    caller = _FakeLLMCaller([_ctx_json()])
    mutator = EvolutionMutator(llm_caller=caller)
    current = _make_template()
    stats = _make_stats()

    result = mutator.evolve_context_summary(current, stats)
    assert result.success is True
    assert result.candidate is not None
    assert result.candidate.version == "v2"
    assert result.retries == 0


def test_evolve_skill_injection_success():
    """LLM 返合法 JSON → 解析为 SkillInjectionTemplate."""
    caller = _FakeLLMCaller([_skill_json()])
    mutator = EvolutionMutator(llm_caller=caller)
    current = _make_skill_template()
    stats = _make_stats(dominant=FailureCategory.ABILITY_INSUFFICIENT)

    result = mutator.evolve_skill_injection(current, stats)
    assert result.success is True
    assert result.candidate is not None
    assert result.candidate.version == "v2"
    assert result.candidate.max_skills == 5


# ── JSON parse 失败重试 ───────────────────────────────────────────────────────


def test_json_parse_fail_retry_then_success():
    """LLM 返非法 JSON → 重试 → 成功."""
    caller = _FakeLLMCaller(["not json", _ctx_json()])
    mutator = EvolutionMutator(llm_caller=caller, max_retries=2)
    current = _make_template()
    stats = _make_stats()

    result = mutator.evolve_context_summary(current, stats)
    assert result.success is True
    assert result.candidate is not None
    assert result.retries == 1  # 重试 1 次后成功


def test_json_parse_fail_all_retries_keep_old():
    """重试 2 次仍失败 → 保持旧（返 failure_type=json_parse）."""
    caller = _FakeLLMCaller(["not json", "still not json", "nope"])
    mutator = EvolutionMutator(llm_caller=caller, max_retries=2)
    current = _make_template()
    stats = _make_stats()

    result = mutator.evolve_context_summary(current, stats)
    assert result.success is False
    assert result.candidate is None
    assert result.failure_type == "json_parse"
    assert result.retries == 2


# ── schema 不匹配重试 ─────────────────────────────────────────────────────────


def test_schema_mismatch_retry_then_success():
    """LLM 返 JSON 但缺字段 → 重试 → 成功."""
    missing_field = json.dumps({"version": "v2"})  # 缺 template_id 等
    caller = _FakeLLMCaller([missing_field, _ctx_json()])
    mutator = EvolutionMutator(llm_caller=caller, max_retries=2)
    current = _make_template()
    stats = _make_stats()

    result = mutator.evolve_context_summary(current, stats)
    assert result.success is True
    assert result.retries == 1


# ── 字段非法重试 ───────────────────────────────────────────────────────────────


def test_illegal_field_retry():
    """LLM 返非法 extractor 名 → 重试."""
    illegal = _ctx_json(extractors=["NonExistentExtractor"])
    caller = _FakeLLMCaller([illegal, _ctx_json()])
    mutator = EvolutionMutator(llm_caller=caller, max_retries=2)
    current = _make_template()
    stats = _make_stats()

    result = mutator.evolve_context_summary(current, stats)
    assert result.success is True
    assert result.retries == 1


def test_illegal_skill_selector_retry():
    """LLM 返非法 skill_selector 名 → 重试."""
    illegal = _skill_json(selector="NonExistentSelector")
    caller = _FakeLLMCaller([illegal, _skill_json()])
    mutator = EvolutionMutator(llm_caller=caller, max_retries=2)
    current = _make_skill_template()
    stats = _make_stats(dominant=FailureCategory.ABILITY_INSUFFICIENT)

    result = mutator.evolve_skill_injection(current, stats)
    assert result.success is True
    assert result.retries == 1


# ── LLM 调用异常重试 ─────────────────────────────────────────────────────────


def test_llm_call_exception_retry():
    """LLM call 抛异常 → 重试."""
    class _ExceptionCaller:
        def __init__(self):
            self.count = 0
            self.calls = []

        def call(self, prompt):
            self.calls.append(prompt)
            self.count += 1
            if self.count == 1:
                raise ConnectionError("network timeout")
            return _ctx_json()

    caller = _ExceptionCaller()
    mutator = EvolutionMutator(llm_caller=caller, max_retries=2)
    current = _make_template()
    stats = _make_stats()

    result = mutator.evolve_context_summary(current, stats)
    assert result.success is True
    assert result.retries == 1


# ── 重试 2 次仍失败保持旧 ───────────────────────────────────────────────────


def test_max_retries_exhausted_failure_type():
    """重试 2 次仍失败 → failure_type 记录最后一次错误类型."""
    caller = _FakeLLMCaller(["bad1", "bad2", "bad3"])
    mutator = EvolutionMutator(llm_caller=caller, max_retries=2)
    current = _make_template()
    stats = _make_stats()

    result = mutator.evolve_context_summary(current, stats)
    assert result.success is False
    assert result.failure_type == "json_parse"
    assert result.retries == 2
    assert "failed after 3" in result.rationale


# ── 多字段自由演化 ───────────────────────────────────────────────────────────


def test_multi_field_evolution():
    """LLM 可同时改 extractors / filters / max_tokens / prompt_skeleton（R2.4）."""
    custom_json = json.dumps({
        "version": "v2",
        "template_id": "default",
        "rationale": "multi-field",
        "extractors": ["LastNMessages", "CodeSnippetsFromMessages", "ErrorLogsFromMessages"],
        "filters": ["TruncateAtTokens", "DeduplicateByHash"],
        "max_tokens": 4000,
        "prompt_skeleton": "Custom skeleton with {messages} {code} {errors}",
    })
    caller = _FakeLLMCaller([custom_json])
    mutator = EvolutionMutator(llm_caller=caller)
    current = _make_template()
    stats = _make_stats()

    result = mutator.evolve_context_summary(current, stats)
    assert result.success is True
    assert result.candidate is not None
    # 反序列化为空 tuple（L1 hot swap 重新构造），但 max_tokens + skeleton 应保留
    assert result.candidate.max_tokens == 4000
    assert "Custom skeleton" in result.candidate.prompt_skeleton


# ── lead 同 model（R2.5，INV-18）────────────────────────────────────────────


def test_evolution_model_none_inherits_lead():
    """evolution_model=None → 继承 lead（R2.5，INV-18）."""
    caller = _FakeLLMCaller([_ctx_json()])
    mutator = EvolutionMutator(llm_caller=caller, evolution_model=None)
    assert mutator._evolution_model is None


def test_evolution_model_custom_config():
    """evolution_model 可配置覆盖."""
    caller = _FakeLLMCaller([_ctx_json()])
    mutator = EvolutionMutator(llm_caller=caller, evolution_model="gpt-4")
    assert mutator._evolution_model == "gpt-4"


# ── 无 LLM caller ───────────────────────────────────────────────────────────


def test_no_llm_caller_returns_failure():
    """无 LLM caller → 返 failure（llm_timeout）."""
    mutator = EvolutionMutator(llm_caller=None)
    current = _make_template()
    stats = _make_stats()

    result = mutator.evolve_context_summary(current, stats)
    assert result.success is False
    assert result.failure_type == "llm_timeout"
    assert result.candidate is None


# ── prompt 含 error_hint 重试 ───────────────────────────────────────────────


def test_retry_prompt_includes_error_hint():
    """重试时 prompt 含 error_hint（R2.2）."""
    caller = _FakeLLMCaller(["not json", _ctx_json()])
    mutator = EvolutionMutator(llm_caller=caller, max_retries=2)
    current = _make_template()
    stats = _make_stats()

    result = mutator.evolve_context_summary(current, stats)
    assert result.success is True
    # 第一次 prompt 无 error_hint，第二次有
    assert len(caller.calls) == 2
    assert "Strict JSON" in caller.calls[1]


def test_retry_prompt_includes_schema_error():
    """schema 不匹配重试时 prompt 含 schema 错误信息."""
    missing_field = json.dumps({"version": "v2"})
    caller = _FakeLLMCaller([missing_field, _ctx_json()])
    mutator = EvolutionMutator(llm_caller=caller, max_retries=2)
    current = _make_template()
    stats = _make_stats()

    result = mutator.evolve_context_summary(current, stats)
    assert result.success is True
    assert "Schema error" in caller.calls[1]


# ── markdown fence 清理 ───────────────────────────────────────────────────────


def test_markdown_fence_cleaned():
    """LLM 输出含 ```json fence → 清理后解析."""
    fenced = "```json\n" + _ctx_json() + "\n```"
    caller = _FakeLLMCaller([fenced])
    mutator = EvolutionMutator(llm_caller=caller)
    current = _make_template()
    stats = _make_stats()

    result = mutator.evolve_context_summary(current, stats)
    assert result.success is True
    assert result.candidate is not None
