"""EvolutionMutator — LLM 演化执行（R2）。

设计（42 文档 §7.7 + spec.md EvolutionMutator Requirement + R2）:
- evolve_context_summary / evolve_skill_injection 调 LLM 单次 + 结构化 JSON 输出 + rationale 字段
- 最多重试 2 次（含首次共 3 次），失败保持旧 is_active + 写 metrics（INV-14）
- 多字段自由演化（不限制单字段，R2.4）
- 输入样本按 failure_category 聚类取 top（每类 2 个，上限 5，INV-17，由 FailureFocuser 生成）
- evolution_model 默认 None（继承 lead，R2.5，INV-18）
- LLM 调用失败/JSON parse 失败/schema 不匹配/字段非法 → 重试 1 次
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Protocol

from agent4ml.backend.agents.multiagent.evolution.types import (
    ContextSummaryTemplate,
    EvolutionArtifact,
    FailureStats,
    SkillInjectionTemplate,
)

logger = logging.getLogger(__name__)

# 演化失败类型（写 metrics 用）
_EVOLUTION_FAILURE_TYPES = (
    "llm_timeout",
    "json_parse",
    "schema_mismatch",
    "illegal_field",
)


class LLMCaller(Protocol):
    """LLM 调用抽象（R2.5：默认 None 继承 lead，可注入测试 mock）.

    call(prompt) → str：返 LLM 原始文本输出（期望 JSON）。
    """

    def call(self, prompt: str) -> str: ...


@dataclass
class EvolutionResult:
    """单次演化结果（成功/失败 + candidate artifact + rationale + 失败类型）."""

    success: bool
    candidate: EvolutionArtifact | None = None
    rationale: str = ""
    failure_type: str = ""  # _EVOLUTION_FAILURE_TYPES 之一
    retries: int = 0
    llm_model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0


class EvolutionMutator:
    """LLM 演化执行（R2）.

    INVARIANT:
    - 单次 LLM 调用 + 结构化 JSON 输出 + rationale 字段（INV-14，R2.1）
    - 最多重试 2 次（含首次共 3 次），失败保持旧 is_active（INV-14/INV-15）
    - 多字段自由演化（R2.4）
    - 输入样本按 failure_category 聚类取 top（INV-17，FailureFocuser 生成）
    - evolution_model 默认 None 继承 lead（INV-18，R2.5）
    - LLM 调用失败/JSON parse/schema/字段非法 → 重试 1 次（R2.2）
    """

    def __init__(
        self,
        llm_caller: LLMCaller | None = None,
        evolution_model: str | None = None,
        max_retries: int = 2,
    ) -> None:
        self._llm_caller = llm_caller
        self._evolution_model = evolution_model
        self._max_retries = max_retries

    def evolve_context_summary(
        self,
        current: ContextSummaryTemplate,
        failures: FailureStats,
        lessons: tuple[str, ...] = (),
    ) -> EvolutionResult:
        """演化 ContextSummaryTemplate（CONTEXT_INSUFFICIENT 占主导时调）."""
        return self._evolve(
            current, failures, "context_summary",
            _CONTEXT_SUMMARY_SCHEMA, lessons,
        )

    def evolve_skill_injection(
        self,
        current: SkillInjectionTemplate,
        failures: FailureStats,
        lessons: tuple[str, ...] = (),
    ) -> EvolutionResult:
        """演化 SkillInjectionTemplate（ABILITY_INSUFFICIENT 占主导时调）."""
        return self._evolve(
            current, failures, "skill_injection",
            _SKILL_INJECTION_SCHEMA, lessons,
        )

    def _evolve(
        self,
        current: EvolutionArtifact,
        failures: FailureStats,
        artifact_type: str,
        schema: dict,
        lessons: tuple[str, ...] = (),
    ) -> EvolutionResult:
        """单次 LLM + 结构化 JSON + 重试 2（R2.1/R2.2）."""
        if self._llm_caller is None:
            return EvolutionResult(
                success=False,
                candidate=None,
                failure_type="llm_timeout",
                rationale="no LLM caller configured",
            )

        prompt = self._build_prompt(current, failures, artifact_type, schema, lessons=lessons)
        last_error = ""
        for attempt in range(self._max_retries + 1):  # 首次 + max_retries
            try:
                raw_output = self._llm_caller.call(prompt)
                candidate = self._parse_and_validate(
                    raw_output, current, artifact_type, schema,
                )
                return EvolutionResult(
                    success=True,
                    candidate=candidate,
                    rationale=candidate.__class__.__name__ + "_evolved",
                    retries=attempt,
                )
            except json.JSONDecodeError:
                last_error = "json_parse"
                prompt = self._build_prompt(
                    current, failures, artifact_type, schema,
                    error_hint="Strict JSON required. Output ONLY JSON, no markdown.",
                    lessons=lessons,
                )
                logger.warning("EvolutionMutator JSON parse failed, retry %d", attempt + 1)
            except _SchemaMismatchError as e:
                last_error = "schema_mismatch"
                prompt = self._build_prompt(
                    current, failures, artifact_type, schema,
                    error_hint=f"Schema error: {e}. Fix and retry.",
                    lessons=lessons,
                )
                logger.warning("EvolutionMutator schema mismatch, retry %d", attempt + 1)
            except _IllegalFieldError as e:
                last_error = "illegal_field"
                prompt = self._build_prompt(
                    current, failures, artifact_type, schema,
                    error_hint=f"Illegal field: {e}. Valid extractors/filters listed in schema.",
                    lessons=lessons,
                )
                logger.warning("EvolutionMutator illegal field, retry %d", attempt + 1)
            except Exception as e:
                last_error = "llm_timeout"
                prompt = self._build_prompt(
                    current, failures, artifact_type, schema,
                    error_hint=f"LLM call failed: {e}. Retry.",
                    lessons=lessons,
                )
                logger.warning("EvolutionMutator LLM call failed: %s, retry %d", e, attempt + 1)

        return EvolutionResult(
            success=False,
            candidate=None,
            failure_type=last_error,
            retries=self._max_retries,
            rationale=f"evolution failed after {self._max_retries + 1} attempts",
        )

    def _build_prompt(
        self,
        current: EvolutionArtifact,
        failures: FailureStats,
        artifact_type: str,
        schema: dict,
        *,
        error_hint: str = "",
        lessons: tuple[str, ...] = (),
    ) -> str:
        """构造 LLM prompt（42 文档 §7.7 骨架 + error_hint 重试 + L3 lessons）."""
        current_json = _serialize_current(current, artifact_type)
        failure_cases = _serialize_failure_samples(failures)
        lessons_section = _serialize_lessons(lessons) if lessons else ""

        prompt = f"""You are a template evolution specialist. Analyze failure patterns and propose
an improved template. Output strictly as JSON matching the schema.

Current active template ({current.version}):
{current_json}

Failure statistics (last 24h):
{_serialize_failure_categories(failures)}

Representative failure cases (top samples):
{failure_cases}
{lessons_section}
Available extractors: {schema.get('available_extractors', [])}
Available filters: {schema.get('available_filters', [])}
Available skill_selectors: {schema.get('available_selectors', [])}

Task: Propose v{int(current.version[1:]) + 1} template that addresses the dominant failure pattern.
Only modify fields relevant to the failure pattern. Preserve unrelated fields.

Output JSON schema:
{json.dumps(schema.get('output_schema', {}), indent=2)}

{error_hint}
"""
        return prompt

    def _parse_and_validate(
        self,
        raw_output: str,
        current: EvolutionArtifact,
        artifact_type: str,
        schema: dict,
    ) -> EvolutionArtifact:
        """解析 LLM JSON 输出 + schema 校验 + 构造 candidate."""
        # 清理 markdown fence
        cleaned = raw_output.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("```")[1]
            if cleaned.startswith("json"):
                cleaned = cleaned[4:]
        data = json.loads(cleaned)

        # schema 校验：必填字段
        required = schema.get("output_schema", {}).get("required", [])
        for field_name in required:
            if field_name not in data:
                raise _SchemaMismatchError(f"missing field: {field_name}")

        # 构造 candidate
        next_version = f"v{int(current.version[1:]) + 1}"
        template_id = current.template_id

        if artifact_type == "context_summary":
            extractors = _validate_extractors(
                data.get("extractors", []),
                schema.get("available_extractors", []),
            )
            filters = _validate_filters(
                data.get("filters", []),
                schema.get("available_filters", []),
            )
            return ContextSummaryTemplate(
                version=next_version,
                template_id=template_id,
                extractors=extractors,
                filters=filters,
                max_tokens=data.get("max_tokens", 2000),
                prompt_skeleton=data.get("prompt_skeleton", ""),
            )
        if artifact_type == "skill_injection":
            selector_name = data.get("skill_selector", "KeywordMatch")
            valid_selectors = schema.get("available_selectors", [])
            if selector_name not in valid_selectors:
                raise _IllegalFieldError(
                    f"skill_selector '{selector_name}' not in {valid_selectors}"
                )
            return SkillInjectionTemplate(
                version=next_version,
                template_id=template_id,
                skill_selector=_DummySelector(),
                injection_format=data.get("injection_format", ""),
                max_skills=data.get("max_skills", 3),
            )
        raise _SchemaMismatchError(f"unknown artifact_type: {artifact_type}")


class _SchemaMismatchError(Exception):
    """LLM 输出 schema 不匹配（重试用）."""


class _IllegalFieldError(Exception):
    """LLM 输出字段值非法（重试用）."""


class _DummySelector:
    """演化产物 selector placeholder（L1 hot swap 时重新构造）."""

    def select(self, goal: str, available_skills: list) -> tuple:
        return ()


def _serialize_current(current: EvolutionArtifact, artifact_type: str) -> str:
    """序列化当前 active 模板为 JSON."""
    if isinstance(current, ContextSummaryTemplate):
        return json.dumps({
            "version": current.version,
            "template_id": current.template_id,
            "extractors": [type(e).__name__ for e in current.extractors],
            "filters": [type(f).__name__ for f in current.filters],
            "max_tokens": current.max_tokens,
            "prompt_skeleton": current.prompt_skeleton,
        }, sort_keys=True)
    if isinstance(current, SkillInjectionTemplate):
        return json.dumps({
            "version": current.version,
            "template_id": current.template_id,
            "skill_selector": type(current.skill_selector).__name__,
            "injection_format": current.injection_format,
            "max_skills": current.max_skills,
        }, sort_keys=True)
    return json.dumps({"version": current.version, "template_id": current.template_id})


def _serialize_failure_categories(failures: FailureStats) -> str:
    """序列化 failure_category 统计."""
    lines = []
    for cat, count in failures.by_category.items():
        lines.append(f"- {cat.value}: {count}")
    if failures.dominant_category:
        lines.append(f"dominant: {failures.dominant_category.value}")
    return "\n".join(lines)


def _serialize_failure_samples(failures: FailureStats) -> str:
    """序列化失败样本（每类 top 2，上限 5）."""
    samples = []
    for cat, records in failures.sample_failures.items():
        for r in records:
            samples.append({
                "category": cat.value,
                "specialist": r.specialist_name,
                "goal": r.goal,
                "severity": r.severity,
            })
    return json.dumps(samples, indent=2)


def _serialize_lessons(lessons: tuple[str, ...]) -> str:
    """序列化 L3 DecisionLog lessons（L2 演化时作输入样本，L3-7.4 决策 b）."""
    if not lessons:
        return ""
    lines = ["Historical lessons from past runs (L3 DecisionLog):"]
    for i, lesson in enumerate(lessons, 1):
        lines.append(f"  {i}. {lesson}")
    return "\n".join(lines) + "\n"


def _validate_extractors(
    names: list[str], available: list[str]
) -> tuple:
    """校验 extractor 名是否在 available 列表（非法抛 _IllegalFieldError）."""
    valid = [n for n in names if n in available]
    if len(valid) != len(names):
        invalid = [n for n in names if n not in available]
        raise _IllegalFieldError(f"extractors {invalid} not in {available}")
    return ()  # 反序列化为空 tuple（L1 hot swap 重新构造）


def _validate_filters(
    names: list[str], available: list[str]
) -> tuple:
    """校验 filter 名是否在 available 列表."""
    valid = [n for n in names if n in available]
    if len(valid) != len(names):
        invalid = [n for n in names if n not in available]
        raise _IllegalFieldError(f"filters {invalid} not in {available}")
    return ()


# LLM prompt schema（42 文档 §7.7）
_CONTEXT_SUMMARY_SCHEMA = {
    "available_extractors": [
        "LastNMessages", "CodeSnippetsFromMessages", "ErrorLogsFromMessages",
        "RelatedFilesFromSandbox", "TestResultsFromSandbox",
    ],
    "available_filters": [
        "TruncateAtTokens", "DeduplicateByHash", "PrioritizeRecent",
    ],
    "output_schema": {
        "type": "object",
        "required": ["version", "template_id", "extractors", "filters", "max_tokens", "prompt_skeleton"],
        "properties": {
            "version": {"type": "string"},
            "template_id": {"type": "string"},
            "rationale": {"type": "string"},
            "extractors": {"type": "array", "items": {"type": "string"}},
            "filters": {"type": "array", "items": {"type": "string"}},
            "max_tokens": {"type": "integer"},
            "prompt_skeleton": {"type": "string"},
        },
    },
}

_SKILL_INJECTION_SCHEMA = {
    "available_selectors": [
        "KeywordMatch", "LLMClassify", "SemanticMatch",
    ],
    "output_schema": {
        "type": "object",
        "required": ["version", "template_id", "skill_selector", "injection_format", "max_skills"],
        "properties": {
            "version": {"type": "string"},
            "template_id": {"type": "string"},
            "rationale": {"type": "string"},
            "skill_selector": {"type": "string"},
            "injection_format": {"type": "string"},
            "max_skills": {"type": "integer"},
        },
    },
}
