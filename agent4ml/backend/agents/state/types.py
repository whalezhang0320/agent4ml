from __future__ import annotations

from dataclasses import dataclass, field
from typing import Annotated, Any, NotRequired, TypedDict

from langchain.agents import AgentState

from agent4ml.backend.agents.state.reducers import (
    merge_artifacts,
    merge_citations,
    merge_errors,
    merge_final_report,
    merge_governance,
    merge_memory_recalled,
    merge_memory_updates,
    merge_metadata,
    merge_observations,
    merge_orchestration,
    merge_reflection_items,
    merge_sandbox,
    merge_sources,
    merge_tagged_context,
    merge_todos,
)

# 治理层共享状态：策略 bundle 自管命名空间 governance.<strategy_name>.*。
# 族无关，不预设固定槽（原 6 固定槽是 volume 族 schema，已删）。
# merge_governance deep-merge（last-write-wins per leaf key）。
# 全值可 JSON 序列化（str/int/dict/list），禁 file handle 等不可序列化对象。
GovernanceState = dict[str, Any]


class OrchestrationState(TypedDict, total=False):
    """Multi-Agent 编排层状态（specialist 产物 + 活跃 specialist）。

    与 lead agent artifacts 分离——specialist 产物写 specialist_artifacts，
    不混入 ThreadState.artifacts（design.md §2 artifacts 分离）。
    merge_orchestration 去重追加（specialist_artifacts 按 path / active_specialists 按 name）。
    """

    specialist_artifacts: list  # list[ArtifactRef]
    active_specialists: list[str]


@dataclass(frozen=True)
class IntentState:
    task_type: str
    depth: str
    objective: str
    constraints: tuple[str, ...] = field(default_factory=tuple)
    output_format: str = "markdown_report"
    clarification_needed: bool = False


@dataclass(frozen=True)
class PlanStep:
    step_id: str
    title: str
    description: str = ""
    status: str = "pending"


@dataclass(frozen=True)
class ResearchPlan:
    plan_id: str
    goal: str
    steps: tuple[PlanStep, ...] = field(default_factory=tuple)
    status: str = "pending"


@dataclass(frozen=True)
class Observation:
    observation_id: str
    step_id: str | None
    content: str
    source_refs: tuple[str, ...] = field(default_factory=tuple)
    created_at: str | None = None


@dataclass(frozen=True)
class Source:
    source_id: str
    url: str
    title: str = ""
    source_type: str = "web"
    retrieved_at: str | None = None
    summary: str = ""


@dataclass(frozen=True)
class Citation:
    citation_id: str
    source_id: str
    quote: str
    claim: str
    location: str | None = None


@dataclass(frozen=True)
class Artifact:
    artifact_id: str
    artifact_type: str
    title: str
    path: str
    summary: str = ""
    source_refs: tuple[str, ...] = field(default_factory=tuple)
    created_at: str | None = None


@dataclass(frozen=True)
class ReflectionItem:
    item_id: str
    scope: str
    kind: str
    question: str
    status: str = "open"
    related_refs: tuple[str, ...] = field(default_factory=tuple)
    created_at: str | None = None


@dataclass(frozen=True)
class AgentError:
    error_id: str
    stage: str
    message: str
    related_refs: tuple[str, ...] = field(default_factory=tuple)
    created_at: str | None = None
    # F8.1：errors 升级为工具调用账本，扩展字段（带默认值兼容既有构造）
    kind: str = "failure"           # "failure" / "success"
    tool_name: str = ""
    attempt: int = 0                # 该 tool 连续失败次（成功归 0）
    error_type: str = ""            # F5 分类
    reason: str = ""                # F8.2 原因模板


class ThreadState(AgentState):
    """LangGraph state schema for Agent4ML research threads.

    Extends AgentState (inherits messages with add_messages reducer).
    Field-level Annotated reducers drive graph-internal state merge.
    """

    user_input: NotRequired[str]
    intent: NotRequired[Any]
    research_question: NotRequired[str]
    plan: NotRequired[Any]
    current_step_id: NotRequired[str | None]
    observations: Annotated[list, merge_observations]
    sources: Annotated[list, merge_sources]
    citations: Annotated[list, merge_citations]
    artifacts: Annotated[list, merge_artifacts]
    reflection_items: Annotated[list, merge_reflection_items]
    final_report: Annotated[str | None, merge_final_report]
    errors: Annotated[list, merge_errors]
    metadata: Annotated[dict, merge_metadata]
    todos: Annotated[list | None, merge_todos]
    governance: Annotated[GovernanceState | None, merge_governance]
    tagged_context: Annotated[dict | None, merge_tagged_context]
    sandbox: Annotated[dict | None, merge_sandbox]
    orchestration: Annotated[OrchestrationState | None, merge_orchestration]
    help_request_count: NotRequired[int]
    pending_help: NotRequired[dict | None]
    skill_suggestion: NotRequired[list[dict] | None]
    recalled_memories: Annotated[list | None, merge_memory_recalled]
    memory_updates: Annotated[list | None, merge_memory_updates]
