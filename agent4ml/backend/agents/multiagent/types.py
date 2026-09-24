"""Multi-Agent 核心数据契约 — frozen dataclass。

SpecialistRequest/Result/RawResult + SubagentRequest/Result + ArtifactRef + TokenUsage + SpecialistCapabilities。

设计（proposal §3 + design.md §8）:
- specialist 黑盒：Agent4ML 只传 goal + context_summary + sandbox_id（INV#1）
- per-specialist 转换器：ContextSummarizer 生成 context_summary，ResultSummarizer 生成 SpecialistResult
- success_criteria 强制：tool handler 必填，ResultSummarizer 校验（INV#10）
- artifacts 分离：specialist 产物写 ThreadState.orchestration.specialist_artifacts，不混 lead agent artifacts
- token 用量 self-report：specialist runtime 返 TokenUsage，可选（specialist 黑盒，可能不暴露）
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class SpecialistCapability(str, Enum):
    """Specialist 能力枚举（list_specialists 按能力过滤）。"""

    CODING = "coding"
    RESEARCH = "research"
    REVIEW = "review"
    PLANNING = "planning"


@dataclass(frozen=True)
class SpecialistCapabilities:
    """Specialist 能力声明（register 时填，list_specialists 过滤用）。"""

    capabilities: tuple[SpecialistCapability, ...] = ()

    def has(self, capability: SpecialistCapability) -> bool:
        return capability in self.capabilities


@dataclass(frozen=True)
class TokenUsage:
    """specialist 调用 token 用量（specialist self-report，可选）。"""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass(frozen=True)
class ArtifactRef:
    """specialist 产物引用（写入 ThreadState.orchestration.specialist_artifacts）。

    与 state.Artifact 区分：ArtifactRef 是 specialist 产物的轻量引用，
    不带 artifact_id（specialist 黑盒，Agent4ML 不分配全局 ID）。
    """

    path: str
    artifact_type: str
    specialist_name: str
    description: str = ""
    created_at: str | None = None


@dataclass(frozen=True)
class SpecialistRequest:
    """specialist 调用请求。

    tool handler 内部构造（LLM 只填 goal + success_criteria + sandbox_id 可选，
    其余从 config + ContextSummarizer 自动生成，INV#10 programmatic eval floor）。
    """

    goal: str
    success_criteria: str
    context_summary: str
    sandbox_id: str | None
    artifacts_path: str | None
    max_steps: int = 50
    timeout_seconds: int = 600
    allowed_tools: tuple[str, ...] = ()
    skill_injection: str | None = None


@dataclass(frozen=True)
class SpecialistRawResult:
    """specialist runtime 返回的 raw output（specialist 黑盒输出）。

    ResultSummarizer 消费此结构，压缩为 SpecialistResult 回传 lead agent。
    """

    raw_output: str
    artifacts: tuple[ArtifactRef, ...] = ()
    usage: TokenUsage | None = None
    duration_seconds: float = 0.0
    exit_code: int = 0


@dataclass(frozen=True)
class SpecialistResult:
    """specialist 调用结果（ResultSummarizer 生成，回传 lead agent）。

    success=False 时 gap_analysis 必填（programmatic eval floor，INV#10）。
    error 字段仅失败时填（错误类型 + 简述，不暴露 specialist 内部状态）。
    L2 扩展：failure_category（L2 FailureFocuser 读取，D-7=c，default None 向后兼容）。
    """

    specialist_name: str
    summary: str
    artifacts: tuple[ArtifactRef, ...] = ()
    success: bool = False
    gap_analysis: str = ""
    usage: TokenUsage | None = None
    duration_seconds: float = 0.0
    error: str | None = None
    failure_category: str | None = None


@dataclass(frozen=True)
class SubagentRequest:
    """Agent4ML self-copy subagent 调用请求（leaf role，复用 lead factory）。

    与 SpecialistRequest 同构（SubagentResult 与 SpecialistResult 同构，
    便于 OrchestrationMiddleware 统一处理）。
    """

    goal: str
    success_criteria: str
    context_summary: str
    sandbox_id: str | None
    artifacts_path: str | None
    max_steps: int = 20
    timeout_seconds: int = 300
    allowed_tools: tuple[str, ...] = ()
    skill_injection: str | None = None


@dataclass(frozen=True)
class SubagentResult:
    """Agent4ML self-copy subagent 返回结果（与 SpecialistResult 同构）。"""

    summary: str
    artifacts: tuple[ArtifactRef, ...] = ()
    success: bool = False
    gap_analysis: str = ""
    usage: TokenUsage | None = None
    duration_seconds: float = 0.0
    error: str | None = None
