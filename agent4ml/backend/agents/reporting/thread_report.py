"""渠道无关的报告生成服务 — 从 thread 累积 state 合成报告 + 保存 artifact。

CLI/API/IM 均可调用 generate_report_from_thread，各自负责 presentation（console/HTTP/IM）。
报告生成逻辑（graph.get_state + reporter 合成 + artifact 保存）集中在 reporting 层，
不耦合具体交互渠道。

依赖方向：agents 层不 import app 层，故用 Protocol 描述 runtime 形状（AppRuntime 结构性满足）。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


@dataclass
class ReportArtifact:
    """报告生成结果。渠道无关。"""

    final_report: str
    artifact_path: str | None


class _ReportRuntime(Protocol):
    """generate_report_from_thread 需要的 runtime 形状。AppRuntime 结构性满足。"""

    leader_agent: Any  # 含 .graph
    thread_id: str
    capability_registry: Any  # 含 get_reporter / get_artifact_store
    thread_dir: Path
    config: Any  # 含 .reporting.save_artifact


def generate_report_from_thread(
    runtime: _ReportRuntime,
    topic: str | None = None,
) -> ReportArtifact:
    """从 thread 累积 state 合成报告 + 保存 artifact。

    1. graph.get_state({"configurable":{"thread_id":runtime.thread_id}}) 取 checkpointer 累积 state
    2. topic 非空 → 覆盖 state["research_question"]
    3. reporter.generate_report(state) 合成（三级 fallback：final_report → observations/sources → last AIMessage）
    4. save_artifact → artifact_store.save_artifact(...)
    """
    config = {"configurable": {"thread_id": runtime.thread_id}}
    snapshot = runtime.leader_agent.graph.get_state(config)
    state: dict[str, Any] = (
        dict(snapshot.values) if snapshot and snapshot.values else {}
    )
    if topic:
        state["research_question"] = topic

    reporter = runtime.capability_registry.get_reporter()
    result = reporter.generate_report(state, run_context=None)

    artifact_path: str | None = None
    if runtime.config.reporting.save_artifact:
        artifact = runtime.capability_registry.get_artifact_store().save_artifact(
            content=result.final_report,
            output_dir=runtime.thread_dir,
            title=topic or "Report",
            filename="report.md",
            metadata={"mode": "default", "topic": topic or ""},
        )
        artifact_path = artifact.path
    return ReportArtifact(final_report=result.final_report, artifact_path=artifact_path)
