"""Versioned workflow definitions and node execution result contracts."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from agent4ml.backend.agents.ml_research.task_models import NodeRecord, TaskRecord


@dataclass(frozen=True)
class WorkflowNodeDefinition:
    node_id: str
    handler: str
    requires_approval: bool = False
    retry_policy: str = "default"


@dataclass(frozen=True)
class WorkflowDefinition:
    workflow_type: str
    version: int
    nodes: tuple[WorkflowNodeDefinition, ...]

    def node(self, node_id: str) -> WorkflowNodeDefinition:
        for node in self.nodes:
            if node.node_id == node_id:
                return node
        raise KeyError(f"unknown workflow node: {node_id}")

    def next_node(self, node_id: str) -> WorkflowNodeDefinition | None:
        for index, node in enumerate(self.nodes):
            if node.node_id == node_id:
                return self.nodes[index + 1] if index + 1 < len(self.nodes) else None
        raise KeyError(f"unknown workflow node: {node_id}")


REPRODUCE_WORKFLOW = WorkflowDefinition(
    workflow_type="paper_reproduction",
    version=1,
    nodes=(
        WorkflowNodeDefinition("analyze_code", "analyze_code"),
        WorkflowNodeDefinition("build_environment", "build_environment"),
        WorkflowNodeDefinition(
            "run_training",
            "run_training",
            requires_approval=True,
            retry_policy="training",
        ),
        WorkflowNodeDefinition("validate_result", "validate_result"),
    ),
)

WORKFLOWS: dict[tuple[str, int], WorkflowDefinition] = {
    (REPRODUCE_WORKFLOW.workflow_type, REPRODUCE_WORKFLOW.version): REPRODUCE_WORKFLOW
}


def get_workflow(workflow_type: str, version: int) -> WorkflowDefinition:
    try:
        return WORKFLOWS[(workflow_type, version)]
    except KeyError as exc:
        raise ValueError(f"unsupported workflow: {workflow_type}@{version}") from exc


@dataclass(frozen=True)
class NodeResult:
    outcome: str
    artifacts: dict[str, str] = field(default_factory=dict)
    external_job_id: str | None = None
    failure_class: str | None = None
    error: str | None = None

    @classmethod
    def succeeded(cls, artifacts: dict[str, str] | None = None) -> NodeResult:
        return cls("succeeded", artifacts=dict(artifacts or {}))

    @classmethod
    def failed(cls, failure_class: str, error: str) -> NodeResult:
        return cls("failed", failure_class=failure_class, error=error)

    @classmethod
    def waiting_external(
        cls, external_job_id: str, artifacts: dict[str, str] | None = None
    ) -> NodeResult:
        return cls(
            "waiting_external",
            artifacts=dict(artifacts or {}),
            external_job_id=external_job_id,
        )


class NodeHandler(Protocol):
    def execute(self, task: TaskRecord, node: NodeRecord) -> NodeResult: ...


class TrainingBackend(Protocol):
    def submit(
        self, *, config_path: str, idempotency_key: str, inputs: dict[str, Any]
    ) -> str: ...

    def get_status(self, external_job_id: str) -> str: ...
