"""Deterministic workflow control plane for node execution and recovery."""
from __future__ import annotations

import uuid
from dataclasses import dataclass

from agent4ml.backend.agents.ml_research.failure_eval import (
    FailureClass,
    RetryDecision,
    retry_decision,
)
from agent4ml.backend.agents.ml_research.task_models import (
    ApprovalRecord,
    ApprovalStatus,
    NodeStatus,
    QueueMessage,
    TaskRecord,
    TaskStatus,
    utc_now_iso,
)
from agent4ml.backend.agents.ml_research.task_store import TaskStore
from agent4ml.backend.agents.ml_research.workflow import NodeResult, get_workflow


@dataclass(frozen=True)
class MessageDecision:
    is_stale: bool
    reason: str | None = None


class WorkflowEngine:
    def __init__(self, store: TaskStore) -> None:
        self.store = store

    def validate_message(
        self, task: TaskRecord | None, message: QueueMessage
    ) -> MessageDecision:
        if task is None:
            return MessageDecision(True, "task does not exist")
        if not task.is_workflow or message.node_id is None:
            return MessageDecision(True, "message is not a workflow node message")
        if task.status is not TaskStatus.QUEUED:
            return MessageDecision(True, f"task is {task.status.value}")
        if task.current_node != message.node_id:
            return MessageDecision(True, "message node is no longer current")
        node = task.nodes.get(message.node_id)
        if node is None or node.status is not NodeStatus.QUEUED:
            return MessageDecision(True, "node is no longer queued")
        if message.expected_version != task.version:
            return MessageDecision(True, "message task version is stale")
        if message.attempt != node.attempt:
            return MessageDecision(True, "message node attempt is stale")
        return MessageDecision(False)

    def apply_result(self, task: TaskRecord, result: NodeResult) -> TaskRecord:
        if not task.is_workflow or task.current_node is None:
            raise ValueError("task is not an active workflow")
        definition = get_workflow(task.workflow_type or "", task.workflow_version or 0)
        node_id = task.current_node
        if result.outcome == "succeeded":
            next_definition = definition.next_node(node_id)
            approval = None
            if next_definition is not None and next_definition.requires_approval:
                limits = task.inputs.get("resource_limits", {})
                approval = ApprovalRecord(
                    approval_id=f"approval-{uuid.uuid4().hex}",
                    action="start_training",
                    reason="正式训练需要人工确认资源占用和实验配置",
                    payload={
                        "node_id": next_definition.node_id,
                        "resource_limits": limits,
                        "training_command": task.inputs.get("training_command", []),
                    },
                    status=ApprovalStatus.PENDING,
                    requested_at=utc_now_iso(),
                )
            return self.store.commit_node_result(
                task.task_id,
                node_id,
                result,
                next_definition.node_id if next_definition else None,
                task.version,
                approval,
            )
        if result.outcome == "waiting_external":
            return self.store.mark_node_waiting_external(
                task.task_id,
                node_id,
                result,
                expected_version=task.version,
            )
        if result.outcome != "failed":
            raise ValueError(f"unsupported node outcome: {result.outcome}")
        node_definition = definition.node(node_id)
        decision = retry_decision(
            result.failure_class or FailureClass.UNKNOWN,
            node_idempotent=node_definition.handler != "run_training",
        )
        node = task.nodes[node_id]
        if decision is RetryDecision.RETRY_AUTOMATICALLY and node.attempt < node.max_attempts:
            return self.store.schedule_retry(
                task.task_id,
                node_id,
                result,
                expected_version=task.version,
            )
        if decision is RetryDecision.REQUIRE_APPROVAL:
            approval = ApprovalRecord(
                approval_id=f"approval-{uuid.uuid4().hex}",
                action="retry_node",
                reason=(
                    "故障修复或重复执行可能改变实验语义，需要人工确认"
                ),
                payload={
                    "node_id": node_id,
                    "attempt": node.attempt,
                    "failure_class": result.failure_class,
                    "error": result.error,
                },
                status=ApprovalStatus.PENDING,
                requested_at=utc_now_iso(),
            )
            return self.store.request_node_approval(
                task.task_id,
                node_id,
                approval,
                expected_version=task.version,
                result=result,
            )
        return self.store.fail_node(
            task.task_id,
            node_id,
            result,
            expected_version=task.version,
        )

    def recover_expired_claim(self, task: TaskRecord) -> TaskRecord:
        """Recover a claimed node without blindly replaying uncertain side effects."""
        if task.current_node is None:
            return task
        node = task.nodes[task.current_node]
        if node.external_job_id:
            result = NodeResult.waiting_external(node.external_job_id)
            if node.status is NodeStatus.RUNNING:
                return self.store.mark_node_waiting_external(
                    task.task_id,
                    node.node_id,
                    result,
                    expected_version=task.version,
                )
            return task
        failure = NodeResult.failed(
            FailureClass.WORKER_LOST.value,
            "worker claim expired before node completion",
        )
        definition = get_workflow(task.workflow_type or "", task.workflow_version or 0)
        if definition.node(node.node_id).handler == "run_training":
            approval = ApprovalRecord(
                approval_id=f"approval-{uuid.uuid4().hex}",
                action="retry_node",
                reason="训练提交状态不确定，重新提交前需要人工确认",
                payload={"node_id": node.node_id, "attempt": node.attempt},
                status=ApprovalStatus.PENDING,
                requested_at=utc_now_iso(),
            )
            return self.store.request_node_approval(
                task.task_id,
                node.node_id,
                approval,
                expected_version=task.version,
                result=failure,
            )
        return self.store.schedule_retry(
            task.task_id,
            node.node_id,
            failure,
            expected_version=task.version,
        )
