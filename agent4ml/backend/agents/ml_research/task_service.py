"""Application service for submitting, observing, and cancelling ML tasks."""
from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any, Sequence

from agent4ml.backend.agents.ml_research.task_models import (
    NodeRecord,
    NodeStatus,
    TERMINAL_TASK_STATUSES,
    TaskRecord,
    TaskStatus,
    utc_now_iso,
)
from agent4ml.backend.agents.ml_research.task_store import TaskNotFoundError, TaskStore
from agent4ml.backend.agents.ml_research.workflow import REPRODUCE_WORKFLOW


class TaskValidationError(ValueError):
    pass


def _contained_path(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


class MLTaskService:
    def __init__(self, store: TaskStore, *, work_root: Path, allowed_root: Path) -> None:
        self.store = store
        self.work_root = work_root.expanduser().resolve()
        self.allowed_root = allowed_root.expanduser().resolve()

    def submit(
        self,
        *,
        name: str,
        command: Sequence[str],
        cwd: str | Path,
        metadata: dict[str, Any] | None = None,
    ) -> TaskRecord:
        normalized_command = tuple(str(item) for item in command)
        if not normalized_command or any(not item for item in normalized_command):
            raise TaskValidationError("command must contain at least one non-empty argument")
        if len(normalized_command) > 128 or any(len(item) > 4096 for item in normalized_command):
            raise TaskValidationError("command exceeds the local service safety limit")
        resolved_cwd = Path(cwd).expanduser().resolve()
        if not resolved_cwd.is_dir():
            raise TaskValidationError(f"cwd is not a directory: {resolved_cwd}")
        if not _contained_path(resolved_cwd, self.allowed_root):
            raise TaskValidationError(
                f"cwd must be under AGENT4ML_ML_ALLOWED_ROOT: {self.allowed_root}"
            )

        task_id = f"task-{uuid.uuid4().hex}"
        now = utc_now_iso()
        run_dir = self.work_root / task_id
        task = TaskRecord(
            task_id=task_id,
            name=name.strip() or task_id,
            command=normalized_command,
            cwd=str(resolved_cwd),
            run_dir=str(run_dir),
            status=TaskStatus.QUEUED,
            created_at=now,
            updated_at=now,
            metadata=dict(metadata or {}),
        )
        self.store.submit(task)
        return task

    def submit_reproduction(
        self,
        *,
        name: str,
        repository_path: str | Path,
        paper_path: str | Path | None = None,
        target_metric: str | None = None,
        resource_limits: dict[str, Any] | None = None,
        training_command: Sequence[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> TaskRecord:
        repository = Path(repository_path).expanduser().resolve()
        if not repository.is_dir():
            raise TaskValidationError(f"repository_path is not a directory: {repository}")
        if not _contained_path(repository, self.allowed_root):
            raise TaskValidationError(
                f"repository_path must be under AGENT4ML_ML_ALLOWED_ROOT: {self.allowed_root}"
            )
        resolved_paper: Path | None = None
        if paper_path is not None:
            resolved_paper = Path(paper_path).expanduser().resolve()
            if not resolved_paper.is_file():
                raise TaskValidationError(f"paper_path is not a file: {resolved_paper}")
            if not _contained_path(resolved_paper, self.allowed_root):
                raise TaskValidationError(
                    f"paper_path must be under AGENT4ML_ML_ALLOWED_ROOT: {self.allowed_root}"
                )
        limits = dict(resource_limits or {})
        gpu_count = limits.get("gpu_count", 0)
        if not isinstance(gpu_count, int) or isinstance(gpu_count, bool) or gpu_count < 0:
            raise TaskValidationError("resource_limits.gpu_count must be a non-negative integer")
        normalized_command = tuple(str(item) for item in (training_command or ()))
        if any(not item for item in normalized_command):
            raise TaskValidationError("training_command cannot contain empty arguments")
        if len(normalized_command) > 128 or any(len(item) > 4096 for item in normalized_command):
            raise TaskValidationError("training_command exceeds the local service safety limit")

        task_id = f"task-{uuid.uuid4().hex}"
        now = utc_now_iso()
        run_dir = self.work_root / task_id
        nodes = {
            definition.node_id: NodeRecord(
                node_id=definition.node_id,
                status=NodeStatus.QUEUED
                if index == 0
                else NodeStatus.PENDING,
                attempt=1 if index == 0 else 0,
            )
            for index, definition in enumerate(REPRODUCE_WORKFLOW.nodes)
        }
        inputs: dict[str, Any] = {
            "repository_path": str(repository),
            "target_metric": target_metric,
            "resource_limits": limits,
            "training_command": list(normalized_command),
        }
        if resolved_paper is not None:
            inputs["paper_path"] = str(resolved_paper)
        first_node = REPRODUCE_WORKFLOW.nodes[0].node_id
        task = TaskRecord(
            task_id=task_id,
            name=name.strip() or task_id,
            command=(),
            cwd=str(repository),
            run_dir=str(run_dir),
            status=TaskStatus.QUEUED,
            created_at=now,
            updated_at=now,
            metadata=dict(metadata or {}),
            workflow_type=REPRODUCE_WORKFLOW.workflow_type,
            workflow_version=REPRODUCE_WORKFLOW.version,
            current_node=first_node,
            nodes=nodes,
            inputs=inputs,
        )
        self.store.submit(task)
        return task

    def get(self, task_id: str) -> TaskRecord:
        task = self.store.get(task_id)
        if task is None:
            raise TaskNotFoundError(task_id)
        return task

    def cancel(self, task_id: str) -> TaskRecord:
        before = self.get(task_id)
        if before.status in TERMINAL_TASK_STATUSES:
            return before
        updated = self.store.request_cancel(task_id)
        event_type = (
            "task.cancelled"
            if updated.status is TaskStatus.CANCELLED
            else "task.cancellation_requested"
        )
        self.store.publish(task_id, event_type, {"status": updated.status.value})
        return updated

    def approve(
        self,
        task_id: str,
        *,
        approval_id: str,
        expected_version: int,
        resolved_by: str | None = None,
    ) -> TaskRecord:
        self.get(task_id)
        return self.store.resolve_approval(
            task_id,
            approval_id,
            approved=True,
            expected_version=expected_version,
            resolved_by=resolved_by,
        )

    def reject(
        self,
        task_id: str,
        *,
        approval_id: str,
        expected_version: int,
        resolved_by: str | None = None,
    ) -> TaskRecord:
        self.get(task_id)
        return self.store.resolve_approval(
            task_id,
            approval_id,
            approved=False,
            expected_version=expected_version,
            resolved_by=resolved_by,
        )

    def retry(self, task_id: str, *, expected_version: int) -> TaskRecord:
        task = self.get(task_id)
        if not task.is_workflow:
            raise TaskValidationError("manual node retry is only available for workflow tasks")
        return self.store.retry_failed_node(task_id, expected_version=expected_version)
