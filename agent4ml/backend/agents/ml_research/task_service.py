"""Application service for submitting, observing, and cancelling ML tasks."""
from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any, Sequence

from agent4ml.backend.agents.ml_research.task_models import (
    TERMINAL_TASK_STATUSES,
    TaskRecord,
    TaskStatus,
    utc_now_iso,
)
from agent4ml.backend.agents.ml_research.task_store import TaskNotFoundError, TaskStore


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
