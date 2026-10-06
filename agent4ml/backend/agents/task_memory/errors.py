from __future__ import annotations


class TaskMemoryError(RuntimeError):
    """Base class for task-memory failures."""


class TaskMemoryIntegrityError(TaskMemoryError):
    """Persisted evidence, WAL, or graph failed an integrity check."""


class TaskMemoryVersionConflict(TaskMemoryError):
    """A graph patch was based on a stale materialized version."""


class TaskMemoryValidationError(TaskMemoryError):
    """A path, graph, event, or patch violated the task-memory contract."""
