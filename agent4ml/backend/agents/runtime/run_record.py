from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class RunStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    ERROR = "error"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"
    INTERRUPTED = "interrupted"


@dataclass(frozen=True)
class RunRecord:
    run_id: str
    thread_id: str
    user_id: str | None
    status: RunStatus
    created_at: str
    updated_at: str
    started_at: str | None = None
    finished_at: str | None = None
    model_name: str | None = None
    error: str | None = None
    total_tokens: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["status"] = self.status.value
        return payload
