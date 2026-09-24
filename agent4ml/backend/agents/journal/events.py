from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone, timedelta
from typing import Any

_CST = timezone(timedelta(hours=8))


@dataclass(frozen=True)
class RunEvent:
    event_id: str
    run_id: str
    event_type: str
    payload: dict[str, Any]
    created_at: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def utc_now_iso() -> str:
    return datetime.now(_CST).isoformat()
