from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agent4ml.backend.agents.config.schema import AppConfig
from agent4ml.backend.agents.journal.run_journal import RunJournal


@dataclass(frozen=True)
class RunContext:
    run_id: str
    thread_id: str
    user_id: str | None
    session_id: str | None
    trace_id: str | None
    config: AppConfig
    budget: dict[str, Any]
    output_dir: Path
    enabled_middlewares: tuple[str, ...]
    journal: RunJournal

    @property
    def record_path(self) -> Path:
        return self.output_dir / "record.json"

    @property
    def events_path(self) -> Path:
        return self.output_dir / "events.jsonl"

    @property
    def artifacts_dir(self) -> Path:
        return self.output_dir / "artifacts"
