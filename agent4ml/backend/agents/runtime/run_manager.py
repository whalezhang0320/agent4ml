from __future__ import annotations

import json
import random
import string
from dataclasses import replace
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

from agent4ml.backend.agents.config.schema import AppConfig
from agent4ml.backend.agents.journal.events import utc_now_iso
from agent4ml.backend.agents.journal.run_journal import RunJournal
from agent4ml.backend.agents.runtime.run_context import RunContext
from agent4ml.backend.agents.runtime.run_record import RunRecord, RunStatus

_CST = timezone(timedelta(hours=8))


def _make_run_id() -> str:
    ts = datetime.now(_CST).strftime("%Y%m%dT%H%M%S")
    suffix = "".join(random.choices(string.ascii_lowercase + string.digits, k=4))
    return f"run-{ts}-{suffix}"


class RunManager:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self._records: dict[str, RunRecord] = {}
        self._contexts: dict[str, RunContext] = {}

    def create_run(
        self,
        thread_id: str,
        user_id: str | None = None,
        run_id: str | None = None,
        session_id: str | None = None,
        trace_id: str | None = None,
        model_name: str | None = None,
        thread_dir: Path | None = None,
    ) -> RunContext:
        created_run_id = run_id or _make_run_id()
        if thread_dir is not None:
            output_dir = thread_dir / "runs" / created_run_id
        else:
            output_dir = Path(self.config.runtime.logs_root) / created_run_id
        journal = RunJournal(
            run_id=created_run_id,
            events_path=output_dir / "events.jsonl",
        )
        context = RunContext(
            run_id=created_run_id,
            thread_id=thread_id,
            user_id=user_id,
            session_id=session_id,
            trace_id=trace_id,
            config=self.config,
            budget={},
            output_dir=output_dir,
            enabled_middlewares=self.config.middleware.enabled,
            journal=journal,
        )
        now = utc_now_iso()
        record = RunRecord(
            run_id=created_run_id,
            thread_id=thread_id,
            user_id=user_id,
            status=RunStatus.PENDING,
            created_at=now,
            updated_at=now,
            model_name=model_name or self.config.models.researcher_model,
            metadata={"expert_mode": self.config.runtime.expert_mode},
        )
        self._contexts[created_run_id] = context
        self._records[created_run_id] = record
        self._write_record(record)
        return context

    def mark_running(self, run_id: str) -> RunRecord:
        record = self._require_record(run_id)
        now = utc_now_iso()
        updated = replace(
            record,
            status=RunStatus.RUNNING,
            started_at=record.started_at or now,
            updated_at=now,
        )
        self._store_record(updated)
        self._require_context(run_id).journal.append(
            "run.started",
            {"expert_mode": self.config.runtime.expert_mode},
        )
        return updated

    def mark_success(
        self,
        run_id: str,
        usage_summary: dict[str, Any] | None = None,
    ) -> RunRecord:
        record = self._require_record(run_id)
        now = utc_now_iso()
        usage = usage_summary or {}
        updated = replace(
            record,
            status=RunStatus.SUCCESS,
            updated_at=now,
            finished_at=now,
            total_tokens=usage.get("total_tokens", record.total_tokens),
        )
        self._store_record(updated)
        self._require_context(run_id).journal.append("run.finished", usage)
        return updated

    def mark_failed(self, run_id: str, error: str) -> RunRecord:
        record = self._require_record(run_id)
        now = utc_now_iso()
        updated = replace(
            record,
            status=RunStatus.ERROR,
            updated_at=now,
            finished_at=now,
            error=error,
        )
        self._store_record(updated)
        self._require_context(run_id).journal.append("run.failed", {"error": error})
        return updated

    def get_run(self, run_id: str) -> RunRecord | None:
        return self._records.get(run_id)

    def _store_record(self, record: RunRecord) -> None:
        self._records[record.run_id] = record
        self._write_record(record)

    def _write_record(self, record: RunRecord) -> None:
        context = self._require_context(record.run_id)
        context.output_dir.mkdir(parents=True, exist_ok=True)
        context.record_path.write_text(
            json.dumps(record.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _require_record(self, run_id: str) -> RunRecord:
        try:
            return self._records[run_id]
        except KeyError as exc:
            raise KeyError(f"Unknown run_id: {run_id}") from exc

    def _require_context(self, run_id: str) -> RunContext:
        try:
            return self._contexts[run_id]
        except KeyError as exc:
            raise KeyError(f"Unknown run_id: {run_id}") from exc
