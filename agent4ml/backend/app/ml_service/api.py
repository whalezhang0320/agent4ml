"""FastAPI routes and replayable SSE stream for ML task control."""
from __future__ import annotations

import asyncio
import json
import re
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, Header, HTTPException, Query, Request, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from agent4ml.backend.agents.ml_research.task_models import (
    InvalidTaskTransition,
    StaleTaskVersion,
    TERMINAL_TASK_STATUSES,
    TaskEvent,
    TaskRecord,
)
from agent4ml.backend.agents.ml_research.task_service import (
    MLTaskService,
    TaskValidationError,
)
from agent4ml.backend.agents.ml_research.task_store import (
    RedisTaskStore,
    TaskNotFoundError,
    TaskStore,
)
from agent4ml.backend.app.ml_service.settings import MLServiceSettings


class TaskSubmitRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=200)
    command: list[str] = Field(min_length=1, max_length=128)
    cwd: str = "."
    metadata: dict[str, object] = Field(default_factory=dict)


class ReproductionSubmitRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=200)
    repository_path: str
    paper_path: str | None = None
    target_metric: str | None = None
    resource_limits: dict[str, object] = Field(default_factory=dict)
    training_command: list[str] | None = Field(default=None, max_length=128)
    metadata: dict[str, object] = Field(default_factory=dict)


class ApprovalRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    approval_id: str = Field(min_length=1)
    expected_version: int = Field(ge=1)
    resolved_by: str | None = None


class RetryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    expected_version: int = Field(ge=1)


class TaskResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    task_id: str
    name: str
    command: list[str]
    cwd: str
    run_dir: str
    status: str
    created_at: str
    updated_at: str
    version: int
    started_at: str | None = None
    finished_at: str | None = None
    worker_id: str | None = None
    worker_heartbeat_at: str | None = None
    pid: int | None = None
    exit_code: int | None = None
    failure_class: str | None = None
    error: str | None = None
    metadata: dict[str, object] = Field(default_factory=dict)
    workflow_type: str | None = None
    workflow_version: int | None = None
    current_node: str | None = None
    nodes: dict[str, object] = Field(default_factory=dict)
    inputs: dict[str, object] = Field(default_factory=dict)
    artifacts: dict[str, str] = Field(default_factory=dict)
    approval: dict[str, object] | None = None


def _response(task: TaskRecord) -> TaskResponse:
    return TaskResponse.model_validate(task.to_dict())


def encode_sse(event: TaskEvent) -> str:
    event_type = event.event_type.replace("\r", "").replace("\n", "")
    payload = json.dumps(event.to_dict(), ensure_ascii=False, separators=(",", ":"))
    return f"id: {event.event_id}\nevent: {event_type}\ndata: {payload}\n\n"


def _validate_stream_id(value: str) -> str:
    if not re.fullmatch(r"\d+-\d+", value):
        raise HTTPException(status_code=400, detail="invalid event stream id")
    return value


async def _event_stream(
    request: Request,
    store: TaskStore,
    task_id: str,
    after_id: str,
    heartbeat_seconds: float,
) -> AsyncIterator[str]:
    cursor = after_id
    block_ms = max(100, int(heartbeat_seconds * 1000))
    while True:
        if await request.is_disconnected():
            return
        events = await asyncio.to_thread(
            store.read_events, task_id, cursor, block_ms, 100
        )
        if events:
            for event in events:
                cursor = event.event_id
                yield encode_sse(event)
            continue
        task = await asyncio.to_thread(store.get, task_id)
        if task is None or task.status in TERMINAL_TASK_STATUSES:
            return
        yield ": heartbeat\n\n"


def create_app(
    *,
    store: TaskStore | None = None,
    settings: MLServiceSettings | None = None,
) -> FastAPI:
    settings = settings or MLServiceSettings.from_env()
    owned_store = store is None
    store = store or RedisTaskStore(
        settings.redis_url, namespace=settings.redis_namespace
    )
    service = MLTaskService(
        store, work_root=settings.work_root, allowed_root=settings.allowed_root
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        del app
        yield
        if owned_store:
            await asyncio.to_thread(store.close)

    app = FastAPI(
        title="Agent4ML ML Task Service",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.state.task_store = store
    app.state.task_service = service
    app.state.settings = settings

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/ready")
    async def ready() -> dict[str, str]:
        try:
            healthy = await asyncio.to_thread(store.ping)
        except Exception as exc:
            raise HTTPException(status_code=503, detail="task store unavailable") from exc
        if not healthy:
            raise HTTPException(status_code=503, detail="task store unavailable")
        return {"status": "ready"}

    @app.post(
        "/v1/tasks",
        response_model=TaskResponse,
        status_code=status.HTTP_202_ACCEPTED,
    )
    def submit(payload: TaskSubmitRequest) -> TaskResponse:
        try:
            task = service.submit(
                name=payload.name,
                command=payload.command,
                cwd=payload.cwd,
                metadata=payload.metadata,
            )
        except TaskValidationError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return _response(task)

    @app.post(
        "/v1/reproductions",
        response_model=TaskResponse,
        status_code=status.HTTP_202_ACCEPTED,
    )
    def submit_reproduction(payload: ReproductionSubmitRequest) -> TaskResponse:
        try:
            task = service.submit_reproduction(
                name=payload.name,
                repository_path=payload.repository_path,
                paper_path=payload.paper_path,
                target_metric=payload.target_metric,
                resource_limits=payload.resource_limits,
                training_command=payload.training_command,
                metadata=payload.metadata,
            )
        except TaskValidationError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return _response(task)

    @app.get("/v1/tasks/{task_id}", response_model=TaskResponse)
    def get_task(task_id: str) -> TaskResponse:
        try:
            return _response(service.get(task_id))
        except TaskNotFoundError as exc:
            raise HTTPException(status_code=404, detail="task not found") from exc

    @app.post("/v1/tasks/{task_id}/cancel", response_model=TaskResponse)
    def cancel_task(task_id: str) -> TaskResponse:
        try:
            return _response(service.cancel(task_id))
        except TaskNotFoundError as exc:
            raise HTTPException(status_code=404, detail="task not found") from exc

    def _approval_action(
        task_id: str, payload: ApprovalRequest, *, approved: bool
    ) -> TaskResponse:
        try:
            operation = service.approve if approved else service.reject
            return _response(
                operation(
                    task_id,
                    approval_id=payload.approval_id,
                    expected_version=payload.expected_version,
                    resolved_by=payload.resolved_by,
                )
            )
        except TaskNotFoundError as exc:
            raise HTTPException(status_code=404, detail="task not found") from exc
        except (StaleTaskVersion, InvalidTaskTransition) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/v1/tasks/{task_id}/approve", response_model=TaskResponse)
    def approve_task(task_id: str, payload: ApprovalRequest) -> TaskResponse:
        return _approval_action(task_id, payload, approved=True)

    @app.post("/v1/tasks/{task_id}/reject", response_model=TaskResponse)
    def reject_task(task_id: str, payload: ApprovalRequest) -> TaskResponse:
        return _approval_action(task_id, payload, approved=False)

    @app.post("/v1/tasks/{task_id}/retry", response_model=TaskResponse)
    def retry_task(task_id: str, payload: RetryRequest) -> TaskResponse:
        try:
            return _response(
                service.retry(task_id, expected_version=payload.expected_version)
            )
        except TaskNotFoundError as exc:
            raise HTTPException(status_code=404, detail="task not found") from exc
        except TaskValidationError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except (StaleTaskVersion, InvalidTaskTransition) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.get("/v1/tasks/{task_id}/events")
    async def task_events(
        request: Request,
        task_id: str,
        after: str | None = Query(default=None),
        last_event_id: str | None = Header(default=None, alias="Last-Event-ID"),
    ) -> StreamingResponse:
        try:
            await asyncio.to_thread(service.get, task_id)
        except TaskNotFoundError as exc:
            raise HTTPException(status_code=404, detail="task not found") from exc
        cursor = _validate_stream_id(after or last_event_id or "0-0")
        return StreamingResponse(
            _event_stream(
                request,
                store,
                task_id,
                cursor,
                settings.sse_heartbeat_seconds,
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache, no-transform",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    return app
