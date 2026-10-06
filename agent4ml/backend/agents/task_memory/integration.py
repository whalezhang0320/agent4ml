from __future__ import annotations

import asyncio
import json
import re
from dataclasses import replace
from pathlib import Path
from typing import Any

from langchain_core.messages import ToolMessage
from langgraph.types import Command

from agent4ml.backend.agents.middlewares.run_journal_middleware import _get_runtime_value
from agent4ml.backend.agents.task_memory.models import FlushReceipt
from agent4ml.backend.agents.task_memory.projector import ContextProjector
from agent4ml.backend.agents.task_memory.reader import TaskMemoryReader
from agent4ml.backend.agents.task_memory.writer import TaskGraphWriter, redact_secrets

TASK_MEMORY_META_KEY = "agent4ml.task_memory"
TASK_MEMORY_READ_TOOL_NAME = "task_memory_read"


def serialize_tool_content(content: Any) -> str:
    """Losslessly normalize JSON-compatible message blocks for the evidence ref."""
    if isinstance(content, str):
        return content
    try:
        return json.dumps(content, ensure_ascii=False, sort_keys=True, default=str)
    except (TypeError, ValueError):
        return str(content)


def iter_tool_messages(result: Any) -> list[ToolMessage]:
    if isinstance(result, ToolMessage):
        return [result]
    if isinstance(result, Command) and isinstance(result.update, dict):
        messages = result.update.get("messages") or []
        if isinstance(messages, ToolMessage):
            messages = [messages]
        return [
            message for message in messages
            if isinstance(message, ToolMessage)
        ]
    return []


def replace_tool_messages(result: Any, replacements: dict[int, ToolMessage]) -> Any:
    """Replace ToolMessages in a naked result or a Command without losing updates."""
    if isinstance(result, ToolMessage):
        return replacements.get(id(result), result)
    if isinstance(result, Command) and isinstance(result.update, dict):
        update = dict(result.update)
        messages = update.get("messages") or []
        if isinstance(messages, ToolMessage):
            messages = [messages]
        update["messages"] = [
            replacements.get(id(message), message)
            for message in messages
        ]
        return replace(result, update=update)
    return result


class TaskMemoryService:
    """Bridge runtime context and LangChain messages to the durable writer."""

    def __init__(self, root: str | Path, *, projection_max_chars: int = 12_000) -> None:
        self.root = Path(root)
        self.projector = ContextProjector(max_chars=projection_max_chars)

    def checkpoint(self, ctx: Any) -> tuple[FlushReceipt, str | None]:
        identity = self._identity(ctx)
        if identity is None:
            return FlushReceipt.failed("", "thread_id is unavailable"), None
        thread_id, task_id, _run_id = identity
        writer = TaskGraphWriter(self.root, thread_id, task_id)
        goal = self._goal(ctx)
        receipt = writer.ensure_task(
            goal,
            success_criteria=self._success_criteria(ctx),
            constraints=self._constraints(ctx),
        )
        if not receipt.safe_to_drop:
            return receipt, None
        try:
            graph = writer.sync_todos(list(ctx.state.get("todos") or []))
            receipt = writer._receipt(graph)
            return receipt, self.projector.project(graph)
        except Exception as exc:
            return FlushReceipt.failed(task_id, exc), None

    def persist_tool_message(
        self,
        ctx: Any,
        message: ToolMessage,
        *,
        tool_call: dict[str, Any] | None = None,
    ) -> FlushReceipt:
        identity = self._identity(ctx)
        if identity is None:
            return FlushReceipt.failed("", "thread_id is unavailable")
        thread_id, task_id, run_id = identity
        writer = TaskGraphWriter(self.root, thread_id, task_id)
        ensured = writer.ensure_task(
            self._goal(ctx),
            success_criteria=self._success_criteria(ctx),
            constraints=self._constraints(ctx),
        )
        if not ensured.safe_to_drop:
            return ensured
        try:
            writer.sync_todos(list(ctx.state.get("todos") or []))
        except Exception as exc:
            return FlushReceipt.failed(task_id, exc)

        call = tool_call or {}
        args = call.get("args") if isinstance(call, dict) else {}
        explicit_node = args.get("node_id") if isinstance(args, dict) else None
        tool_name = message.name or str(call.get("name") or "unknown")
        tool_call_id = message.tool_call_id or str(call.get("id") or "")
        if not tool_call_id:
            return FlushReceipt.failed(task_id, "tool_call_id is required")
        content = serialize_tool_content(message.content)
        summary = re.sub(r"\s+", " ", content).strip()[:300]
        return writer.commit_tool_result(
            content=content,
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            tool_status=(
                "error" if getattr(message, "status", None) == "error" else "ok"
            ),
            run_id=run_id,
            message_id=getattr(message, "id", None),
            explicit_node_id=str(explicit_node) if explicit_node is not None else None,
            input_summary=self._input_summary(tool_name, args),
            summary=summary,
            goal=self._goal(ctx),
            success_criteria=self._success_criteria(ctx),
            constraints=self._constraints(ctx),
        )

    def bind(self, ctx: Any) -> "BoundTaskMemorySink":
        return BoundTaskMemorySink(self, ctx)

    def verify_persisted_message(
        self, ctx: Any, message: ToolMessage
    ) -> FlushReceipt | None:
        metadata = (message.additional_kwargs or {}).get(TASK_MEMORY_META_KEY)
        if not isinstance(metadata, dict) or not metadata.get("safe_to_drop"):
            return None
        identity = self._identity(ctx)
        if identity is None or not getattr(message, "id", None):
            return None
        thread_id, task_id, _run_id = identity
        if metadata.get("task_id") != task_id:
            return None
        result_ref = metadata.get("result_ref")
        try:
            reader = TaskMemoryReader(self.root, thread_id, task_id)
            reader.read_ref(str(result_ref))
            graph = reader.writer.get_graph()
        except Exception:
            return None
        message_id = str(message.id)
        return FlushReceipt(
            task_id=task_id,
            graph_version=graph.graph_version,
            persisted_message_ids=(message_id,),
            event_ids=tuple(metadata.get("event_ids") or ()),
            result_refs=(str(result_ref),),
            safe_to_drop=True,
            graph_path=str(reader.store.graph_path),
            current_node=graph.current_node,
            status=graph.status,
            message_refs={message_id: str(result_ref)},
        )

    def _identity(self, ctx: Any) -> tuple[str, str, str] | None:
        runtime = getattr(ctx, "runtime", None)
        thread_id = _get_runtime_value(runtime, "thread_id", None)
        if not thread_id:
            return None
        pointer = ctx.state.get("task_memory") or {}
        task_id = (
            _get_runtime_value(runtime, "task_id", None)
            or pointer.get("task_id")
            or thread_id
        )
        run_id = _get_runtime_value(runtime, "run_id", None) or "unknown-run"
        return str(thread_id), str(task_id), str(run_id)

    @staticmethod
    def _goal(ctx: Any) -> str:
        state = ctx.state or {}
        return str(
            state.get("research_question")
            or state.get("user_input")
            or "Continue the current Agent4ML task"
        )

    @staticmethod
    def _success_criteria(ctx: Any) -> list[str]:
        state = ctx.state or {}
        criteria = state.get("success_criteria")
        if isinstance(criteria, list):
            return [str(item) for item in criteria if str(item).strip()]
        return []

    @staticmethod
    def _constraints(ctx: Any) -> list[str]:
        intent = (ctx.state or {}).get("intent")
        if isinstance(intent, dict):
            values = intent.get("constraints") or []
        else:
            values = getattr(intent, "constraints", ()) if intent is not None else ()
        return [str(item) for item in values if str(item).strip()]

    @staticmethod
    def _input_summary(tool_name: str, args: Any) -> str:
        if not isinstance(args, dict):
            return f"{tool_name} input"
        safe: dict[str, Any] = {}
        for key in ("query", "q", "url", "path", "pattern"):
            if key in args:
                safe[key] = redact_secrets(str(args[key]))[:160]
        if not safe:
            return f"{tool_name} fields={sorted(str(key) for key in args)[:12]}"
        return f"{tool_name} " + json.dumps(safe, ensure_ascii=False, sort_keys=True)


class BoundTaskMemorySink:
    """MemorySink bound to one governance hook and its runtime identity."""

    def __init__(self, service: TaskMemoryService, ctx: Any) -> None:
        self.service = service
        self.ctx = ctx

    def flush(self, messages_to_drop: list) -> FlushReceipt:
        checkpoint, _projection = self.service.checkpoint(self.ctx)
        if not checkpoint.safe_to_drop:
            return checkpoint
        if any(not isinstance(message, ToolMessage) for message in messages_to_drop):
            return FlushReceipt.failed(
                checkpoint.task_id,
                "task-memory sink only covers ToolMessage evidence",
            )
        tool_messages = [m for m in messages_to_drop if isinstance(m, ToolMessage)]
        if not tool_messages:
            return checkpoint
        receipts: list[FlushReceipt] = []
        for message in tool_messages:
            if not getattr(message, "id", None):
                return FlushReceipt.failed(
                    checkpoint.task_id,
                    "ToolMessage without a stable message id cannot be removed",
                )
            receipt = self.service.verify_persisted_message(self.ctx, message)
            if receipt is None:
                if (message.additional_kwargs or {}).get("agent4ml.externalized"):
                    return FlushReceipt.failed(
                        checkpoint.task_id,
                        "legacy externalized result lacks a verified task-memory receipt",
                    )
                receipt = self.service.persist_tool_message(self.ctx, message)
            if not receipt.safe_to_drop:
                return receipt
            receipts.append(receipt)
        return self._combine(checkpoint, receipts)

    async def aflush(self, messages_to_drop: list) -> FlushReceipt:
        return await asyncio.to_thread(self.flush, messages_to_drop)

    @staticmethod
    def _combine(
        checkpoint: FlushReceipt, receipts: list[FlushReceipt]
    ) -> FlushReceipt:
        message_ids = tuple(
            dict.fromkeys(
                item for receipt in receipts for item in receipt.persisted_message_ids
            )
        )
        event_ids = tuple(
            dict.fromkeys(item for receipt in receipts for item in receipt.event_ids)
        )
        result_refs = tuple(
            dict.fromkeys(item for receipt in receipts for item in receipt.result_refs)
        )
        message_refs: dict[str, str] = {}
        for receipt in receipts:
            message_refs.update(receipt.message_refs)
        latest = receipts[-1] if receipts else checkpoint
        return FlushReceipt(
            task_id=checkpoint.task_id,
            graph_version=latest.graph_version,
            persisted_message_ids=message_ids,
            event_ids=event_ids,
            result_refs=result_refs,
            safe_to_drop=True,
            graph_path=latest.graph_path,
            current_node=latest.current_node,
            status=latest.status,
            message_refs=message_refs,
        )
