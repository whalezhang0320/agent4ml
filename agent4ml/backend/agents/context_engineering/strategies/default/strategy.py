from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import datetime
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agent4ml.backend.agents.context_engineering.contract import (
    GovernanceContext,
    GovernanceResult,
)
from agent4ml.backend.agents.context_engineering.registry import register_strategy
from agent4ml.backend.agents.context_engineering.strategies.default._constants import CST
from agent4ml.backend.agents.context_engineering.strategies.default.budget import (
    BudgetTrackerExecutor,
)
from agent4ml.backend.agents.context_engineering.strategies.default.externalizer import (
    ExternalizerExecutor,
)
from agent4ml.backend.agents.context_engineering.strategies.default.snapshot import (
    SnapshotExecutor,
)
from agent4ml.backend.agents.context_engineering.strategies.default.summarizer import (
    SummarizerExecutor,
)
from agent4ml.backend.agents.context_engineering.utilities import resolve_window_size
from agent4ml.backend.agents.middlewares.tagged_context_middleware import AGENT4ML_THINKING
from agent4ml.backend.agents.middlewares.run_journal_middleware import _get_runtime_value
from agent4ml.backend.agents.task_memory.integration import (
    TASK_MEMORY_READ_TOOL_NAME,
    TaskMemoryService,
    iter_tool_messages,
    replace_tool_messages,
)

logger = logging.getLogger(__name__)

_DEFAULT_THRESHOLDS: dict[str, float] = {
    "p1_externalize": 0.40,
    "p2_thinking": 0.50,
    "p3_observations": 0.60,
    "p4_summarize": 0.80,
    "p5_stop_toolcall": 0.90,
    "hard_stop": 0.99,
}


@register_strategy("default")
class DefaultStrategy:
    """默认上下文治理策略：分段优先级舍弃 P0-P5。"""

    def __init__(self, params: dict | None = None, model: Any = None, summarize_model: Any = None) -> None:
        params = params or {}
        self._model = model
        self._summarize_model_name: str | None = params.get("summarize_model")
        self._thresholds: dict[str, float] = {
            **_DEFAULT_THRESHOLDS,
            **(params.get("thresholds") or {}),
        }
        self._budget = BudgetTrackerExecutor(self._thresholds)
        self._externalizer = ExternalizerExecutor(
            externalize_dir=params.get("externalize_dir", ".agent4ml/externalized"),
            min_chars=params.get("externalize_min_chars", 500),
            preview_chars=params.get("externalize_preview_chars", 500),
            exempt_rounds=params.get("exempt_rounds", 2),
            tool_metadata=params.get("tool_metadata"),
        )
        self._summarizer = SummarizerExecutor(
            model=summarize_model or model,
            preserve_recent=params.get("preserve_recent", 6),
        )
        self._snapshot = SnapshotExecutor(
            snapshot_dir=params.get("snapshot_dir", ".agent4ml/snapshots"),
        )
        self._task_memory: TaskMemoryService | None = None
        if params.get("task_memory_enabled", True):
            self._task_memory = TaskMemoryService(
                params.get("task_memory_dir", ".agent4ml/task_memory"),
                projection_max_chars=params.get("task_memory_projection_max_chars", 12_000),
            )

    def before_agent(self, ctx: GovernanceContext) -> GovernanceResult:
        governance = self._budget.init_budget(ctx.governance)
        governance, pointer = self._checkpoint_task_memory(ctx, governance)
        return GovernanceResult(state_patch=self._state_patch(governance, pointer))

    async def abefore_agent(self, ctx: GovernanceContext) -> GovernanceResult:
        return await asyncio.to_thread(self.before_agent, ctx)

    def after_agent(self, ctx: GovernanceContext) -> GovernanceResult:
        governance, pointer = self._checkpoint_task_memory(ctx, ctx.governance or {})
        governance = self._budget.clear_run_state(governance)
        return GovernanceResult(state_patch=self._state_patch(governance, pointer))

    async def aafter_agent(self, ctx: GovernanceContext) -> GovernanceResult:
        return await asyncio.to_thread(self.after_agent, ctx)

    def before_model(self, ctx: GovernanceContext) -> GovernanceResult:
        # 防御性配对校验：送 LLM 前确保 AIMessage(tool_calls) 都有配对 ToolMessage。
        # 不早退——pairing 补全与 compaction（P1/P4）并行执行，避免 pairing 问题阻塞压缩。
        pairing_patch = self._ensure_pairing(ctx.messages or [])
        if pairing_patch:
            logger.warning("pairing 补全：%d 个缺失 ToolMessage", len(pairing_patch))

        governance, task_pointer = self._checkpoint_task_memory(
            ctx, ctx.governance or {}
        )
        task_memory_required = (
            self._task_memory is not None and self._has_thread_identity(ctx)
        )
        memory_sink = self._task_memory.bind(ctx) if (
            self._task_memory is not None and task_pointer is not None
        ) else None
        d = governance.get("default") or {}
        pending = d.get("pending") or []
        fraction = (d.get("budget") or {}).get("fraction", 0.0)

        # P3: observations top-N（渲染层筛）。
        # 当前 observations 不进上下文块（Q5 舍弃 <observations> 标签），
        # P3 设 p3_obs_limit flag 供未来渲染层消费，当前为 no-op。
        if "P3" in pending:
            obs_count = len(ctx.state.get("observations") or [])
            p3_limit = max(3, obs_count // 3) if fraction >= 0.60 else obs_count
            d["p3_obs_limit"] = p3_limit
            governance = dict(governance)
            governance["default"] = d
            logger.info("compaction P3 observations top-N=%d (total=%d, fraction=%.2f)", p3_limit, obs_count, fraction)
            self._emit_trace(ctx, "triggered", "P3", obs_limit=p3_limit, obs_total=obs_count, fraction=fraction)

        p4_skip_until = d.get("p4_skip_until_fraction", 0.0)
        p4_suppressed = "P4" in pending and fraction < p4_skip_until
        if p4_suppressed:
            logger.info(
                "compaction P4 skipped (interval suppression): fraction=%.2f < skip_until=%.2f",
                fraction,
                p4_skip_until,
            )
            self._emit_trace(
                ctx,
                "skipped",
                "P4",
                reason="interval_suppression",
                fraction=fraction,
            )

        if "P4" in pending and not p4_suppressed:
            logger.info("compaction P4 summarize 触发 fraction=%.2f", fraction)
            self._emit_trace(ctx, "triggered", "P4", fraction=fraction, window=(d.get("budget") or {}).get("window", 0), pending=pending)
            if task_memory_required and task_pointer is None:
                logger.error("P4 task-memory checkpoint failed; preserving original context")
                return GovernanceResult(
                    state_patch=self._state_patch(governance, task_pointer),
                    messages_patch=pairing_patch,
                )
            snapshot_gov = self._snapshot.snapshot_if_pending(
                governance, ctx.messages or [], ctx.state
            )
            if snapshot_gov is None:
                logger.error("P4 snapshot failed; preserving original context")
                return GovernanceResult(
                    state_patch=self._state_patch(governance, task_pointer),
                    messages_patch=pairing_patch,
                )
            result = self._summarizer.summarize_if_pending(
                snapshot_gov,
                ctx.messages or [],
                self._externalizer,
                memory_sink=memory_sink,
            )
            self._emit_trace(ctx, "completed", "P4", fraction_after=fraction)
            if result is not None:
                state_patch = dict(result.state_patch or {})
                result_governance = dict(
                    state_patch.get("governance") or snapshot_gov
                )
                result_default = dict(result_governance.get("default") or {})
                result_default["p4_skip_until_fraction"] = fraction + 0.10
                result_governance["default"] = result_default
                state_patch["governance"] = result_governance
                if task_pointer is not None:
                    state_patch["task_memory"] = task_pointer
                # 合并 pairing_patch 进 compaction result
                if pairing_patch:
                    combined_msgs = (result.messages_patch or []) + pairing_patch
                    return GovernanceResult(
                        state_patch=state_patch,
                        messages_patch=combined_msgs,
                        jump_to=result.jump_to,
                    )
                return GovernanceResult(
                    state_patch=state_patch or None,
                    messages_patch=result.messages_patch,
                    metrics=result.metrics,
                    jump_to=result.jump_to,
                )
            # Summary/checkpoint failure is fail-closed: keep all original messages.
            return GovernanceResult(
                state_patch=self._state_patch(snapshot_gov, task_pointer),
                messages_patch=pairing_patch,
            )

        if "P1" in pending:
            # 间隔抑制：fraction 未涨超 p1_skip_until_fraction → 跳过
            skip_until = d.get("p1_skip_until_fraction", 0.0)
            if fraction < skip_until:
                logger.info("compaction P1 skipped (interval suppression): fraction=%.2f < skip_until=%.2f", fraction, skip_until)
                self._emit_trace(ctx, "skipped", "P1", reason="interval_suppression", fraction=fraction)
            else:
                logger.info("compaction P1 externalize 触发 fraction=%.2f", fraction)
                self._emit_trace(ctx, "triggered", "P1", fraction=fraction, pending=pending)
                if task_memory_required and memory_sink is None:
                    logger.error("P1 task-memory checkpoint failed; preserving tool results")
                    return GovernanceResult(
                        state_patch=self._state_patch(governance, task_pointer),
                        messages_patch=pairing_patch,
                    )
                messages_patch = self._externalizer.externalize_history(
                    ctx.messages or [], memory_sink=memory_sink
                )
                ext_count = len(messages_patch) if messages_patch else 0
                self._emit_trace(ctx, "completed", "P1", externalized_count=ext_count, fraction_after=fraction)
                # 更新间隔抑制 flag
                ext_count = len(messages_patch) if messages_patch else 0
                new_skip = fraction + 0.10
                d["p1_skip_until_fraction"] = new_skip
                d["p1_completed"] = (ext_count == 0)
                governance["default"] = d
                if messages_patch:
                    if pairing_patch:
                        messages_patch = messages_patch + pairing_patch
                    return GovernanceResult(
                        state_patch=self._state_patch(governance, task_pointer),
                        messages_patch=messages_patch,
                    )
                # P1 执行但无可外化内容 → 更新 governance 后继续检查 pairing_patch
                logger.info("compaction P1 completed (no externalizable content, p1_completed=True)")
                if pairing_patch:
                    return GovernanceResult(
                        state_patch=self._state_patch(governance, task_pointer),
                        messages_patch=pairing_patch,
                    )
                return GovernanceResult(
                    state_patch=self._state_patch(governance, task_pointer)
                )

        # 无 compaction 触发，仅返 pairing 补全 + P3 governance 更新（若有）
        if pairing_patch:
            return GovernanceResult(
                state_patch=self._state_patch(governance, task_pointer),
                messages_patch=pairing_patch,
            )
        if "P3" in pending or task_pointer is not None:
            return GovernanceResult(
                state_patch=self._state_patch(governance, task_pointer)
            )
        return GovernanceResult()

    @staticmethod
    def _ensure_pairing(messages: list) -> list | None:
        """校验 AIMessage(tool_calls) 都有配对 ToolMessage，缺则补 error ToolMessage。"""
        tool_msg_ids = {m.tool_call_id for m in messages if isinstance(m, ToolMessage)}
        patch = []
        for msg in messages:
            if isinstance(msg, AIMessage):
                for tc in msg.tool_calls or []:
                    tc_id = tc.get("id") if isinstance(tc, dict) else None
                    if tc_id and tc_id not in tool_msg_ids:
                        patch.append(ToolMessage(
                            content=f"⚠️ 工具结果缺失（{tc_id}），已补占位。",
                            tool_call_id=tc_id,
                            status="error",
                        ))
        return patch or None

    @staticmethod
    def _emit_compaction_event(event_type: str, **fields: Any) -> None:
        """发 compaction custom event 到 stream。无 graph context 时 fallback logger。"""
        try:
            from langgraph.config import get_stream_writer

            get_stream_writer()({"type": event_type, **fields})
        except Exception:
            logger.info("compaction event (no stream writer): %s %s", event_type, fields)

    def _emit_trace(self, ctx: GovernanceContext, event: str, stage: str, **fields: Any) -> None:
        """compaction 全流程 trace：写 compaction.jsonl + journal 关键事件 + stream 事件。"""
        runtime = ctx.runtime
        ts = datetime.now(CST).isoformat()
        record = {"ts": ts, "event": event, "stage": stage, **fields}

        # 1. 写 compaction.jsonl
        run_dir = self._get_run_dir(runtime)
        if run_dir:
            try:
                os.makedirs(run_dir, exist_ok=True)
                with open(os.path.join(run_dir, "compaction.jsonl"), "a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
            except (OSError, TypeError) as exc:
                logger.warning("compaction trace write failed: %s", exc)

        # 2. 关键事件入 journal（events.jsonl）
        if event in ("triggered", "completed", "skipped"):
            try:
                from agent4ml.backend.agents.middlewares.run_journal_middleware import _get_runtime_value
                journal = _get_runtime_value(runtime, "journal", None)
                if journal is not None:
                    journal.append(f"compaction.{event}", {"stage": stage, **fields})
            except Exception:
                pass

        # 3. stream 事件（CLI 渲染 compaction_start/end）
        stream_type = {"triggered": "compaction_start", "completed": "compaction_end"}.get(event, f"compaction_{event}")
        self._emit_compaction_event(stream_type, tool_name=stage, **fields)

    @staticmethod
    def _get_run_dir(runtime: Any) -> str | None:
        """从 runtime 取 run 日志目录路径。"""
        try:
            from agent4ml.backend.agents.middlewares.run_journal_middleware import _get_runtime_value
            tid = _get_runtime_value(runtime, "thread_id", None)
            rid = _get_runtime_value(runtime, "run_id", None)
            if tid and rid:
                return os.path.join(".agent4ml", "logs", "threads", str(tid), "runs", str(rid))
        except Exception:
            pass
        return None

    async def abefore_model(self, ctx: GovernanceContext) -> GovernanceResult:
        return await asyncio.to_thread(self.before_model, ctx)

    def after_model(self, ctx: GovernanceContext) -> GovernanceResult:
        messages = ctx.messages or []
        config = ctx.config or {}
        # window 优先 config 显式配置，无则 resolve_window_size(model) 查表（model 属性 → model_name 映射 → 128000）
        config_window = config.get("window") if isinstance(config, dict) else None
        window = config_window or resolve_window_size(self._model)
        governance = self._budget.track(ctx.governance, messages, ctx.token_counter, window)
        messages_patch = self._mark_thinking(messages)

        # P5/P99：fraction >= 90% → 剥 tool_calls + 收尾提示 + jump model（warned 防重复触发死循环）
        d = governance.get("default") or {}
        pending = d.get("pending") or []
        warned = d.get("warned", False)
        fraction = (d.get("budget") or {}).get("fraction", 0.0)

        if "P5" in pending and not warned:
            last_ai = next(
                (m for m in reversed(messages)
                 if isinstance(m, AIMessage) and getattr(m, "tool_calls", None)),
                None,
            )
            if last_ai:
                cleared = self._strip_tool_calls(last_ai, fraction)
                stop_msg = HumanMessage(
                    name="context_budget_stop",
                    additional_kwargs={"hide_from_ui": True},
                    content=self._build_stop_message(fraction),
                )
                d["warned"] = True
                governance["default"] = d
                combined_patch = (messages_patch or []) + [cleared, stop_msg]
                self._emit_trace(ctx, "triggered", "P5", fraction=fraction, action="strip_tool_calls")
                return GovernanceResult(
                    state_patch={"governance": governance},
                    messages_patch=combined_patch,
                    jump_to="model",
                )

        return GovernanceResult(state_patch={"governance": governance}, messages_patch=messages_patch)

    @staticmethod
    def _strip_tool_calls(last_ai: AIMessage, fraction: float) -> AIMessage:
        """剥 tool_calls 保 pairing：复用原 id → add_messages 替换（非追加）。

        同步剥 additional_kwargs.tool_calls/function_call 防 _has_tool_call_intent 误判。
        原始 AIMessage(tool_calls) 被 id 匹配替换 → jump model 跳过 ToolNode → 无孤立 tool_calls →
        下一轮 model 调用不 400。
        """
        new_kwargs = {
            k: v for k, v in (last_ai.additional_kwargs or {}).items()
            if k not in ("tool_calls", "function_call")
        }
        new_kwargs["context_budget_stop"] = round(fraction, 4)
        return AIMessage(
            id=last_ai.id,
            content=last_ai.content or "",
            tool_calls=[],
            additional_kwargs=new_kwargs,
        )

    @staticmethod
    def _build_stop_message(fraction: float) -> str:
        if fraction >= 0.99:
            return (
                "<system_reminder>\n"
                "上下文窗口占用已达 99% 硬底线，强制收尾。"
                "不可再调用任何工具，请立即基于已有信息给出最终答案。\n"
                "</system_reminder>"
            )
        return (
            "<system_reminder>\n"
            f"上下文窗口占用已达 {fraction:.0%}，超过 90% 阈值。"
            "请基于已有信息收尾，不再调用工具。若证据不足，基于现有信息给出最佳答案并说明缺口。\n"
            "</system_reminder>"
        )

    @staticmethod
    def _mark_thinking(messages: list) -> list | None:
        patch = []
        for msg in messages:
            if (isinstance(msg, AIMessage)
                    and msg.additional_kwargs.get("reasoning_content")
                    and not msg.additional_kwargs.get(AGENT4ML_THINKING)
                    and not msg.tool_calls):
                new_kwargs = {**msg.additional_kwargs, AGENT4ML_THINKING: True}
                patch.append(msg.model_copy(update={"additional_kwargs": new_kwargs}))
        return patch or None

    async def aafter_model(self, ctx: GovernanceContext) -> GovernanceResult:
        return self.after_model(ctx)

    def wrap_model_call(self, ctx: GovernanceContext) -> GovernanceResult:
        return GovernanceResult()

    async def awrap_model_call(self, ctx: GovernanceContext) -> GovernanceResult:
        return GovernanceResult()

    def wrap_tool_call(self, ctx: GovernanceContext) -> GovernanceResult:
        tool_result = ctx.tool_result
        tool_call = getattr(ctx.tool_call_request, "tool_call", None) or {}
        # This tool only expands state/evidence already protected by the same task-memory
        # store. Re-ingesting its output would duplicate refs and, for read_ref, immediately
        # replace the requested evidence with another short receipt.
        if tool_call.get("name") == TASK_MEMORY_READ_TOOL_NAME:
            return GovernanceResult()
        if self._task_memory is not None and self._has_thread_identity(ctx):
            replacements: dict[int, ToolMessage] = {}
            for message in iter_tool_messages(tool_result):
                receipt = self._task_memory.persist_tool_message(
                    ctx, message, tool_call=tool_call
                )
                if not receipt.safe_to_drop:
                    logger.error(
                        "task-memory commit failed for tool_call_id=%s: %s",
                        message.tool_call_id,
                        receipt.errors,
                    )
                    continue
                rewritten = self._externalizer.rewrite_from_receipt(message, receipt)
                if rewritten is not None:
                    replacements[id(message)] = rewritten
            if replacements:
                return GovernanceResult(
                    request_override=replace_tool_messages(tool_result, replacements)
                )
            return GovernanceResult()
        if isinstance(tool_result, ToolMessage):
            rewritten = self._externalizer.externalize_if_needed(tool_result)
            if rewritten is not None:
                return GovernanceResult(request_override=rewritten)
        return GovernanceResult()

    async def awrap_tool_call(self, ctx: GovernanceContext) -> GovernanceResult:
        return await asyncio.to_thread(self.wrap_tool_call, ctx)

    def _checkpoint_task_memory(
        self, ctx: GovernanceContext, governance: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any] | None]:
        if self._task_memory is None or not self._has_thread_identity(ctx):
            return governance, None
        receipt, projection = self._task_memory.checkpoint(ctx)
        g = dict(governance or {})
        d = dict(g.get("default") or {})
        metrics = dict(d.get("metrics") or {})
        if receipt.safe_to_drop and projection is not None:
            d["task_memory_projection"] = projection
            d["task_memory_graph_version"] = receipt.graph_version
            d.pop("task_memory_error", None)
            metrics["task_memory_graph_version"] = receipt.graph_version
            pointer = receipt.pointer()
        else:
            d["task_memory_error"] = "; ".join(receipt.errors)
            metrics["task_memory_validation_failures"] = (
                metrics.get("task_memory_validation_failures", 0) + 1
            )
            pointer = None
        d["metrics"] = metrics
        g["default"] = d
        return g, pointer

    @staticmethod
    def _state_patch(
        governance: dict[str, Any], pointer: dict[str, Any] | None
    ) -> dict[str, Any]:
        patch: dict[str, Any] = {"governance": governance}
        if pointer is not None:
            patch["task_memory"] = pointer
        return patch

    @staticmethod
    def _has_thread_identity(ctx: GovernanceContext) -> bool:
        return bool(_get_runtime_value(ctx.runtime, "thread_id", None))
