"""EvolutionManager — 自进化闭环编排。

run_cycle：扫所有 trigger → focus → mutate → eval → gate → create_version/record + journal。
evolve_skill：手动 FIX（/skill evolve）。
capture_skill：手动 CAPTURED（/skill capture）。

2a 同步触发。asyncio 后台留 2b（todo_docs/03 §3.8）。
"""
from __future__ import annotations

import uuid
from typing import Any

from agent4ml.backend.agents.journal.events import utc_now_iso
from agent4ml.backend.agents.skill.evolution.types import (
    EvalContext,
    EvolutionContext,
    EvolutionRecord,
)
from agent4ml.backend.agents.skill.types import SkillRecord


class EvolutionManager:
    """编排：trigger → focus → mutate → eval → gate → create_version/record + journal。

    2a 同步。支持 FIX + CAPTURED（DERIVED 留 2b）。
    """

    def __init__(
        self,
        store: Any,
        triggers: list[Any],
        focuser: Any,
        mutator: Any,
        eval_bridge: Any,
        gate: Any,
        llm: Any | None = None,
        journal: Any | None = None,
    ) -> None:
        self._store = store
        self._triggers = triggers
        self._focuser = focuser
        self._mutator = mutator
        self._eval_bridge = eval_bridge
        self._gate = gate
        self._llm = llm
        self._journal = journal

    def run_cycle(self) -> list[EvolutionRecord]:
        """扫所有 trigger，跑一轮自进化。返本轮所有 EvolutionRecord。"""
        records: list[EvolutionRecord] = []
        for trigger in self._triggers:
            contexts = trigger.should_trigger(self._store)
            for ctx in contexts:
                rec = self._run_evolution(ctx)
                if rec is not None:
                    records.append(rec)
                # anti-loop：MetricMonitorTrigger 标记已进化
                if ctx.target_skill is not None and hasattr(trigger, "mark_evolved"):
                    trigger.mark_evolved(ctx.target_skill.name, ctx.target_skill.total_selections)
        return records

    def evolve_skill(self, skill_name: str) -> EvolutionRecord:
        """手动触发单 skill FIX 进化（/skill evolve）。"""
        rec = self._store.get_active(skill_name)
        if rec is None:
            raise ValueError(f"skill not found: {skill_name}")
        ctx = EvolutionContext(
            trigger="METRIC",
            evolution_type="FIX",
            target_skill=rec,
            fix_direction="手动触发进化",
        )
        result = self._run_evolution(ctx)
        if result is None:
            raise RuntimeError("evolution produced no record")
        return result

    def capture_skill(self, pattern: str, suggested_name: str) -> EvolutionRecord:
        """手动 CAPTURED 沉淀新 skill（/skill capture）。"""
        # 优先用 CaptureTrigger.manual_capture（若注册）
        from agent4ml.backend.agents.skill.evolution.triggers.capture_trigger import CaptureTrigger
        ctx: EvolutionContext | None = None
        for t in self._triggers:
            if isinstance(t, CaptureTrigger):
                ctx = t.manual_capture(pattern, suggested_name)
                break
        if ctx is None:
            ctx = EvolutionContext(
                trigger="CAPTURE",
                evolution_type="CAPTURED",
                target_skill=None,
                capture_pattern=pattern,
                suggested_name=suggested_name,
            )
        result = self._run_evolution(ctx)
        if result is None:
            raise RuntimeError("capture produced no record")
        return result

    def _run_evolution(self, ctx: EvolutionContext) -> EvolutionRecord | None:
        """跑单次进化闭环。返 EvolutionRecord（accept/reject 均记）。"""
        # 1. focus
        ctx = self._focuser.focus(ctx, self._store)
        # 2. mutate
        candidate, diff = self._mutator.mutate(ctx, self._llm)
        # 3. eval
        baseline = ctx.target_skill
        metrics_baseline = None
        if baseline is not None:
            try:
                metrics_baseline = self._store.get_metrics(baseline.skill_id)
            except Exception:
                metrics_baseline = None
        eval_ctx = EvalContext(
            baseline=baseline if baseline is not None else candidate,
            candidate=candidate,
            metrics_baseline=metrics_baseline,
        )
        eval_result = self._eval_bridge.evaluate(eval_ctx)
        # 4. gate
        decision = self._gate.decide(candidate, baseline, eval_result)  # type: ignore[arg-type]
        # 5. create_version if accept
        created_id: str | None = None
        if decision.recommendation in ("accept", "accept_new_best"):
            parent_id = baseline.skill_id if baseline is not None else ""
            try:
                created_id = self._store.create_version(parent_id, candidate, candidate.lineage.origin)
            except Exception:
                created_id = None
        # 6. record
        rec = EvolutionRecord(
            evolution_id=f"evo_{uuid.uuid4().hex[:12]}",
            skill_name=candidate.name,
            evolution_type=ctx.evolution_type,
            trigger=ctx.trigger,
            baseline_id=baseline.skill_id if baseline is not None else None,
            candidate_id=candidate.skill_id,
            failure_focus=ctx.fix_direction or ctx.capture_pattern,
            mutation_diff=diff,
            eval_score=eval_result.score,
            gate_decision=decision.recommendation,
            created_version_id=created_id,
            timestamp=utc_now_iso(),
        )
        try:
            self._store.record_evolution(rec)
        except Exception:
            pass
        # 7. journal
        if self._journal is not None:
            self._emit_journal(ctx, decision, rec)
        return rec

    def _emit_journal(
        self, ctx: EvolutionContext, decision: Any, rec: EvolutionRecord,
    ) -> None:
        """journal 事件：skill.evolve / skill.captured / skill.evolve_rejected。"""
        if ctx.evolution_type == "CAPTURED":
            event_type = "skill.captured"
        elif decision.recommendation in ("accept", "accept_new_best"):
            event_type = "skill.evolve"
        else:
            event_type = "skill.evolve_rejected"
        try:
            self._journal.append(event_type, {
                "evolution_id": rec.evolution_id,
                "skill_name": rec.skill_name,
                "evolution_type": rec.evolution_type,
                "eval_score": rec.eval_score,
                "gate_decision": rec.gate_decision,
                "created_version_id": rec.created_version_id,
            })
        except Exception:
            pass
