"""Deterministic offline A/B benchmark for Agent4ML task memory.

The benchmark deliberately replays fixed tool outputs instead of calling an LLM
or the network.  It measures context size, governance overhead, durable storage,
evidence recovery, and traceability.  It does *not* claim that replay wall time
is model latency; a live-model layer can consume the generated JSONL later.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Literal, Sequence

from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, ToolMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES

from agent4ml.backend.agents.context_engineering.contract import (
    GovernanceContext,
    GovernanceResult,
)
from agent4ml.backend.agents.context_engineering.strategies.default.strategy import (
    DefaultStrategy,
)
from agent4ml.backend.agents.context_engineering.utilities import token_counter
from agent4ml.backend.agents.middlewares.tagged_context_middleware import (
    AGENT4ML_EXTERNALIZED,
    ContextAssembler,
)
from agent4ml.backend.agents.state.thread_state import create_initial_thread_state
from agent4ml.backend.agents.task_memory import TaskMemoryReader

Variant = Literal["raw", "legacy", "task_memory"]
VARIANTS: tuple[Variant, ...] = ("raw", "legacy", "task_memory")


@dataclass(frozen=True)
class Scenario:
    name: str
    description: str
    rounds: int
    payload_chars: int
    human_chars: int
    assistant_chars: int


SCENARIOS: tuple[Scenario, ...] = (
    Scenario(
        name="short_control",
        description="短任务控制组：验证任务记忆的固定开销",
        rounds=3,
        payload_chars=300,
        human_chars=160,
        assistant_chars=160,
    ),
    Scenario(
        name="tool_heavy",
        description="大量工具结果：观察上下文和磁盘成本",
        rounds=20,
        payload_chars=12_000,
        human_chars=220,
        assistant_chars=240,
    ),
    Scenario(
        name="needle_recovery",
        description="早期精确事实恢复：事实位于工具输出尾部",
        rounds=16,
        payload_chars=8_000,
        human_chars=260,
        assistant_chars=220,
    ),
    Scenario(
        name="dialogue_heavy",
        description="对话密集：观察当前渐进式 P4 策略的边界",
        rounds=24,
        payload_chars=300,
        human_chars=1_200,
        assistant_chars=1_000,
    ),
)


@dataclass(frozen=True)
class RunMetrics:
    scenario: str
    description: str
    variant: Variant
    repeat: int
    seed: int
    profile: str
    window_tokens: int
    model_input_calls: int
    cumulative_input_tokens: int
    peak_context_tokens: int
    final_context_tokens: int
    replay_wall_ms: float
    governance_ms: float
    storage_bytes: int
    evidence_bytes: int
    recoverable_fact_rate: float
    current_fact_rate: float
    constraint_retention_rate: float
    traceability_rate: float
    integrity_ok: bool
    graph_version: int
    tool_results: int
    final_messages: int


class _ExtractiveSummaryModel:
    """Stable local summary model used only by the legacy P4 path."""

    @staticmethod
    def invoke(prompt: Any, config: dict[str, Any] | None = None) -> Any:
        del config
        text = str(prompt)
        retained: list[str] = []
        for token in text.replace("\n", " ").split():
            if token.startswith("CONSTRAINT_") and token not in retained:
                retained.append(token)
        summary = "OFFLINE_BENCH_SUMMARY"
        if retained:
            summary += " " + " ".join(retained)
        return SimpleNamespace(text=summary)


def _scenario_map() -> dict[str, Scenario]:
    return {scenario.name: scenario for scenario in SCENARIOS}


def _fill_text(prefix: str, target_chars: int, rng: random.Random) -> str:
    parts = [prefix]
    index = 0
    while sum(len(part) + 1 for part in parts) < target_chars:
        parts.append(
            f"record_{index:05d} metric_{rng.randrange(1_000_000):06d} "
            f"sample_{rng.randrange(1_000_000):06d}"
        )
        index += 1
    return " ".join(parts)[:target_chars]


def _round_payload(
    scenario: Scenario, round_index: int, rng: random.Random
) -> tuple[str, str]:
    fact = f"BENCH_FACT_{round_index:03d}=VALUE_{rng.randrange(10**9):09d}"
    body_target = max(0, scenario.payload_chars - len(fact) - 2)
    body = _fill_text(f"TOOL_RECORD_{round_index:03d}", body_target, rng)
    # Put the fact at the tail so previews cannot accidentally retain it.
    return f"{body}\n{fact}", fact


def _governance_context(
    state: dict[str, Any],
    runtime: Any,
    *,
    hook: str,
    window_tokens: int,
    messages: list[Any] | None = None,
    tool_call_request: Any = None,
    tool_result: Any = None,
) -> GovernanceContext:
    return GovernanceContext(
        state=state,
        governance=state.get("governance"),
        config={"window": window_tokens},
        token_counter=token_counter,
        runtime=runtime,
        hook=hook,
        messages=messages,
        tool_call_request=tool_call_request,
        tool_result=tool_result,
    )


def _apply_message_patch(
    messages: list[Any], patch: Iterable[Any] | None
) -> list[Any]:
    result = list(messages)
    for item in patch or ():
        if isinstance(item, RemoveMessage):
            if item.id == REMOVE_ALL_MESSAGES:
                result = []
            else:
                result = [message for message in result if message.id != item.id]
            continue
        item_id = getattr(item, "id", None)
        replaced = False
        if item_id is not None:
            for index, message in enumerate(result):
                if getattr(message, "id", None) == item_id:
                    result[index] = item
                    replaced = True
                    break
        if not replaced:
            result.append(item)
    return result


def _apply_result(state: dict[str, Any], result: GovernanceResult) -> None:
    for key, value in (result.state_patch or {}).items():
        if key != "messages":
            state[key] = value
    if result.messages_patch:
        state["messages"] = _apply_message_patch(
            list(state.get("messages") or []), result.messages_patch
        )


def _rendered_context(state: dict[str, Any]) -> str:
    assembler = ContextAssembler()
    governance = state.get("governance")
    return "\n".join((
        assembler.render_context_block(state, governance),
        assembler.render_messages(list(state.get("messages") or [])),
    ))


def _context_tokens(state: dict[str, Any]) -> int:
    return token_counter([HumanMessage(content=_rendered_context(state))])


def _directory_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _read_text_tree(path: Path) -> tuple[str, int]:
    if not path.exists():
        return "", 0
    chunks: list[str] = []
    total = 0
    for item in sorted(path.rglob("*")):
        if not item.is_file():
            continue
        data = item.read_bytes()
        total += len(data)
        chunks.append(data.decode("utf-8", errors="replace"))
    return "\n".join(chunks), total


def _strategy_for(
    variant: Variant,
    run_dir: Path,
    profile: str,
) -> DefaultStrategy | None:
    if variant == "raw":
        return None
    thresholds = None
    if profile == "accelerated":
        thresholds = {
            "p1_externalize": 0.20,
            "p2_thinking": 0.35,
            "p3_observations": 0.45,
            "p4_summarize": 0.55,
            "p5_stop_toolcall": 10.0,
            "hard_stop": 10.0,
        }
    params: dict[str, Any] = {
        "task_memory_enabled": variant == "task_memory",
        "task_memory_dir": str(run_dir / "task-memory"),
        "externalize_dir": str(run_dir / "externalized"),
        "snapshot_dir": str(run_dir / "snapshots"),
        "externalize_min_chars": 500,
        "externalize_preview_chars": 240,
        "preserve_recent": 6,
    }
    if thresholds is not None:
        params["thresholds"] = thresholds
    return DefaultStrategy(
        params=params,
        summarize_model=_ExtractiveSummaryModel(),
    )


def run_one(
    scenario: Scenario,
    variant: Variant,
    *,
    repeat: int,
    seed: int,
    output_dir: Path,
    window_tokens: int,
    profile: str,
) -> RunMetrics:
    run_dir = output_dir / "runs" / scenario.name / variant / f"repeat-{repeat:02d}"
    run_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    thread_id = f"bench-{scenario.name}-{variant}-{repeat}"
    run_id = f"run-{repeat}"
    goal = f"Offline task-memory benchmark: {scenario.name}"
    state: dict[str, Any] = dict(create_initial_thread_state(goal))
    state["research_question"] = goal
    state["messages"] = []
    runtime = SimpleNamespace(
        state=state,
        context={"thread_id": thread_id, "run_id": run_id, "journal": None},
    )
    strategy = _strategy_for(variant, run_dir, profile)
    governance_seconds = 0.0
    model_inputs: list[int] = []
    facts: list[str] = []
    constraints: list[str] = []

    def call(method: str, ctx: GovernanceContext) -> GovernanceResult:
        nonlocal governance_seconds
        if strategy is None:
            return GovernanceResult()
        start = time.perf_counter()
        result = getattr(strategy, method)(ctx)
        governance_seconds += time.perf_counter() - start
        _apply_result(state, result)
        return result

    def model_boundary() -> None:
        call(
            "before_model",
            _governance_context(
                state,
                runtime,
                hook="before_model",
                window_tokens=window_tokens,
                messages=list(state["messages"]),
            ),
        )
        model_inputs.append(_context_tokens(state))

    total_start = time.perf_counter()
    call(
        "before_agent",
        _governance_context(
            state,
            runtime,
            hook="before_agent",
            window_tokens=window_tokens,
            messages=[],
        ),
    )

    for index in range(scenario.rounds):
        constraint = f"CONSTRAINT_{index:03d}=RULE_{rng.randrange(10**8):08d}"
        constraints.append(constraint)
        human = _fill_text(
            f"Round {index}. Preserve {constraint}.", scenario.human_chars, rng
        )
        state["messages"].append(HumanMessage(id=f"human-{index}", content=human))
        model_boundary()

        call_id = f"call-{index}"
        tool_call = {
            "name": "benchmark_fixture",
            "id": call_id,
            "args": {"round": index},
            "type": "tool_call",
        }
        state["messages"].append(AIMessage(
            id=f"ai-call-{index}",
            content=f"Collect deterministic evidence for round {index}.",
            tool_calls=[tool_call],
        ))
        call(
            "after_model",
            _governance_context(
                state,
                runtime,
                hook="after_model",
                window_tokens=window_tokens,
                messages=list(state["messages"]),
            ),
        )

        payload, fact = _round_payload(scenario, index, rng)
        facts.append(fact)
        tool_message = ToolMessage(
            id=f"tool-message-{index}",
            content=payload,
            tool_call_id=call_id,
            name="benchmark_fixture",
        )
        request = SimpleNamespace(
            runtime=runtime,
            state=state,
            tool_call=tool_call,
        )
        result = call(
            "wrap_tool_call",
            _governance_context(
                state,
                runtime,
                hook="wrap_tool_call",
                window_tokens=window_tokens,
                tool_call_request=request,
                tool_result=tool_message,
            ),
        )
        persisted_message = result.request_override or tool_message
        state["messages"].append(persisted_message)
        model_boundary()

        conclusion = _fill_text(
            f"Round {index} evidence recorded.", scenario.assistant_chars, rng
        )
        state["messages"].append(AIMessage(
            id=f"ai-result-{index}", content=conclusion
        ))
        call(
            "after_model",
            _governance_context(
                state,
                runtime,
                hook="after_model",
                window_tokens=window_tokens,
                messages=list(state["messages"]),
            ),
        )

    model_boundary()
    replay_seconds = time.perf_counter() - total_start
    rendered = _rendered_context(state)
    final_tokens = _context_tokens(state)

    if variant == "task_memory":
        evidence_root = run_dir / "task-memory" / "threads" / thread_id / "tasks" / thread_id / "refs"
    elif variant == "legacy":
        evidence_root = run_dir / "externalized"
    else:
        evidence_root = run_dir / "no-evidence"
    evidence_text, evidence_bytes = _read_text_tree(evidence_root)
    recoverable = sum(
        1 for fact in facts if fact in rendered or fact in evidence_text
    )
    current_facts = sum(1 for fact in facts if fact in rendered)
    retained_constraints = sum(1 for item in constraints if item in rendered)

    traceability = 0.0
    graph_version = 0
    integrity_ok = True
    if variant == "task_memory":
        try:
            reader = TaskMemoryReader(run_dir / "task-memory", thread_id, thread_id)
            graph = reader.get_graph()
            graph_version = int(graph["graph_version"])
            tool_events = [
                event for event in reader.writer.store.read_events()
                if event.get("event_type") == "tool_result"
            ]
            for event in tool_events:
                reader.read_ref(event["result_ref"])
            traceability = min(1.0, len(tool_events) / scenario.rounds)
        except Exception:
            integrity_ok = False
    elif variant == "legacy":
        traced = sum(
            1 for message in state["messages"]
            if isinstance(message, ToolMessage)
            and message.additional_kwargs.get(AGENT4ML_EXTERNALIZED)
        )
        traceability = min(1.0, traced / scenario.rounds)

    return RunMetrics(
        scenario=scenario.name,
        description=scenario.description,
        variant=variant,
        repeat=repeat,
        seed=seed,
        profile=profile,
        window_tokens=window_tokens,
        model_input_calls=len(model_inputs),
        cumulative_input_tokens=sum(model_inputs),
        peak_context_tokens=max(model_inputs, default=0),
        final_context_tokens=final_tokens,
        replay_wall_ms=round(replay_seconds * 1_000, 3),
        governance_ms=round(governance_seconds * 1_000, 3),
        storage_bytes=_directory_bytes(run_dir),
        evidence_bytes=evidence_bytes,
        recoverable_fact_rate=round(recoverable / scenario.rounds, 4),
        current_fact_rate=round(current_facts / scenario.rounds, 4),
        constraint_retention_rate=round(retained_constraints / scenario.rounds, 4),
        traceability_rate=round(traceability, 4),
        integrity_ok=integrity_ok,
        graph_version=graph_version,
        tool_results=scenario.rounds,
        final_messages=len(state["messages"]),
    )


def _median(values: Iterable[float | int]) -> float:
    return float(statistics.median(values))


def summarize_runs(runs: Sequence[RunMetrics]) -> dict[str, Any]:
    grouped: dict[tuple[str, Variant], list[RunMetrics]] = {}
    for run in runs:
        grouped.setdefault((run.scenario, run.variant), []).append(run)
    summaries: list[dict[str, Any]] = []
    by_scenario: dict[str, dict[str, dict[str, Any]]] = {}
    for (scenario, variant), items in sorted(grouped.items()):
        row = {
            "scenario": scenario,
            "variant": variant,
            "repeats": len(items),
            "cumulative_input_tokens_median": round(_median(
                item.cumulative_input_tokens for item in items
            )),
            "peak_context_tokens_median": round(_median(
                item.peak_context_tokens for item in items
            )),
            "final_context_tokens_median": round(_median(
                item.final_context_tokens for item in items
            )),
            "replay_wall_ms_median": round(_median(
                item.replay_wall_ms for item in items
            ), 3),
            "governance_ms_median": round(_median(
                item.governance_ms for item in items
            ), 3),
            "storage_bytes_median": round(_median(item.storage_bytes for item in items)),
            "recoverable_fact_rate_median": round(_median(
                item.recoverable_fact_rate for item in items
            ), 4),
            "current_fact_rate_median": round(_median(
                item.current_fact_rate for item in items
            ), 4),
            "constraint_retention_rate_median": round(_median(
                item.constraint_retention_rate for item in items
            ), 4),
            "traceability_rate_median": round(_median(
                item.traceability_rate for item in items
            ), 4),
            "integrity_pass_rate": round(
                sum(1 for item in items if item.integrity_ok) / len(items), 4
            ),
        }
        summaries.append(row)
        by_scenario.setdefault(scenario, {})[variant] = row

    comparisons: list[dict[str, Any]] = []
    for scenario, variants in sorted(by_scenario.items()):
        task = variants.get("task_memory")
        legacy = variants.get("legacy")
        raw = variants.get("raw")
        if task is None or legacy is None:
            continue

        def saving(reference: dict[str, Any] | None, key: str) -> float | None:
            if not reference or not reference[key]:
                return None
            return round(
                100.0 * (reference[key] - task[key]) / reference[key], 2
            )

        comparisons.append({
            "scenario": scenario,
            "task_memory_vs_legacy_input_token_saving_pct": saving(
                legacy, "cumulative_input_tokens_median"
            ),
            "task_memory_vs_raw_input_token_saving_pct": saving(
                raw, "cumulative_input_tokens_median"
            ),
            "task_memory_vs_legacy_governance_time_delta_ms": round(
                task["governance_ms_median"] - legacy["governance_ms_median"], 3
            ),
            "task_memory_vs_legacy_storage_delta_bytes": (
                task["storage_bytes_median"] - legacy["storage_bytes_median"]
            ),
            "task_memory_vs_legacy_recoverable_fact_delta": round(
                task["recoverable_fact_rate_median"]
                - legacy["recoverable_fact_rate_median"],
                4,
            ),
            "task_memory_vs_legacy_current_fact_delta": round(
                task["current_fact_rate_median"]
                - legacy["current_fact_rate_median"],
                4,
            ),
            "task_memory_vs_legacy_constraint_retention_delta": round(
                task["constraint_retention_rate_median"]
                - legacy["constraint_retention_rate_median"],
                4,
            ),
            "task_memory_vs_legacy_traceability_delta": round(
                task["traceability_rate_median"]
                - legacy["traceability_rate_median"],
                4,
            ),
        })
    return {"groups": summaries, "comparisons": comparisons}


def _markdown_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Task Memory A/B Benchmark",
        "",
        "> 本报告是固定输入的离线 replay。`replay_wall_ms` 不是模型响应时间；",
        "> 真正的端到端提速需要接入真实模型 usage/latency 后测量。",
        "",
        "## 分组中位数",
        "",
        "| Scenario | Variant | Input tokens | Peak context | Governance ms | Storage bytes | Recoverable | Current facts | Constraints | Traceable |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["groups"]:
        lines.append(
            f"| {row['scenario']} | {row['variant']} | "
            f"{row['cumulative_input_tokens_median']} | "
            f"{row['peak_context_tokens_median']} | "
            f"{row['governance_ms_median']} | "
            f"{row['storage_bytes_median']} | "
            f"{row['recoverable_fact_rate_median']:.0%} | "
            f"{row['current_fact_rate_median']:.0%} | "
            f"{row['constraint_retention_rate_median']:.0%} | "
            f"{row['traceability_rate_median']:.0%} |"
        )
    lines.extend([
        "",
        "## Task memory 相对收益",
        "",
        "| Scenario | vs legacy token | vs raw token | governance Δms | storage Δbytes | recovery Δ | current facts Δ | constraints Δ | traceability Δ |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in summary["comparisons"]:
        lines.append(
            f"| {row['scenario']} | "
            f"{row['task_memory_vs_legacy_input_token_saving_pct']}% | "
            f"{row['task_memory_vs_raw_input_token_saving_pct']}% | "
            f"{row['task_memory_vs_legacy_governance_time_delta_ms']} | "
            f"{row['task_memory_vs_legacy_storage_delta_bytes']} | "
            f"{row['task_memory_vs_legacy_recoverable_fact_delta']:+.2f} | "
            f"{row['task_memory_vs_legacy_current_fact_delta']:+.2f} | "
            f"{row['task_memory_vs_legacy_constraint_retention_delta']:+.2f} | "
            f"{row['task_memory_vs_legacy_traceability_delta']:+.2f} |"
        )
    lines.extend([
        "",
        "正数 token saving 表示 task memory 更省；负数表示当前实现比对照组使用更多 token。",
        "治理耗时只衡量本地持久化、投影和压缩逻辑，不包含 LLM 与网络时间。",
        "",
    ])
    return "\n".join(lines)


def run_suite(
    *,
    output_dir: Path,
    repeats: int = 3,
    scenario_names: Sequence[str] | None = None,
    variants: Sequence[Variant] = VARIANTS,
    profile: str = "accelerated",
    window_tokens: int | None = None,
    seed: int = 20260930,
) -> tuple[list[RunMetrics], dict[str, Any]]:
    if repeats < 1:
        raise ValueError("repeats must be >= 1")
    scenarios = _scenario_map()
    selected_names = list(scenario_names or scenarios)
    unknown = sorted(set(selected_names).difference(scenarios))
    if unknown:
        raise ValueError(f"unknown scenarios: {unknown}")
    if profile not in {"accelerated", "production"}:
        raise ValueError("profile must be accelerated or production")
    effective_window = window_tokens or (
        8_000 if profile == "accelerated" else 128_000
    )
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    runs: list[RunMetrics] = []
    stable_scenario_order = list(scenarios)
    for scenario_name in selected_names:
        scenario = scenarios[scenario_name]
        scenario_index = stable_scenario_order.index(scenario_name)
        for repeat in range(1, repeats + 1):
            for variant in variants:
                # All variants in one scenario/repeat receive identical content.
                run_seed = seed + repeat * 10_000 + scenario_index
                runs.append(run_one(
                    scenario,
                    variant,
                    repeat=repeat,
                    seed=run_seed,
                    output_dir=output_dir,
                    window_tokens=effective_window,
                    profile=profile,
                ))

    runs_path = output_dir / "runs.jsonl"
    runs_path.write_text(
        "".join(json.dumps(asdict(run), ensure_ascii=False) + "\n" for run in runs),
        encoding="utf-8",
    )
    summary = summarize_runs(runs)
    summary.update({
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "profile": profile,
        "window_tokens": effective_window,
        "repeats": repeats,
        "scenarios": selected_names,
        "variants": list(variants),
    })
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "summary.md").write_text(
        _markdown_report(summary), encoding="utf-8"
    )
    return runs, summary


def _default_output_dir() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return Path(".agent4ml") / "benchmarks" / f"task-memory-ab-{timestamp}"


def _print_summary(summary: dict[str, Any], output_dir: Path) -> None:
    print("\nTask Memory A/B benchmark complete")
    print(f"profile={summary['profile']} repeats={summary['repeats']} window={summary['window_tokens']}")
    for row in summary["comparisons"]:
        print(
            f"- {row['scenario']}: "
            f"vs legacy tokens={row['task_memory_vs_legacy_input_token_saving_pct']}% "
            f"vs raw tokens={row['task_memory_vs_raw_input_token_saving_pct']}% "
            f"governance_delta={row['task_memory_vs_legacy_governance_time_delta_ms']}ms "
            f"traceability_delta={row['task_memory_vs_legacy_traceability_delta']:+.2f}"
        )
    print(f"report: {output_dir / 'summary.md'}")
    print(f"raw data: {output_dir / 'runs.jsonl'}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run deterministic raw/legacy/task-memory A/B benchmarks."
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--scenario",
        action="append",
        choices=[scenario.name for scenario in SCENARIOS],
        help="Scenario to run; repeat the option to select multiple (default: all).",
    )
    parser.add_argument(
        "--profile",
        choices=("accelerated", "production"),
        default="accelerated",
        help="Accelerated lowers governance thresholds; production keeps defaults.",
    )
    parser.add_argument("--window", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260930)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    output_dir = (args.output or _default_output_dir()).resolve()
    _runs, summary = run_suite(
        output_dir=output_dir,
        repeats=args.repeats,
        scenario_names=args.scenario,
        profile=args.profile,
        window_tokens=args.window,
        seed=args.seed,
    )
    _print_summary(summary, output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
