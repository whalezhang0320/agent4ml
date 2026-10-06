from __future__ import annotations

import json
from pathlib import Path

from agent4ml.backend.benchmarks.task_memory_ab import run_suite


def test_task_memory_benchmark_writes_reproducible_reports(tmp_path: Path) -> None:
    runs, summary = run_suite(
        output_dir=tmp_path,
        repeats=1,
        scenario_names=["short_control", "needle_recovery"],
        profile="accelerated",
        window_tokens=4_000,
        seed=123,
    )
    assert len(runs) == 6
    assert {run.variant for run in runs} == {"raw", "legacy", "task_memory"}
    assert len(summary["comparisons"]) == 2
    assert (tmp_path / "summary.md").is_file()
    assert (tmp_path / "summary.json").is_file()
    lines = (tmp_path / "runs.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 6
    assert all(json.loads(line)["integrity_ok"] for line in lines)

    task_runs = [run for run in runs if run.variant == "task_memory"]
    assert all(run.traceability_rate == 1.0 for run in task_runs)
    assert all(run.recoverable_fact_rate == 1.0 for run in task_runs)


def test_task_memory_benchmark_same_seed_has_same_fixture_results(
    tmp_path: Path,
) -> None:
    first, _summary = run_suite(
        output_dir=tmp_path / "one",
        repeats=1,
        scenario_names=["short_control"],
        seed=456,
    )
    second, _summary = run_suite(
        output_dir=tmp_path / "two",
        repeats=1,
        scenario_names=["short_control"],
        seed=456,
    )
    comparable = (
        "cumulative_input_tokens",
        "peak_context_tokens",
        "final_context_tokens",
        "recoverable_fact_rate",
        "constraint_retention_rate",
        "traceability_rate",
    )
    for left, right in zip(first, second):
        assert left.variant == right.variant
        assert all(getattr(left, key) == getattr(right, key) for key in comparable)
