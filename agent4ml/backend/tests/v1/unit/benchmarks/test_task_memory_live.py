from __future__ import annotations

from dataclasses import replace

from agent4ml.backend.benchmarks.task_memory_live import (
    CASE_SPECS,
    VARIANTS,
    LiveRunMetrics,
    _score_turn,
    build_fixtures,
    summarize_runs,
    validate_suite,
)


def _run(case: str, variant: str) -> LiveRunMetrics:
    return LiveRunMetrics(
        case=case,
        description=case,
        variant=variant,  # type: ignore[arg-type]
        requested_model="same-model",
        resolved_models=("same-resolved-model",),
        temperature=0.0,
        max_output_tokens=256,
        window_tokens=24_000,
        system_prompt_sha256="system-hash",
        user_prompts_sha256=f"prompts-{case}",
        rounds_completed=16,
        answer_calls=16,
        internal_summary_calls=1 if variant == "legacy_summary" else 0,
        total_model_calls=17 if variant == "legacy_summary" else 16,
        answer_input_tokens=1_600,
        answer_output_tokens=160,
        total_input_tokens={
            "full_history": 2_000,
            "legacy_summary": 1_500,
            "task_memory": 1_000,
        }[variant],
        total_output_tokens=160,
        peak_answer_input_tokens=2_000,
        answer_latency_ms=1_600,
        total_model_latency_ms={
            "full_history": 2_000,
            "legacy_summary": 1_500,
            "task_memory": 1_000,
        }[variant],
        run_wall_ms=2_000,
        governance_ms=20,
        final_context_tokens=1_000,
        storage_bytes=100,
        quality_checks_passed=32,
        quality_checks_total=32,
        quality_score=1.0,
        nonempty_response_rate=1.0,
        summarization_count=1 if variant == "legacy_summary" else 0,
        task_memory_graph_version=32 if variant == "task_memory" else 0,
        task_memory_tool_events=32 if variant == "task_memory" else 0,
        task_memory_integrity_ok=True,
        attempt_dir="attempt-01",
    )


def test_fixtures_have_exactly_five_cases_and_sixteen_turns() -> None:
    fixtures = build_fixtures()
    assert len(fixtures) == 5
    assert set(fixtures) == {spec.name for spec in CASE_SPECS}
    assert all(len(turns) == 16 for turns in fixtures.values())
    assert all(
        turn.current_marker not in turn.user_prompt
        for turns in fixtures.values()
        for turn in turns
    )


def test_score_turn_checks_current_and_carry_markers() -> None:
    turns = build_fixtures()[CASE_SPECS[0].name]
    early = turns[0]
    assert _score_turn(
        early, f"CURRENT={early.current_marker}\nCARRY=NONE\nNOTE=ok"
    ) == (True, True, 2, 2)
    later = turns[10]
    assert later.carry_marker is not None
    assert _score_turn(
        later,
        f"CURRENT={later.current_marker}\nCARRY={later.carry_marker}\nNOTE=ok",
    ) == (True, True, 2, 2)


def test_summary_and_validation_cover_fairness_contract() -> None:
    runs = [
        _run(spec.name, variant)
        for spec in CASE_SPECS
        for variant in VARIANTS
    ]
    summary = summarize_runs(runs)
    assert summary["overall"]["task_vs_full_history_input_saving_pct"] == 50.0
    assert summary["overall"]["task_vs_legacy_summary_input_saving_pct"] == 33.33
    validation = validate_suite(runs, window_tokens=24_000)
    assert validation["passed"] is True
    assert validation["answer_calls"] == 240


def test_validation_rejects_a_run_outside_window() -> None:
    runs = [
        _run(spec.name, variant)
        for spec in CASE_SPECS
        for variant in VARIANTS
    ]
    runs[0] = replace(runs[0], peak_answer_input_tokens=24_001)
    validation = validate_suite(runs, window_tokens=24_000)
    assert validation["passed"] is False
    assert validation["checks"]["within_window"] is False
