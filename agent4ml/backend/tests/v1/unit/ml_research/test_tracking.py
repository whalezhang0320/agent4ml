from __future__ import annotations

import json

from agent4ml.backend.agents.ml_research.tracking import LocalExperimentTracker, main


def test_tracker_records_metric_marker_and_snapshot(tmp_path):
    tracker = LocalExperimentTracker(tmp_path / "run-001")
    manifest = tracker.initialize(
        name="baseline",
        command=["python", "train.py"],
        cwd=tmp_path,
        metadata={"seed": 42},
    )

    parsed = tracker.record_output_line(
        'AGENT4ML_METRIC {"step": 7, "split": "train", "metrics": {"loss": 0.25}}\n'
    )
    snapshot = tracker.snapshot(tail_lines=5)

    assert parsed is True
    assert manifest["status"] == "planned"
    assert snapshot["metric_records"] == 1
    assert snapshot["latest_metrics"]["loss"]["value"] == 0.25
    assert snapshot["latest_metrics"]["loss"]["step"] == 7
    assert snapshot["log_tail"][0].startswith("AGENT4ML_METRIC")


def test_tracker_ignores_malformed_metric_but_preserves_log(tmp_path):
    tracker = LocalExperimentTracker(tmp_path / "run-002")
    tracker.initialize(name="broken")

    assert tracker.record_output_line("AGENT4ML_METRIC not-json") is False
    events = [
        json.loads(line)
        for line in tracker.events_path.read_text(encoding="utf-8").splitlines()
    ]
    assert events[-1]["type"] == "metric.parse_failed"
    assert "not-json" in tracker.stdout_path.read_text(encoding="utf-8")


def test_wait_keeps_failed_snapshot_visible(tmp_path, capsys):
    tracker = LocalExperimentTracker(tmp_path / "run-failed")
    tracker.initialize(name="failed")
    tracker.update_manifest(status="failed", exit_code=3)

    assert main(["wait", str(tracker.run_dir), "--timeout", "0"]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["manifest"]["status"] == "failed"
    assert output["manifest"]["exit_code"] == 3
