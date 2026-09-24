"""Dependency-free local experiment tracking for long-running ML jobs.

The module intentionally uses JSON/JSONL and plain log files so experiments remain
readable without a service such as Weights & Biases. It can also launch a detached
local process and expose non-blocking status snapshots to the agent.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

SCHEMA_VERSION = 1
METRIC_PREFIX = "AGENT4ML_METRIC "
TERMINAL_STATUSES = {"success", "failed", "cancelled"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _read_json(path: Path, default: dict[str, Any]) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return dict(default)
    return value if isinstance(value, dict) else dict(default)


def _tail(path: Path, lines: int) -> list[str]:
    if lines <= 0:
        return []
    try:
        return path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:]
    except OSError:
        return []


class LocalExperimentTracker:
    """Persist one experiment as a manifest, events, metrics, and stdout log."""

    def __init__(self, run_dir: str | Path) -> None:
        self.run_dir = Path(run_dir).expanduser().resolve()
        self.manifest_path = self.run_dir / "manifest.json"
        self.events_path = self.run_dir / "events.jsonl"
        self.metrics_path = self.run_dir / "metrics.jsonl"
        self.stdout_path = self.run_dir / "stdout.log"

    def initialize(
        self,
        *,
        name: str,
        command: Sequence[str] = (),
        cwd: str | Path | None = None,
        metadata: dict[str, Any] | None = None,
        run_id: str | None = None,
    ) -> dict[str, Any]:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "run_id": run_id or f"exp-{uuid.uuid4().hex[:12]}",
            "name": name,
            "status": "planned",
            "command": list(command),
            "cwd": str(Path(cwd).expanduser().resolve()) if cwd else str(Path.cwd()),
            "created_at": _now(),
            "started_at": None,
            "finished_at": None,
            "pid": None,
            "worker_pid": None,
            "exit_code": None,
            "metadata": metadata or {},
        }
        _write_json_atomic(self.manifest_path, manifest)
        self.log_event("experiment.initialized", data={"name": name})
        return manifest

    def update_manifest(self, **changes: Any) -> dict[str, Any]:
        manifest = _read_json(self.manifest_path, {})
        if not manifest:
            raise FileNotFoundError(f"experiment is not initialized: {self.run_dir}")
        manifest.update(changes)
        _write_json_atomic(self.manifest_path, manifest)
        return manifest

    def log_event(
        self,
        event_type: str,
        *,
        message: str = "",
        data: dict[str, Any] | None = None,
    ) -> None:
        _append_jsonl(
            self.events_path,
            {"timestamp": _now(), "type": event_type, "message": message, "data": data or {}},
        )

    def log_metrics(
        self,
        metrics: dict[str, int | float],
        *,
        step: int | float | None = None,
        epoch: int | float | None = None,
        split: str | None = None,
    ) -> None:
        if not metrics:
            return
        normalized = {
            str(key): float(value)
            for key, value in metrics.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
        if not normalized:
            return
        _append_jsonl(
            self.metrics_path,
            {
                "timestamp": _now(),
                "step": step,
                "epoch": epoch,
                "split": split,
                "metrics": normalized,
            },
        )

    def record_output_line(self, line: str) -> bool:
        """Append raw output and parse an optional ``AGENT4ML_METRIC`` JSON marker."""
        self.run_dir.mkdir(parents=True, exist_ok=True)
        with self.stdout_path.open("a", encoding="utf-8") as handle:
            handle.write(line if line.endswith("\n") else line + "\n")
            handle.flush()
        stripped = line.strip()
        if not stripped.startswith(METRIC_PREFIX):
            return False
        try:
            payload = json.loads(stripped[len(METRIC_PREFIX):])
        except json.JSONDecodeError:
            self.log_event("metric.parse_failed", message=stripped[:500])
            return False
        if not isinstance(payload, dict):
            return False
        metrics = payload.get("metrics")
        if not isinstance(metrics, dict):
            reserved = {"step", "epoch", "split", "timestamp"}
            metrics = {key: value for key, value in payload.items() if key not in reserved}
        self.log_metrics(
            metrics,
            step=payload.get("step"),
            epoch=payload.get("epoch"),
            split=payload.get("split"),
        )
        return True

    def snapshot(self, *, tail_lines: int = 20) -> dict[str, Any]:
        manifest = _read_json(self.manifest_path, {})
        latest: dict[str, dict[str, Any]] = {}
        metric_count = 0
        try:
            metric_lines = self.metrics_path.read_text(encoding="utf-8").splitlines()
        except OSError:
            metric_lines = []
        for line in metric_lines:
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(item, dict) or not isinstance(item.get("metrics"), dict):
                continue
            metric_count += 1
            for key, value in item["metrics"].items():
                latest[key] = {
                    "value": value,
                    "step": item.get("step"),
                    "epoch": item.get("epoch"),
                    "split": item.get("split"),
                    "timestamp": item.get("timestamp"),
                }
        return {
            "manifest": manifest,
            "metric_records": metric_count,
            "latest_metrics": latest,
            "log_tail": _tail(self.stdout_path, tail_lines),
        }


def _worker(run_dir: Path, cwd: Path, command: list[str]) -> int:
    tracker = LocalExperimentTracker(run_dir)
    tracker.update_manifest(status="running", started_at=_now(), worker_pid=os.getpid())
    tracker.log_event("experiment.started", data={"command": command})
    exit_code = 1
    try:
        process = subprocess.Popen(
            command,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        tracker.update_manifest(pid=process.pid)
        assert process.stdout is not None
        for line in process.stdout:
            tracker.record_output_line(line)
        exit_code = process.wait()
    except Exception as exc:
        tracker.record_output_line(f"[agent4ml tracker] launch failed: {exc}")
        tracker.log_event("experiment.launch_failed", message=str(exc))
    status = "success" if exit_code == 0 else "failed"
    tracker.update_manifest(status=status, exit_code=exit_code, finished_at=_now())
    tracker.log_event("experiment.finished", data={"status": status, "exit_code": exit_code})
    return exit_code


def _parse_json_object(raw: str, option: str) -> dict[str, Any]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"{option} must be a JSON object: {exc}") from exc
    if not isinstance(value, dict):
        raise SystemExit(f"{option} must be a JSON object")
    return value


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Agent4ML local ML experiment tracker")
    sub = parser.add_subparsers(dest="action", required=True)

    start = sub.add_parser("start", help="launch a detached local experiment")
    start.add_argument("run_dir")
    start.add_argument("--name", required=True)
    start.add_argument("--cwd", default=".")
    start.add_argument("--metadata-json", default="{}")

    status = sub.add_parser("status", help="print a non-blocking JSON snapshot")
    status.add_argument("run_dir")
    status.add_argument("--tail", type=int, default=20)

    wait = sub.add_parser("wait", help="wait up to a bounded timeout")
    wait.add_argument("run_dir")
    wait.add_argument("--timeout", type=float, default=60.0)
    wait.add_argument("--interval", type=float, default=2.0)
    wait.add_argument("--tail", type=int, default=20)

    worker = sub.add_parser("_worker")
    worker.add_argument("run_dir")
    worker.add_argument("--cwd", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    raw_args = list(argv) if argv is not None else sys.argv[1:]
    command: list[str] = []
    if "--" in raw_args:
        separator = raw_args.index("--")
        command = raw_args[separator + 1:]
        raw_args = raw_args[:separator]
    args = _build_parser().parse_args(raw_args)

    if args.action == "start":
        if not command:
            raise SystemExit("start requires a command after --")
        run_dir = Path(args.run_dir).expanduser().resolve()
        cwd = Path(args.cwd).expanduser().resolve()
        metadata = _parse_json_object(args.metadata_json, "--metadata-json")
        tracker = LocalExperimentTracker(run_dir)
        tracker.initialize(name=args.name, command=command, cwd=cwd, metadata=metadata)
        tracker.update_manifest(status="starting")
        worker_args = [
            sys.executable,
            "-m",
            "agent4ml.backend.agents.ml_research.tracking",
            "_worker",
            str(run_dir),
            "--cwd",
            str(cwd),
            "--",
            *command,
        ]
        process = subprocess.Popen(
            worker_args,
            cwd=str(cwd),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        tracker.update_manifest(worker_pid=process.pid)
        print(json.dumps({"run_dir": str(run_dir), "worker_pid": process.pid}))
        return 0

    if args.action == "_worker":
        if not command:
            return 2
        return _worker(Path(args.run_dir), Path(args.cwd), command)

    tracker = LocalExperimentTracker(args.run_dir)
    if args.action == "status":
        print(json.dumps(tracker.snapshot(tail_lines=args.tail), ensure_ascii=False, indent=2))
        return 0

    deadline = time.monotonic() + max(0.0, args.timeout)
    while True:
        snapshot = tracker.snapshot(tail_lines=args.tail)
        status = snapshot.get("manifest", {}).get("status")
        if status in TERMINAL_STATUSES or time.monotonic() >= deadline:
            print(json.dumps(snapshot, ensure_ascii=False, indent=2))
            # This is an observation command: keep stdout visible to sandbox tools
            # even when the experiment failed or the bounded wait timed out.
            return 0
        time.sleep(max(0.1, args.interval))


if __name__ == "__main__":
    raise SystemExit(main())
