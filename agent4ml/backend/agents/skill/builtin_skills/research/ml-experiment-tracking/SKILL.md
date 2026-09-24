---
name: ml-experiment-tracking
description: "Track local machine-learning experiments with live stdout capture, structured JSONL metrics, non-blocking status checks, failure evidence, and resumable run records."
allowed-tools:
  - bash
  - read_file
  - write_file
  - list_dir
license: MIT
---

# ML Experiment Tracking

Track training locally without requiring a hosted service. The canonical record is
the run directory: `manifest.json`, `events.jsonl`, `metrics.jsonl`, and
`stdout.log`. Existing TensorBoard, CSV, JSONL, or framework logs may remain the
source of truth; add an adapter instead of rewriting working training code.

## Launch without blocking the Agent

Use Agent4ML's local tracker from the same Python environment that runs Agent4ML:

```bash
python -m agent4ml.backend.agents.ml_research.tracking start \
  /absolute/path/to/runs/run-001 \
  --name paper-baseline \
  --cwd /absolute/path/to/source \
  --metadata-json '{"seed": 42, "config": "configs/baseline.yaml"}' \
  -- python train.py --config /absolute/path/to/configs/baseline.yaml
```

This returns immediately while a local worker writes the logs. Do not use a single
blocking `bash` call for a multi-hour training job.

## Emit structured metrics

When code can be edited, print one line per reporting interval:

```python
import json
print("AGENT4ML_METRIC " + json.dumps({
    "step": step,
    "epoch": epoch,
    "split": "train",
    "metrics": {"loss": float(loss), "lr": float(lr)},
}), flush=True)
```

Emit validation metrics with `split: "validation"`. Keep metric names stable and
include the primary paper metric. Do not log every batch when that would dominate
runtime or disk usage.

If code cannot be edited, preserve its raw log and write a small parser that
appends the same JSONL schema. Record the parser version and unmatched-line count.

## Observe and diagnose

Take non-blocking snapshots:

```bash
python -m agent4ml.backend.agents.ml_research.tracking status \
  /absolute/path/to/runs/run-001 --tail 30
```

For a bounded wait (never an indefinite wait):

```bash
python -m agent4ml.backend.agents.ml_research.tracking wait \
  /absolute/path/to/runs/run-001 --timeout 60 --tail 30
```

During monitoring, compare step timestamps and metrics rather than treating new
log text as proof of progress. Check for:

- process exit and non-zero exit code;
- no new steps for the expected interval;
- NaN/Inf, loss divergence, zero throughput, data-loader errors;
- CUDA OOM, device mismatch, disk-full, corrupt checkpoint, and interrupted runs;
- validation regression or train/validation divergence.

Record diagnoses and interventions in `events.jsonl` or the reproduction report.
Never overwrite a failed run; resume into a new run id or record the checkpoint and
parent run explicitly.

## What “real time” means locally

The worker captures stdout and metric markers as they are emitted. Agent4ML reads
periodic snapshots between reasoning steps; the current TUI does not stream every
training line into the conversation. Prefer concise milestone updates and notify
the user immediately on completion, failure, or required action.
