---
name: ml-result-analysis
description: "Analyze and compare completed machine-learning experiments against paper claims using aligned protocols, multi-seed statistics, curves, failure evidence, and calibrated reproduction verdicts."
allowed-tools:
  - bash
  - read_file
  - write_file
  - list_dir
  - present_files
license: MIT
---

# ML Result Analysis

Compare finished experiment runs with each other and with the paper. A comparison
is valid only when metric definitions, dataset version, split, preprocessing,
evaluation code, checkpoint selection, and compute budget are aligned.

## Evidence intake

Read each run's `manifest.json`, immutable config, `metrics.jsonl`, `events.jsonl`,
and relevant log tail. Classify runs as completed, failed, partial, resumed, or
invalid. Do not drop failed seeds without reporting them.

Build a comparison matrix containing at least:

- code revision and patch;
- environment and hardware;
- dataset identifier/fingerprint and split;
- seed and effective batch size;
- training steps/epochs and wall-clock time;
- checkpoint-selection rule;
- primary and secondary metrics;
- deviations from the paper.

## Statistical analysis

- For three or more comparable seeds, report count, mean, sample standard
  deviation, median, min/max, and a 95% confidence interval. For fewer seeds,
  report individual values and state that uncertainty is not estimable reliably.
- Preserve metric direction (`higher is better` or `lower is better`) and units.
- Compare against the paper's exact reported value and tolerance. Report absolute
  and relative gaps, but avoid relative percentages when the denominator is zero
  or semantically meaningless.
- Inspect learning curves, not only best checkpoints. Identify instability,
  overfitting, premature stopping, and suspicious discontinuities.
- Separate implementation failure, optimization failure, evaluation mismatch,
  insufficient compute, and evidence that challenges the original claim.

## Required report

Write `analysis/result-analysis.md` with:

1. verdict: exact, close, scaled proxy, partial, not reproduced, or inconclusive;
2. scope and protocol alignment;
3. paper-versus-local headline table;
4. per-run and aggregate statistics;
5. curves or plots with readable labels and run ids;
6. deviations, failures, and sensitivity/ablation findings;
7. likely explanations ranked by evidence strength;
8. concrete next experiments, ordered by information gain and local cost.

Every headline number must trace to a run id and metric record. Never label the
paper's number as locally reproduced, and never treat a scaled proxy as a faithful
reproduction. Present the report and generated tables/plots as artifacts.
