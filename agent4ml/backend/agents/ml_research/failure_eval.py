"""Deterministic failure classification and evaluation for local ML jobs."""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Iterable, Sequence


class FailureClass(StrEnum):
    CUDA_OOM = "cuda_oom"
    DEPENDENCY_CONFLICT = "dependency_conflict"
    NETWORK_TIMEOUT = "network_timeout"
    WORKER_LOST = "worker_lost"
    CONFIG_MISSING = "config_missing"
    SYNTAX_ERROR = "syntax_error"
    UNKNOWN = "unknown"


class RetryDecision(StrEnum):
    RETRY_AUTOMATICALLY = "retry_automatically"
    REQUIRE_APPROVAL = "require_approval"
    FAIL_PERMANENTLY = "fail_permanently"


@dataclass(frozen=True)
class FailureDetection:
    failure_class: FailureClass
    evidence: str
    pattern: str


_CUDA_OOM_PATTERNS = (
    re.compile(r"cuda\s+out\s+of\s+memory", re.IGNORECASE),
    re.compile(r"torch\.cuda\.OutOfMemoryError", re.IGNORECASE),
    re.compile(r"CUDNN_STATUS_ALLOC_FAILED", re.IGNORECASE),
    re.compile(r"CUBLAS_STATUS_ALLOC_FAILED", re.IGNORECASE),
    re.compile(r"HIP\s+out\s+of\s+memory", re.IGNORECASE),
)

_DEPENDENCY_CONFLICT_PATTERNS = (
    re.compile(r"ResolutionImpossible", re.IGNORECASE),
    re.compile(r"conflicting dependencies", re.IGNORECASE),
    re.compile(r"(?:version|dependency) conflict", re.IGNORECASE),
    re.compile(r"ContextualVersionConflict", re.IGNORECASE),
    re.compile(r"has requirement .+?, but you have", re.IGNORECASE),
    re.compile(r"requires .+?, but .+? is installed", re.IGNORECASE),
)

_NETWORK_TIMEOUT_PATTERNS = (
    re.compile(r"(?:connection|read|connect)\s+timed?\s*out", re.IGNORECASE),
    re.compile(r"TimeoutError", re.IGNORECASE),
    re.compile(r"temporary failure in name resolution", re.IGNORECASE),
    re.compile(r"connection reset by peer", re.IGNORECASE),
)

_CONFIG_MISSING_PATTERNS = (
    re.compile(r"(?:config|configuration).*(?:missing|not found)", re.IGNORECASE),
    re.compile(r"No such file or directory: .*(?:ya?ml|json|toml)", re.IGNORECASE),
)

_SYNTAX_ERROR_PATTERNS = (re.compile(r"(?:SyntaxError|IndentationError):", re.IGNORECASE),)

_PATTERNS: tuple[tuple[FailureClass, tuple[re.Pattern[str], ...]], ...] = (
    (FailureClass.CUDA_OOM, _CUDA_OOM_PATTERNS),
    (FailureClass.DEPENDENCY_CONFLICT, _DEPENDENCY_CONFLICT_PATTERNS),
    (FailureClass.NETWORK_TIMEOUT, _NETWORK_TIMEOUT_PATTERNS),
    (FailureClass.CONFIG_MISSING, _CONFIG_MISSING_PATTERNS),
    (FailureClass.SYNTAX_ERROR, _SYNTAX_ERROR_PATTERNS),
)


def retry_decision(
    failure_class: FailureClass | str,
    *,
    node_idempotent: bool = True,
) -> RetryDecision:
    """Map deterministic failure evidence to a workflow retry decision."""
    try:
        value = FailureClass(failure_class)
    except ValueError:
        value = FailureClass.UNKNOWN
    if value is FailureClass.NETWORK_TIMEOUT:
        return RetryDecision.RETRY_AUTOMATICALLY
    if value is FailureClass.WORKER_LOST:
        return (
            RetryDecision.RETRY_AUTOMATICALLY
            if node_idempotent
            else RetryDecision.REQUIRE_APPROVAL
        )
    if value in {FailureClass.CUDA_OOM, FailureClass.DEPENDENCY_CONFLICT}:
        return RetryDecision.REQUIRE_APPROVAL
    return RetryDecision.FAIL_PERMANENTLY


class FailureClassifier:
    """Incrementally classify known ML failure evidence without guessing."""

    def __init__(self) -> None:
        self._detections: list[FailureDetection] = []
        self._seen: set[FailureClass] = set()

    def feed(self, line: str) -> FailureDetection | None:
        for failure_class, patterns in _PATTERNS:
            if failure_class in self._seen:
                continue
            for pattern in patterns:
                if pattern.search(line):
                    detection = FailureDetection(
                        failure_class=failure_class,
                        evidence=line.strip()[:1000],
                        pattern=pattern.pattern,
                    )
                    self._seen.add(failure_class)
                    self._detections.append(detection)
                    return detection
        return None

    @property
    def detections(self) -> tuple[FailureDetection, ...]:
        return tuple(self._detections)

    @property
    def primary(self) -> FailureDetection | None:
        return self._detections[0] if self._detections else None


def classify_lines(lines: Iterable[str]) -> FailureDetection | None:
    classifier = FailureClassifier()
    for line in lines:
        classifier.feed(line)
    return classifier.primary


@dataclass(frozen=True)
class FailureEvalCase:
    name: str
    lines: tuple[str, ...]
    expected: FailureClass


@dataclass(frozen=True)
class FailureEvalResult:
    name: str
    expected: str
    actual: str
    passed: bool
    evidence: str | None


BUILTIN_EVAL_CASES = (
    FailureEvalCase(
        "pytorch-cuda-oom",
        (
            "step=128 loss=3.4",
            "torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB",
        ),
        FailureClass.CUDA_OOM,
    ),
    FailureEvalCase(
        "cudnn-allocation-failure",
        ("RuntimeError: cuDNN error: CUDNN_STATUS_ALLOC_FAILED",),
        FailureClass.CUDA_OOM,
    ),
    FailureEvalCase(
        "pip-resolution-conflict",
        (
            "ERROR: Cannot install torch==2.4 and xformers==0.0.20 because these package versions have conflicting dependencies.",
            "ERROR: ResolutionImpossible",
        ),
        FailureClass.DEPENDENCY_CONFLICT,
    ),
    FailureEvalCase(
        "pip-check-conflict",
        ("xformers 0.0.20 has requirement torch<2.1, but you have torch 2.4.0.",),
        FailureClass.DEPENDENCY_CONFLICT,
    ),
    FailureEvalCase(
        "ordinary-training-failure-is-not-misclassified",
        ("ValueError: target shape does not match input shape",),
        FailureClass.UNKNOWN,
    ),
    FailureEvalCase(
        "network-timeout",
        ("TimeoutError: connection timed out while pulling image",),
        FailureClass.NETWORK_TIMEOUT,
    ),
    FailureEvalCase(
        "missing-config",
        ("training configuration not found: config.yaml",),
        FailureClass.CONFIG_MISSING,
    ),
    FailureEvalCase(
        "syntax-error",
        ("SyntaxError: invalid syntax",),
        FailureClass.SYNTAX_ERROR,
    ),
)


def run_failure_eval(
    cases: Sequence[FailureEvalCase] = BUILTIN_EVAL_CASES,
) -> list[FailureEvalResult]:
    results: list[FailureEvalResult] = []
    for case in cases:
        detection = classify_lines(case.lines)
        actual = detection.failure_class if detection else FailureClass.UNKNOWN
        results.append(
            FailureEvalResult(
                name=case.name,
                expected=case.expected.value,
                actual=actual.value,
                passed=actual is case.expected,
                evidence=detection.evidence if detection else None,
            )
        )
    return results


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Agent4ML ML failure classification evals")
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = parser.parse_args(argv)
    results = run_failure_eval()
    passed = sum(result.passed for result in results)
    if args.json:
        print(json.dumps([asdict(result) for result in results], ensure_ascii=False, indent=2))
    else:
        for result in results:
            marker = "PASS" if result.passed else "FAIL"
            print(f"[{marker}] {result.name}: expected={result.expected} actual={result.actual}")
        print(f"{passed}/{len(results)} passed")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
