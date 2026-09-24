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
    UNKNOWN = "unknown"


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

_PATTERNS: tuple[tuple[FailureClass, tuple[re.Pattern[str], ...]], ...] = (
    (FailureClass.CUDA_OOM, _CUDA_OOM_PATTERNS),
    (FailureClass.DEPENDENCY_CONFLICT, _DEPENDENCY_CONFLICT_PATTERNS),
)


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
