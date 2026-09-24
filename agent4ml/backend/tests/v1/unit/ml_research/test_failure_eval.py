from __future__ import annotations

from agent4ml.backend.agents.ml_research.failure_eval import (
    BUILTIN_EVAL_CASES,
    FailureClass,
    FailureClassifier,
    classify_lines,
    run_failure_eval,
)


def test_builtin_failure_eval_suite_passes():
    results = run_failure_eval()

    assert len(results) == len(BUILTIN_EVAL_CASES)
    assert all(result.passed for result in results)


def test_classifier_emits_each_failure_class_only_once():
    classifier = FailureClassifier()

    first = classifier.feed("CUDA out of memory while allocating 2 GiB")
    duplicate = classifier.feed("torch.cuda.OutOfMemoryError: CUDA out of memory")

    assert first is not None
    assert first.failure_class is FailureClass.CUDA_OOM
    assert duplicate is None
    assert len(classifier.detections) == 1


def test_generic_import_error_is_not_called_dependency_conflict():
    detection = classify_lines(["ModuleNotFoundError: No module named 'optional_pkg'"])

    assert detection is None
