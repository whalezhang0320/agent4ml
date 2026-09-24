from __future__ import annotations

from agent4ml.backend.agents.sandbox.utils.search import (
    DEFAULT_LINE_SUMMARY_LENGTH,
    DEFAULT_MAX_FILE_SIZE_BYTES,
    IGNORE_PATTERNS,
)


class TestIgnorePatterns:
    def test_contains_git(self) -> None:
        assert ".git" in IGNORE_PATTERNS

    def test_contains_node_modules(self) -> None:
        assert "node_modules" in IGNORE_PATTERNS

    def test_contains_pycache(self) -> None:
        assert "__pycache__" in IGNORE_PATTERNS

    def test_contains_venv(self) -> None:
        assert ".venv" in IGNORE_PATTERNS
        assert "venv" in IGNORE_PATTERNS

    def test_contains_glob_patterns(self) -> None:
        assert "*.egg-info" in IGNORE_PATTERNS
        assert "*.log" in IGNORE_PATTERNS

    def test_contains_cache_dirs(self) -> None:
        assert ".pytest_cache" in IGNORE_PATTERNS
        assert ".mypy_cache" in IGNORE_PATTERNS


class TestConstants:
    def test_max_file_size(self) -> None:
        assert DEFAULT_MAX_FILE_SIZE_BYTES == 1_000_000

    def test_line_summary_length(self) -> None:
        assert DEFAULT_LINE_SUMMARY_LENGTH == 200
