from __future__ import annotations

import fnmatch

IGNORE_PATTERNS: list[str] = [
    ".git", ".svn", ".hg", ".bzr",
    "node_modules", "__pycache__", ".venv", "venv", ".env", "env",
    ".tox", ".nox", ".eggs", "*.egg-info", "site-packages",
    "dist", "build", ".next", ".nuxt", ".output", ".turbo", "target", "out",
    ".idea", ".vscode", "*.swp", "*.swo", "*~",
    ".DS_Store", "Thumbs.db", "desktop.ini", "*.lnk",
    "*.log", "*.tmp", "*.temp", "*.bak", "*.cache",
    ".coverage", "coverage", ".nyc_output", "htmlcov",
    ".pytest_cache", ".mypy_cache", ".ruff_cache",
]

DEFAULT_MAX_FILE_SIZE_BYTES = 1_000_000
DEFAULT_LINE_SUMMARY_LENGTH = 200


def should_ignore_path(path: str) -> bool:
    """检查路径的任一段是否匹配 IGNORE_PATTERNS（含 .git / node_modules 等）。"""
    for segment in path.replace("\\", "/").split("/"):
        if not segment:
            continue
        for pattern in IGNORE_PATTERNS:
            if fnmatch.fnmatch(segment, pattern):
                return True
    return False


def truncate_line(line: str, max_chars: int = DEFAULT_LINE_SUMMARY_LENGTH) -> str:
    """截断行到 max_chars，超出加 '...' 后缀。"""
    line = line.rstrip("\n\r")
    if len(line) <= max_chars:
        return line
    return line[: max_chars - 3] + "..."
