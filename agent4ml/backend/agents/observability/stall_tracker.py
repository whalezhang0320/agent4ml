"""StallTracker — detect when the agent is stuck in a dead-end.

Four signal types (design_docs/45 §3.3):
1. Capability exhaustion: same capability fails via 5 different command attempts.
2. Error pattern repetition: same error class recurs 5 times.
3. Todo stagnation: same todo stays in_progress for 15 consecutive LLM rounds.
4. No-progress long operation: tool running with no new output for 180s
   (requires RunActivityTracker heartbeat — wired in Phase 3).

All three active signals are suppressed when a successful tool call occurred
within the last 120 seconds (_success_decay_window). This prevents false
positives during active long-running coding tasks where transient errors
co-exist with genuine progress.

The tracker is stateless across runs; reset() clears all signals after a
help request is resolved so the agent gets a fresh start.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolFailure:
    capability: str
    error_class: str
    command: str
    error: str
    timestamp: float


_CAPABILITY_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("sandbox", re.compile(r"winerror|sandbox|exec failed|write failed|read failed|aio_sandbox|agent_sandbox", re.IGNORECASE)),
    ("postgres", re.compile(r"postgres|psql|pg_isready|pgvector", re.IGNORECASE)),
    ("root", re.compile(r"apt-get|apt |dpkg|sudo|/var/lib/apt", re.IGNORECASE)),
    ("docker", re.compile(r"docker|docker-compose|containerd", re.IGNORECASE)),
    ("network", re.compile(r"curl|wget|http://|https://|504|gateway.{0,10}timeout", re.IGNORECASE)),
]

_ERROR_CLASS_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("permission", re.compile(r"permission denied|eacces|403|forbidden", re.IGNORECASE)),
    ("sandbox", re.compile(r"winerror|sandbox.{0,20}(fail|error|refus)|exec failed|write failed|read failed", re.IGNORECASE)),
    ("network", re.compile(r"timeout|504|gateway.{0,10}timeout|connection refused|unreachable", re.IGNORECASE)),
    ("not_found", re.compile(r"not found|no such file|404|not installed|absent", re.IGNORECASE)),
]


def classify_capability(tool_name: str, tool_input: dict[str, Any], error: str) -> str:
    text = f"{tool_name} {tool_input} {error}"
    for cap, pattern in _CAPABILITY_PATTERNS:
        if pattern.search(text):
            return cap
    return "unknown"


def classify_error_class(error: str) -> str:
    for cls, pattern in _ERROR_CLASS_PATTERNS:
        if pattern.search(error):
            return cls
    return "unknown"


@dataclass
class StallTracker:
    capability_failure_threshold: int = 5
    error_pattern_threshold: int = 5
    todo_stagnation_rounds: int = 15
    no_progress_timeout: float = 180.0

    _failures: list[ToolFailure] = field(default_factory=list)
    _error_counts: dict[str, int] = field(default_factory=dict)
    _todo_in_progress_hash: str | None = None
    _todo_stagnation_count: int = 0
    _last_progress_ts: float = field(default_factory=time.time)
    # Timestamp of last successful tool call — used to decay stale failure signals.
    # 初始化为 0（epoch）让"从未成功过"时 _recent_success() 返 False（不抑制 stuck 信号）。
    # 只有显式调 record_tool_success() 后才设为 time.time()。
    _last_success_ts: float = 0.0
    # Window (seconds) within which a success resets capability failure tracking.
    _success_decay_window: float = 120.0

    @property
    def stuck(self) -> bool:
        return self._capability_stuck() or self._error_pattern_stuck() or self._todo_stuck()

    def _recent_success(self) -> bool:
        """True if a successful tool call happened within the decay window."""
        return (time.time() - self._last_success_ts) < self._success_decay_window

    def _capability_stuck(self) -> bool:
        if self._recent_success():
            return False
        caps: dict[str, set[str]] = {}
        for f in self._failures:
            if f.capability == "unknown":
                continue
            caps.setdefault(f.capability, set()).add(f.command)
        return any(len(cmds) >= self.capability_failure_threshold for cmds in caps.values())

    def _error_pattern_stuck(self) -> bool:
        if self._recent_success():
            return False
        return any(count >= self.error_pattern_threshold for count in self._error_counts.values())

    def _todo_stuck(self) -> bool:
        if self._recent_success():
            return False
        return self._todo_stagnation_count >= self.todo_stagnation_rounds

    def record_tool_failure(self, tool_name: str, tool_input: dict[str, Any], error: str) -> None:
        cap = classify_capability(tool_name, tool_input, error)
        cls = classify_error_class(error)
        cmd = str(tool_input.get("command", tool_input))[:200]
        self._failures.append(ToolFailure(cap, cls, cmd, error[:500], time.time()))
        self._error_counts[cls] = self._error_counts.get(cls, 0) + 1

    def record_tool_success(self) -> None:
        """Call after any successful tool execution to decay stale failure signals."""
        self._last_success_ts = time.time()
        self._last_progress_ts = time.time()

    def record_todo_state(self, todos: list[dict[str, Any]]) -> None:
        in_progress = sorted(
            t.get("content", "") for t in todos if t.get("status") == "in_progress"
        )
        current_hash = "|".join(in_progress)
        if current_hash == self._todo_in_progress_hash and current_hash:
            self._todo_stagnation_count += 1
        else:
            self._todo_stagnation_count = 1 if current_hash else 0
            self._todo_in_progress_hash = current_hash or None

    def record_progress(self) -> None:
        self._last_progress_ts = time.time()
        self._last_success_ts = time.time()

    def get_failures(self) -> list[ToolFailure]:
        return list(self._failures)

    def get_stuck_reason(self) -> str | None:
        if self._capability_stuck():
            caps = {}
            for f in self._failures:
                caps.setdefault(f.capability, set()).add(f.command)
            exhausted = [c for c, cmds in caps.items() if len(cmds) >= self.capability_failure_threshold]
            return f"capability exhausted: {', '.join(exhausted)}"
        if self._error_pattern_stuck():
            exceeded = [c for c, n in self._error_counts.items() if n >= self.error_pattern_threshold]
            return f"error pattern repeated: {', '.join(exceeded)}"
        if self._todo_stuck():
            return f"todo stagnated for {self._todo_stagnation_count} rounds"
        return None

    def reset(self) -> None:
        self._failures.clear()
        self._error_counts.clear()
        self._todo_in_progress_hash = None
        self._todo_stagnation_count = 0
        self._last_progress_ts = time.time()
        # reset 后 _last_success_ts=0，_recent_success() 返 False（不抑制新的失败信号）。
        # 与构造时一致：从未成功过 = 不 decay。
        self._last_success_ts = 0.0
