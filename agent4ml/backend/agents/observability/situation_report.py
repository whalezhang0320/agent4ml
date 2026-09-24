"""SituationReport — structured analysis when help is requested.

Programmatic part: extracts attempts, environment, blocker from
StallTracker failure records.
LLM part: one extra LLM call to synthesize 2-4 options + recommendation.

Design: design_docs/45 §3.5
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from agent4ml.backend.agents.observability.stall_tracker import ToolFailure


@dataclass
class SituationReport:
    reason: str
    attempts: list[dict[str, Any]] = field(default_factory=list)
    environment: dict[str, Any] = field(default_factory=dict)
    blocker: str = ""
    options: list[str] = field(default_factory=list)
    recommendation: str = ""

    def to_text(self) -> str:
        parts = [f"## Help Request · {self.reason}\n"]
        if self.attempts:
            parts.append("### Attempts")
            for i, a in enumerate(self.attempts, 1):
                parts.append(f"{i}. {a['command']} → {a['error']}")
            parts.append("")
        if self.environment:
            parts.append("### Environment")
            for k, v in self.environment.items():
                parts.append(f"- {k}: {v}")
            parts.append("")
        if self.blocker:
            parts.append(f"### Blocker\n{self.blocker}\n")
        if self.options:
            parts.append("### Options")
            for i, opt in enumerate(self.options, 1):
                marker = " (recommended)" if i == 1 and self.recommendation else ""
                parts.append(f"{i}. {opt}{marker}")
        return "\n".join(parts)


def build_programmatic_report(
    failures: list[ToolFailure],
    reason: str,
    environment: dict[str, Any] | None = None,
) -> SituationReport:
    """Build the programmatic part of a SituationReport from StallTracker data."""
    attempts = [
        {"command": f.command, "error": f.error, "capability": f.capability}
        for f in failures
    ]
    caps = {}
    for f in failures:
        caps.setdefault(f.capability, []).append(f.error)
    blocker_parts = []
    for cap, errors in caps.items():
        blocker_parts.append(f"{cap}: {len(errors)} failure(s)")
    blocker = "; ".join(blocker_parts) if blocker_parts else reason
    return SituationReport(
        reason=reason,
        attempts=attempts,
        environment=environment or {},
        blocker=blocker,
    )


_OPTIONS_PROMPT = """Given the following situation, list 2-4 viable options for the user.

Situation: {reason}
Blocker: {blocker}
Attempts:
{attempts}

For each option, provide a short one-line description. Mark the recommended
option with [RECOMMENDED] at the start. Return only the options, one per line."""


def synthesize_options_with_llm(
    report: SituationReport,
    llm: Any,
) -> SituationReport:
    """Call LLM to synthesize options + recommendation. Mutates and returns report."""
    attempts_text = "\n".join(
        f"- {a['command']} → {a['error']}" for a in report.attempts
    ) or "(no attempts recorded)"
    prompt = _OPTIONS_PROMPT.format(
        reason=report.reason, blocker=report.blocker, attempts=attempts_text,
    )
    try:
        response = llm.invoke(prompt)
        text = response.content if hasattr(response, "content") else str(response)
        lines = [ln.strip() for ln in text.strip().split("\n") if ln.strip()]
        for line in lines:
            if line.startswith("[RECOMMENDED]"):
                report.options.append(line.replace("[RECOMMENDED]", "").strip())
                report.recommendation = report.options[-1]
            else:
                report.options.append(line)
    except Exception:
        report.options = []
    return report
