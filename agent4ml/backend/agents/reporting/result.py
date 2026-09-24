from __future__ import annotations

from dataclasses import dataclass, field

from agent4ml.backend.agents.state.types import Artifact


@dataclass(frozen=True)
class ReportResult:
    final_report: str
    artifacts: tuple[Artifact, ...] = field(default_factory=tuple)
    warnings: tuple[str, ...] = field(default_factory=tuple)
