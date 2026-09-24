from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


class CapabilityMissingError(KeyError):
    """Raised when a required runtime capability is missing."""


@dataclass(frozen=True)
class CapabilityRegistry:
    models: dict[str, Any] = field(default_factory=dict)
    tools: dict[str, Any] = field(default_factory=dict)
    reporter: Any | None = None
    artifact_store: Any | None = None
    sandbox_provider: Any | None = None
    skill_store: Any | None = None
    specialist_registry: Any | None = None
    subagent_provider: Any | None = None
    memory_provider: Any | None = None

    def get_model(self, name: str) -> Any:
        try:
            return self.models[name]
        except KeyError as exc:
            raise CapabilityMissingError(f"model not registered: {name}") from exc

    def get_tool(self, name: str) -> Any:
        try:
            return self.tools[name]
        except KeyError as exc:
            raise CapabilityMissingError(f"tool not registered: {name}") from exc

    def get_reporter(self) -> Any:
        if self.reporter is None:
            raise CapabilityMissingError("reporter not registered")
        return self.reporter

    def get_artifact_store(self) -> Any:
        if self.artifact_store is None:
            raise CapabilityMissingError("artifact_store not registered")
        return self.artifact_store

    def get_sandbox_provider(self) -> Any:
        if self.sandbox_provider is None:
            raise CapabilityMissingError("sandbox_provider not registered")
        return self.sandbox_provider

    def get_skill_store(self) -> Any:
        if self.skill_store is None:
            raise CapabilityMissingError("skill_store not registered")
        return self.skill_store

    def get_specialist_registry(self) -> Any:
        if self.specialist_registry is None:
            raise CapabilityMissingError("specialist_registry not registered")
        return self.specialist_registry

    def get_subagent_provider(self) -> Any:
        if self.subagent_provider is None:
            raise CapabilityMissingError("subagent_provider not registered")
        return self.subagent_provider

    def get_memory_provider(self) -> Any:
        if self.memory_provider is None:
            raise CapabilityMissingError("memory_provider not registered")
        return self.memory_provider
