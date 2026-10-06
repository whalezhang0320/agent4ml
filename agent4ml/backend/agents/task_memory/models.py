from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from agent4ml.backend.agents.task_memory.errors import TaskMemoryValidationError

NodeType = Literal["action", "finding", "decision", "verification", "milestone"]
NodeStatus = Literal[
    "todo", "ready", "doing", "blocked", "verifying", "done", "failed",
    "skipped", "superseded",
]
TaskStatus = Literal["active", "blocked", "verifying", "done", "cancelled"]
EdgeType = Literal[
    "next", "depends_on", "branches_to", "derived_from", "blocks",
    "validates", "supersedes",
]

_NODE_ID_RE = re.compile(r"^N[A-Za-z0-9_-]{1,63}$")
_EVENT_ID_RE = re.compile(r"^E\d{6,}$")


def validate_result_ref(value: str) -> str:
    """Accept only a task-local POSIX path below ``refs/``."""
    if not value or "\\" in value:
        raise ValueError("result_ref must be a POSIX task-local path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or len(path.parts) != 2:
        raise ValueError("result_ref must be directly below refs/")
    if path.parts[0] != "refs" or path.suffix != ".md":
        raise ValueError("result_ref must match refs/*.md")
    return value


class TaskNode(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str = Field(min_length=1, max_length=500)
    type: NodeType = "action"
    status: NodeStatus = "todo"
    summary: str = Field(default="", max_length=300)
    next_action: str | None = Field(default=None, max_length=150)
    acceptance_criteria: list[str] = Field(default_factory=list)
    event_refs: list[str] = Field(default_factory=list)
    result_refs: list[str] = Field(default_factory=list)
    created_at: str = ""
    updated_at: str = ""
    revision: int = Field(default=1, ge=1)

    @model_validator(mode="after")
    def validate_refs_and_done(self) -> "TaskNode":
        if len(self.event_refs) != len(set(self.event_refs)):
            raise ValueError("node event_refs must be unique")
        if len(self.result_refs) != len(set(self.result_refs)):
            raise ValueError("node result_refs must be unique")
        if any(not _EVENT_ID_RE.fullmatch(event_id) for event_id in self.event_refs):
            raise ValueError("invalid event id in node")
        for result_ref in self.result_refs:
            validate_result_ref(result_ref)
        if self.type == "action" and self.status == "done":
            if not self.acceptance_criteria:
                raise ValueError("done action requires acceptance criteria")
            if not self.event_refs or not self.result_refs:
                raise ValueError("done action requires persisted evidence")
        return self


class TaskEdge(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    from_node: str = Field(alias="from")
    to: str
    type: EdgeType
    label: str | None = Field(default=None, max_length=300)

    def as_dict(self) -> dict[str, Any]:
        return self.model_dump(by_alias=True, exclude_none=True)


class TaskGraph(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    graph_version: int = Field(ge=1)
    task_id: str
    thread_id: str
    goal: str = Field(min_length=1)
    success_criteria: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    status: TaskStatus = "active"
    current_node: str | None
    pending_event_refs: list[str] = Field(default_factory=list)
    created_at: str
    updated_at: str
    offload_cursor: int = Field(ge=1)
    nodes: dict[str, TaskNode]
    edges: list[TaskEdge] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_graph(self) -> "TaskGraph":
        if not self.nodes:
            raise ValueError("task graph requires at least one node")
        for node_id in self.nodes:
            if not _NODE_ID_RE.fullmatch(node_id):
                raise ValueError(f"invalid node id: {node_id}")
        if self.current_node is None:
            if self.status not in {"done", "cancelled"}:
                raise ValueError("active task requires current_node")
        elif self.current_node not in self.nodes:
            raise ValueError("current_node does not exist")
        if len(self.pending_event_refs) != len(set(self.pending_event_refs)):
            raise ValueError("pending_event_refs must be unique")
        if any(not _EVENT_ID_RE.fullmatch(event_id) for event_id in self.pending_event_refs):
            raise ValueError("invalid pending event id")

        seen_edges: set[tuple[str, str, str, str | None]] = set()
        precedence: dict[str, set[str]] = {node_id: set() for node_id in self.nodes}
        for edge in self.edges:
            if edge.from_node not in self.nodes or edge.to not in self.nodes:
                raise ValueError("edge endpoint does not exist")
            key = (edge.from_node, edge.to, edge.type, edge.label)
            if key in seen_edges:
                raise ValueError("duplicate edge")
            seen_edges.add(key)
            # Normalize both execution relationships to prerequisite -> successor.
            if edge.type == "next":
                precedence[edge.from_node].add(edge.to)
            elif edge.type == "depends_on":
                precedence[edge.to].add(edge.from_node)
        _assert_acyclic(precedence)
        return self

    def as_dict(self) -> dict[str, Any]:
        return self.model_dump(by_alias=True, exclude_none=False)

    def validate_event_refs(self, known_event_ids: set[str]) -> None:
        referenced = set(self.pending_event_refs)
        for node in self.nodes.values():
            referenced.update(node.event_refs)
        missing = referenced.difference(known_event_ids)
        if missing:
            raise TaskMemoryValidationError(
                f"graph references unknown events: {sorted(missing)}"
            )


def _assert_acyclic(adjacency: dict[str, set[str]]) -> None:
    state: dict[str, int] = {}

    def visit(node: str) -> None:
        marker = state.get(node, 0)
        if marker == 1:
            raise ValueError("execution graph contains a cycle")
        if marker == 2:
            return
        state[node] = 1
        for successor in adjacency.get(node, ()):
            visit(successor)
        state[node] = 2

    for node in adjacency:
        visit(node)


@dataclass(frozen=True)
class FlushReceipt:
    task_id: str
    graph_version: int
    persisted_message_ids: tuple[str, ...] = ()
    event_ids: tuple[str, ...] = ()
    result_refs: tuple[str, ...] = ()
    safe_to_drop: bool = False
    errors: tuple[str, ...] = ()
    graph_path: str = ""
    current_node: str | None = None
    status: str = "active"
    message_refs: dict[str, str] = field(default_factory=dict)

    @classmethod
    def failed(cls, task_id: str, error: Exception | str) -> "FlushReceipt":
        return cls(task_id=task_id, graph_version=0, errors=(str(error),))

    def pointer(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "graph_path": self.graph_path,
            "graph_version": self.graph_version,
            "current_node": self.current_node,
            "status": self.status,
        }
