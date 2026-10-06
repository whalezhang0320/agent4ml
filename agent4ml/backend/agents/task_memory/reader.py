from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from agent4ml.backend.agents.task_memory.errors import TaskMemoryValidationError
from agent4ml.backend.agents.task_memory.projector import ContextProjector
from agent4ml.backend.agents.task_memory.writer import TaskGraphWriter


class TaskMemoryReader:
    """Safe task-local expansion API used by future internal tools."""

    def __init__(self, root: str | Path, thread_id: str, task_id: str) -> None:
        self.writer = TaskGraphWriter(root, thread_id, task_id)
        self.store = self.writer.store

    def get_graph(self) -> dict[str, Any]:
        return self.writer.get_graph().as_dict()

    def get_context(self, *, max_chars: int = 12_000) -> str:
        return ContextProjector(max_chars=max_chars).project(self.writer.get_graph())

    def get_events(self, event_ids: Iterable[str]) -> list[dict[str, Any]]:
        wanted = set(event_ids)
        with self.store.transaction():
            graph, events = self.writer._recover_locked()  # recovery precedes every read
            self.writer._verify_locked(graph, events)
        found = [event for event in events if event["event_id"] in wanted]
        if {event["event_id"] for event in found} != wanted:
            raise TaskMemoryValidationError("one or more requested events do not exist")
        return found

    def read_ref(self, result_ref: str) -> dict[str, Any]:
        with self.store.transaction():
            _graph, events = self.writer._recover_locked()
            matches = [
                event for event in events
                if event.get("event_type") == "tool_result"
                and event.get("result_ref") == result_ref
            ]
            if len(matches) != 1:
                raise TaskMemoryValidationError("ref is not registered exactly once")
            event = matches[0]
            self.store.verify_ref(result_ref, event["content_hash"])
            content = self.store.read_ref(result_ref).decode("utf-8")
        return {
            "trust": "untrusted_evidence",
            "result_ref": result_ref,
            "event_id": event["event_id"],
            "content": content,
        }

    def expand_node(self, node_id: str) -> dict[str, Any]:
        graph = self.writer.get_graph()
        if node_id not in graph.nodes:
            raise TaskMemoryValidationError(f"unknown node: {node_id}")
        neighbors = []
        for edge in graph.edges:
            if edge.from_node == node_id or edge.to == node_id:
                neighbors.append(edge.as_dict())
        events = self.get_events(graph.nodes[node_id].event_refs)
        return {
            "node_id": node_id,
            "node": graph.nodes[node_id].model_dump(),
            "edges": neighbors,
            "events": events,
        }

    def trace_node(self, node_id: str) -> dict[str, Any]:
        graph = self.writer.get_graph()
        if node_id not in graph.nodes:
            raise TaskMemoryValidationError(f"unknown node: {node_id}")
        incoming = [
            edge.as_dict() for edge in graph.edges
            if edge.to == node_id and edge.type in {"next", "derived_from", "validates", "supersedes"}
        ]
        outgoing = [
            edge.as_dict() for edge in graph.edges
            if edge.from_node == node_id
        ]
        return {"node_id": node_id, "incoming": incoming, "outgoing": outgoing}
