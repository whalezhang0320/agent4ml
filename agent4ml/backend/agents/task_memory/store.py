from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import tempfile
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any, BinaryIO

from agent4ml.backend.agents.task_memory.errors import (
    TaskMemoryIntegrityError,
    TaskMemoryValidationError,
)
from agent4ml.backend.agents.task_memory.models import TaskGraph, validate_result_ref

_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_GLOBAL_LOCK_GUARD = threading.Lock()
_GLOBAL_LOCKS: dict[str, threading.RLock] = {}


def validate_storage_id(value: str, field_name: str) -> str:
    if not _SAFE_ID_RE.fullmatch(value or "") or value in {".", ".."}:
        raise TaskMemoryValidationError(f"invalid {field_name}: {value!r}")
    return value


def sha256_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


class TaskMemoryStore:
    """Low-level durable files for one explicit thread/task pair."""

    def __init__(self, root: str | Path, thread_id: str, task_id: str) -> None:
        self.root = Path(root).expanduser().resolve()
        self.thread_id = validate_storage_id(str(thread_id), "thread_id")
        self.task_id = validate_storage_id(str(task_id), "task_id")
        self.task_dir = self.root / "threads" / self.thread_id / "tasks" / self.task_id
        self.refs_dir = self.task_dir / "refs"
        self.graph_path = self.task_dir / "task_graph.json"
        self.log_path = self.task_dir / "offload.jsonl"
        self.lock_path = self.task_dir / ".task-memory.lock"

    def ensure_dirs(self) -> None:
        self.refs_dir.mkdir(parents=True, exist_ok=True)
        if self.refs_dir.is_symlink():
            raise TaskMemoryValidationError("refs directory may not be a symlink")
        resolved_task = self.task_dir.resolve()
        expected_parent = (self.root / "threads" / self.thread_id / "tasks").resolve()
        if expected_parent not in resolved_task.parents:
            raise TaskMemoryValidationError("task directory escaped task-memory root")
        if self.refs_dir.resolve().parent != resolved_task:
            raise TaskMemoryValidationError("refs directory escaped task directory")

    @contextlib.contextmanager
    def transaction(self) -> Iterator[None]:
        """Serialize a full ref/WAL/graph transaction in-process and cross-process."""
        self.ensure_dirs()
        lock_key = str(self.lock_path)
        with _GLOBAL_LOCK_GUARD:
            lock = _GLOBAL_LOCKS.setdefault(lock_key, threading.RLock())
        with lock:
            with open(self.lock_path, "a+b") as handle:
                self._lock_file(handle)
                try:
                    yield
                finally:
                    self._unlock_file(handle)

    @staticmethod
    def _lock_file(handle: BinaryIO) -> None:
        try:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        except ImportError:  # pragma: no cover - exercised on Windows.
            import msvcrt

            handle.seek(0, os.SEEK_END)
            if handle.tell() == 0:
                handle.write(b"\0")
                handle.flush()
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)

    @staticmethod
    def _unlock_file(handle: BinaryIO) -> None:
        try:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        except ImportError:  # pragma: no cover
            import msvcrt

            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)

    def read_events(self, *, repair_tail: bool = False) -> list[dict[str, Any]]:
        if not self.log_path.exists():
            return []
        raw = self.log_path.read_bytes()
        if not raw:
            return []
        lines = raw.splitlines(keepends=True)
        events: list[dict[str, Any]] = []
        offset = 0
        for index, line in enumerate(lines):
            is_last = index == len(lines) - 1
            try:
                event = json.loads(line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                if repair_tail and is_last and not line.endswith(b"\n"):
                    self._truncate_log(offset)
                    break
                raise TaskMemoryIntegrityError(
                    f"invalid offload.jsonl line {index + 1}: {exc}"
                ) from exc
            if not isinstance(event, dict):
                raise TaskMemoryIntegrityError("offload event must be a JSON object")
            events.append(event)
            offset += len(line)
            if repair_tail and is_last and not line.endswith(b"\n"):
                with open(self.log_path, "ab") as handle:
                    handle.write(b"\n")
                    handle.flush()
                    os.fsync(handle.fileno())
        self._validate_event_sequence(events)
        return events

    def _truncate_log(self, size: int) -> None:
        with open(self.log_path, "r+b") as handle:
            handle.truncate(size)
            handle.flush()
            os.fsync(handle.fileno())
        self._fsync_dir(self.task_dir)

    @staticmethod
    def _validate_event_sequence(events: list[dict[str, Any]]) -> None:
        for expected, event in enumerate(events, start=1):
            if event.get("seq") != expected:
                raise TaskMemoryIntegrityError("offload seq is not a contiguous prefix")
            if event.get("event_id") != f"E{expected:06d}":
                raise TaskMemoryIntegrityError("offload event_id does not match seq")
            if expected == 1 and event.get("event_type") != "task_created":
                raise TaskMemoryIntegrityError("task_created must be the first event")
            if expected > 1 and event.get("event_type") == "task_created":
                raise TaskMemoryIntegrityError("task_created may appear only once")
            event_type = event.get("event_type")
            if event_type not in {"task_created", "tool_result", "graph_patch"}:
                raise TaskMemoryIntegrityError(f"unknown offload event type: {event_type!r}")
            if event_type == "task_created":
                if not isinstance(event.get("initial_graph"), dict):
                    raise TaskMemoryIntegrityError("task_created lacks initial_graph")
            elif event_type == "tool_result":
                required = {
                    "task_id", "run_id", "tool_call_id", "tool_name", "tool_status",
                    "routing_status", "result_ref", "content_hash", "payload_hash",
                }
                if required.difference(event):
                    raise TaskMemoryIntegrityError("tool_result event is incomplete")
                if event.get("routing_status") not in {"assigned", "orphan"}:
                    raise TaskMemoryIntegrityError("invalid tool_result routing_status")
                if event.get("tool_status") not in {"ok", "error"}:
                    raise TaskMemoryIntegrityError("invalid tool_result tool_status")
                if event.get("routing_status") == "orphan" and event.get("node_id") is not None:
                    raise TaskMemoryIntegrityError("orphan tool_result must not name a node")
                if event.get("routing_status") == "assigned" and not isinstance(event.get("node_id"), str):
                    raise TaskMemoryIntegrityError("assigned tool_result requires node_id")
                try:
                    validate_result_ref(str(event.get("result_ref")))
                except ValueError as exc:
                    raise TaskMemoryIntegrityError("invalid tool_result result_ref") from exc
                for hash_name in ("content_hash", "payload_hash"):
                    value = str(event.get(hash_name) or "")
                    if not re.fullmatch(r"sha256:[0-9a-f]{64}", value):
                        raise TaskMemoryIntegrityError(f"invalid {hash_name}")
            elif event_type == "graph_patch":
                if not isinstance(event.get("operations"), list):
                    raise TaskMemoryIntegrityError("graph_patch operations must be a list")
                base = event.get("base_graph_version")
                target = event.get("target_graph_version")
                if not isinstance(base, int) or not isinstance(target, int) or target != base + 1:
                    raise TaskMemoryIntegrityError("invalid graph_patch version transition")

    def append_event(self, event: dict[str, Any]) -> None:
        data = (
            json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n"
        ).encode("utf-8")
        fd = os.open(self.log_path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
        try:
            written = os.write(fd, data)
            if written != len(data):
                raise OSError("short offload.jsonl append")
            os.fsync(fd)
        finally:
            os.close(fd)

    def load_graph(self) -> TaskGraph | None:
        if not self.graph_path.exists():
            return None
        try:
            return TaskGraph.model_validate_json(self.graph_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise TaskMemoryIntegrityError(f"invalid task_graph.json: {exc}") from exc

    def write_graph(self, graph: TaskGraph) -> None:
        data = json.dumps(
            graph.as_dict(), ensure_ascii=False, indent=2, sort_keys=True
        ).encode("utf-8")
        self._atomic_replace(self.graph_path, data)

    def write_ref(self, result_ref: str, data: bytes) -> str:
        validate_result_ref(result_ref)
        target = self._resolve_ref(result_ref)
        if target.exists():
            raise TaskMemoryIntegrityError(f"ref already exists: {result_ref}")
        fd, temp_name = tempfile.mkstemp(prefix=".ref-", suffix=".tmp", dir=self.refs_dir)
        temp = Path(temp_name)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            try:
                os.link(temp, target)
                temp.unlink()
            except (AttributeError, OSError):
                if target.exists():
                    raise TaskMemoryIntegrityError(f"ref already exists: {result_ref}")
                os.rename(temp, target)
            self._fsync_dir(self.refs_dir)
        finally:
            if temp.exists():
                temp.unlink()
        return sha256_bytes(data)

    def read_ref(self, result_ref: str) -> bytes:
        target = self._resolve_ref(result_ref)
        if not target.exists() or not target.is_file() or target.is_symlink():
            raise TaskMemoryIntegrityError(f"missing or unsafe ref: {result_ref}")
        return target.read_bytes()

    def verify_ref(self, result_ref: str, expected_hash: str) -> None:
        actual = sha256_bytes(self.read_ref(result_ref))
        if actual != expected_hash:
            raise TaskMemoryIntegrityError(
                f"ref hash mismatch for {result_ref}: {actual} != {expected_hash}"
            )

    def list_unregistered_refs(self, events: list[dict[str, Any]]) -> list[str]:
        registered = {
            event.get("result_ref")
            for event in events
            if event.get("event_type") == "tool_result"
        }
        return sorted(
            f"refs/{path.name}"
            for path in self.refs_dir.glob("*.md")
            if f"refs/{path.name}" not in registered
        )

    def _resolve_ref(self, result_ref: str) -> Path:
        validate_result_ref(result_ref)
        target = self.task_dir / PurePathCompat(result_ref)
        resolved_parent = target.parent.resolve()
        if resolved_parent != self.refs_dir.resolve():
            raise TaskMemoryValidationError("ref escaped task refs directory")
        return target

    def _atomic_replace(self, target: Path, data: bytes) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_name = tempfile.mkstemp(prefix=f".{target.name}-", suffix=".tmp", dir=target.parent)
        temp = Path(temp_name)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp, target)
            self._fsync_dir(target.parent)
        finally:
            if temp.exists():
                temp.unlink()

    @staticmethod
    def _fsync_dir(path: Path) -> None:
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        fd = os.open(path, flags)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def PurePathCompat(value: str) -> Path:
    """Convert an already validated POSIX task-relative path on this platform."""
    return Path(*value.split("/"))
