from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, is_dataclass
from datetime import datetime
from typing import Any

from agent4ml.backend.agents.context_engineering.strategies.default._constants import CST

logger = logging.getLogger(__name__)


class SnapshotExecutor:
    """P4 压缩前存快照 + 记 path。"""

    def __init__(self, snapshot_dir: str = ".agent4ml/snapshots") -> None:
        self._dir = snapshot_dir

    def snapshot_if_pending(self, governance: dict | None, messages: list, state: Any) -> dict | None:
        governance = governance or {}
        pending = (governance.get("default") or {}).get("pending") or []
        if "P4" not in pending:
            return None
        logger.info("snapshot: P4 pending, building snapshot (msgs=%d, dir=%s)", len(messages), self._dir)
        snapshot = self._build_snapshot(messages, state)
        path = self._write_to_disk(snapshot)
        if path is None:
            logger.error("snapshot: _write_to_disk failed, snapshot NOT saved")
            return None
        logger.info("snapshot: saved to %s", path)
        g = dict(governance)
        d = dict(g.get("default") or {})
        d["snapshot_path"] = path
        metrics = dict(d.get("metrics") or {})
        metrics["snapshot_count"] = metrics.get("snapshot_count", 0) + 1
        d["metrics"] = metrics
        g["default"] = d
        return g

    def _build_snapshot(self, messages: list, state: Any) -> dict:
        state = state or {}
        return {
            "created_at": datetime.now(CST).isoformat(),
            "messages": [self._serialize_msg(m) for m in messages],
            "research_question": state.get("research_question"),
            "todos": self._serialize_list(state.get("todos")),
            "observations": self._serialize_list(state.get("observations")),
            "reflection_items": self._serialize_list(state.get("reflection_items")),
        }

    @staticmethod
    def _serialize_list(items: Any) -> list:
        """把 dataclass 列表转成 dict 列表，防 json.dump 序列化失败。"""
        if not items:
            return []
        result: list = []
        for item in items:
            if is_dataclass(item):
                result.append(asdict(item))
            elif isinstance(item, dict):
                result.append(item)
            else:
                result.append(str(item))
        return result

    @staticmethod
    def _serialize_msg(msg: Any) -> dict:
        return {
            "type": type(msg).__name__,
            "content": msg.content if isinstance(msg.content, str) else str(msg.content),
            "id": getattr(msg, "id", None),
            "tool_calls": getattr(msg, "tool_calls", None) or None,
            "tool_call_id": getattr(msg, "tool_call_id", None),
            "name": getattr(msg, "name", None),
        }

    def _write_to_disk(self, snapshot: dict) -> str | None:
        try:
            abs_dir = os.path.abspath(self._dir)
            os.makedirs(abs_dir, exist_ok=True)
            ts = datetime.now(CST).strftime("%Y%m%d_%H%M%S")
            filename = f"snapshot-{ts}.json"
            filepath = os.path.join(abs_dir, filename)
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(snapshot, f, ensure_ascii=False, indent=2, default=str)
            return filepath
        except (OSError, TypeError) as exc:
            logger.error("snapshot write failed: %s (dir=%s, type=%s)", exc, self._dir, type(exc).__name__)
            return None
