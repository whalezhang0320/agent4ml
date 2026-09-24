from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage


class BudgetTrackerExecutor:
    """累计 token + 算 fraction + 标 pending。"""

    def __init__(self, thresholds: dict[str, float]) -> None:
        self._thresholds = thresholds

    def init_budget(self, governance: dict | None) -> dict:
        g = dict(governance or {})
        d = dict(g.get("default") or {})
        d["budget"] = {"input": 0, "output": 0, "total": 0, "current": 0, "window": 0, "fraction": 0.0}
        d["seen_msgs"] = {}
        d["pending"] = []
        d["warned"] = False
        d["p1_completed"] = False
        d["p1_skip_until_fraction"] = 0.0
        g["default"] = d
        return g

    def track(self, governance: dict | None, messages: list, token_counter: Any, window: int) -> dict:
        g = dict(governance or {})
        d = dict(g.get("default") or {})
        seen = dict(d.get("seen_msgs") or {})
        budget = dict(d.get("budget") or {"input": 0, "output": 0, "total": 0, "current": 0, "window": window, "fraction": 0.0})
        budget["window"] = window

        for msg in messages:
            if isinstance(msg, AIMessage) and msg.id:
                usage = getattr(msg, "usage_metadata", None) or {}
                in_t = usage.get("input_tokens", 0)
                out_t = usage.get("output_tokens", 0)
                prev_in, prev_out = seen.get(msg.id, (0, 0))
                diff_in = max(0, in_t - prev_in)
                diff_out = max(0, out_t - prev_out)
                if diff_in or diff_out:
                    budget["input"] += diff_in
                    budget["output"] += diff_out
                    budget["total"] += diff_in + diff_out
                    seen[msg.id] = (in_t, out_t)

        current = token_counter(messages)
        budget["current"] = current
        budget["fraction"] = current / window if window > 0 else 0.0

        fraction = budget["fraction"]
        pending: list[str] = []
        if fraction >= self._thresholds["p1_externalize"]:
            pending.append("P1")
        if fraction >= self._thresholds["p2_thinking"]:
            pending.append("P2")
        if fraction >= self._thresholds["p4_summarize"]:
            pending.append("P4")
        if fraction >= self._thresholds["p5_stop_toolcall"]:
            pending.append("P5")

        d["budget"] = budget
        d["seen_msgs"] = seen
        d["pending"] = pending
        g["default"] = d
        return g

    def clear_run_state(self, governance: dict | None) -> dict:
        g = dict(governance or {})
        d = dict(g.get("default") or {})
        for k in ("budget", "seen_msgs", "pending", "warned", "p1_completed", "p1_skip_until_fraction"):
            d.pop(k, None)
        g["default"] = d
        return g
