"""熔断器 — per-tool 健康状态机。

INVARIANT:
- closed: 正常态，allow_call 返 True
- open: 拒绝调用，cooldown 后转 half_open 放探针
- half_open: 放行 1 次探针，成功归 closed，失败重置 open
- 线程安全：_lock 保护状态转换
"""
from __future__ import annotations

import threading
import time

_CIRCUIT_BREAKER_THRESHOLD = 3
_CIRCUIT_BREAKER_COOLDOWN_SEC = 60.0


class CircuitBreaker:
    """每工具一个实例。被动触发，不主动 ping。"""

    __slots__ = ("failure_count", "state", "opened_at", "_lock")

    def __init__(self) -> None:
        self.failure_count: int = 0
        self.state: str = "closed"
        self.opened_at: float = 0.0
        self._lock = threading.Lock()

    def allow_call(self) -> bool:
        """closed→True；open 且 cooldown 未过→False；open 且 cooldown 过→half_open 返 True。"""
        with self._lock:
            if self.state == "closed":
                return True
            if self.state == "open":
                if time.time() - self.opened_at >= _CIRCUIT_BREAKER_COOLDOWN_SEC:
                    self.state = "half_open"
                    return True
                return False
            # half_open：只放一次探针
            return True

    def record_success(self) -> None:
        """成功归零，closed。"""
        with self._lock:
            self.failure_count = 0
            self.state = "closed"

    def record_failure(self) -> None:
        """失败计数++；closed 下连续 3 次→open；half_open 失败→重置 open。"""
        with self._lock:
            self.failure_count += 1
            if self.state == "half_open":
                self.state = "open"
                self.opened_at = time.time()
            elif self.state == "closed" and self.failure_count >= _CIRCUIT_BREAKER_THRESHOLD:
                self.state = "open"
                self.opened_at = time.time()
