"""Interrupt protection for atomic tasks.

Some operations (context compression, evidence persistence, report
generation) must not be interrupted mid-flight. This module provides
a thread-local flag that StallDetectionMiddleware checks before pausing.

Borrowed from hermes _aux_interrupt_protection pattern.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import Iterator

_protection_state = threading.local()


def is_interrupt_protected() -> bool:
    """Return True if the current thread is running an interrupt-protected task."""
    return bool(getattr(_protection_state, "active", False))


@contextmanager
def interrupt_protection(active: bool = True) -> Iterator[None]:
    """Mark the current thread as running an interrupt-protected task.

    StallDetectionMiddleware checks is_interrupt_protected() before
    pausing the graph. If protected, the pause is deferred until the
    context manager exits.
    """
    prev = getattr(_protection_state, "active", False)
    _protection_state.active = active
    try:
        yield
    finally:
        _protection_state.active = prev
