"""Middleware that wraps handler invocation."""

from __future__ import annotations

import time
from typing import Any, Callable

from eventbus.event import Event


class TimingMiddleware:
    """Records the topic and duration of every call it wraps."""

    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []

    def __call__(self, event: Event, nxt: Callable[[Event], Any]) -> Any:
        start = time.perf_counter()
        try:
            return nxt(event)
        finally:
            self.records.append(
                {"topic": event.topic, "duration": time.perf_counter() - start}
            )

    def clear(self) -> None:
        self.records.clear()
