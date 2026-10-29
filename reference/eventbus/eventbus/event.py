"""The event model."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


class EventBusError(Exception):
    """Base class for every error eventbus raises."""


@dataclass(frozen=True)
class Event:
    """An immutable published event.

    ``timestamp`` is excluded from equality: two events describing the same
    thing are equal regardless of when they were constructed.
    """

    topic: str
    payload: Any
    sequence: int
    timestamp: float = field(default_factory=time.time, compare=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "topic": self.topic,
            "payload": self.payload,
            "sequence": self.sequence,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Event:
        return cls(
            topic=data["topic"],
            payload=data.get("payload"),
            sequence=int(data["sequence"]),
            timestamp=float(data.get("timestamp", time.time())),
        )
