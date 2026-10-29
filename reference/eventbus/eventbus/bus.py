"""The bus: subscription, dispatch, retry and audit."""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any, Callable

from eventbus.event import Event, EventBusError
from eventbus.matching import matches, specificity


@dataclass
class Subscription:
    """Handle returned by ``subscribe``; pass it back to ``unsubscribe``."""

    id: int
    pattern: str
    handler: Callable[[Event], Any] = field(compare=False, repr=False)

    def __hash__(self) -> int:
        return hash(self.id)


@dataclass
class PublishResult:
    """What one publish did."""

    sequence: int
    matched: int = 0
    succeeded: int = 0
    failed: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "sequence": self.sequence,
            "matched": self.matched,
            "succeeded": self.succeeded,
            "failed": self.failed,
        }


class EventBus:
    """In-process publish/subscribe bus."""

    def __init__(self, max_attempts: int = 1) -> None:
        if max_attempts < 1:
            raise EventBusError("max_attempts must be at least 1")
        self.max_attempts = max_attempts
        self._subs: list[Subscription] = []
        self._middleware: list[Callable[[Event, Callable], Any]] = []
        self._audit: list[Event] = []
        self._ids = itertools.count(1)
        self._sequence = itertools.count(1)

    # ---- registration ----
    def subscribe(self, pattern: str, handler: Callable[[Event], Any]) -> Subscription:
        specificity(pattern)  # validates the pattern eagerly
        sub = Subscription(id=next(self._ids), pattern=pattern, handler=handler)
        self._subs.append(sub)
        return sub

    def unsubscribe(self, subscription: Subscription) -> None:
        """Remove a subscription. Removing one twice is a no-op."""
        self._subs = [s for s in self._subs if s.id != subscription.id]

    def use(self, middleware: Callable[[Event, Callable], Any]) -> None:
        """Register middleware. First registered is outermost."""
        self._middleware.append(middleware)

    # ---- dispatch ----
    def publish(self, topic: str, payload: Any = None) -> PublishResult:
        event = Event(topic=topic, payload=payload, sequence=next(self._sequence))
        self._audit.append(event)

        targets = [s for s in self._subs if matches(s.pattern, topic)]
        targets.sort(key=lambda s: (-specificity(s.pattern), s.id))

        result = PublishResult(sequence=event.sequence, matched=len(targets))
        for sub in targets:
            if self._invoke(sub, event):
                result.succeeded += 1
            else:
                result.failed += 1
        return result

    def _invoke(self, sub: Subscription, event: Event) -> bool:
        """Run one handler through the middleware chain, with retries."""
        for _ in range(self.max_attempts):
            try:
                self._chain(sub.handler)(event)
                return True
            except Exception:
                continue
        return False

    def _chain(self, handler: Callable[[Event], Any]) -> Callable[[Event], Any]:
        """Wrap a handler in the middleware stack, outermost first."""
        call = handler
        for mw in reversed(self._middleware):
            call = _bind(mw, call)
        return call

    # ---- audit ----
    def audit(self) -> list[Event]:
        return list(self._audit)

    def clear_audit(self) -> None:
        self._audit.clear()


def _bind(middleware: Callable[[Event, Callable], Any], nxt: Callable[[Event], Any]):
    def call(event: Event) -> Any:
        return middleware(event, nxt)

    return call
