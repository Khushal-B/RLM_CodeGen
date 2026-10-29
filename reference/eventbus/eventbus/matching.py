"""Topic pattern matching and specificity."""

from __future__ import annotations

from eventbus.event import EventBusError


def _segments(text: str) -> list[str]:
    return text.split(".") if text else []


def _validate(pattern: str) -> list[str]:
    parts = _segments(pattern)
    for i, part in enumerate(parts):
        if part == "#" and i != len(parts) - 1:
            raise EventBusError(
                "'#' may only appear as the last segment of a pattern: " + pattern
            )
    return parts


def matches(pattern: str, topic: str) -> bool:
    """True when ``topic`` matches ``pattern``.

    ``*`` matches exactly one segment; ``#`` matches one or more trailing
    segments and is only legal as the final segment.
    """
    pat = _validate(pattern)
    top = _segments(topic)

    for i, part in enumerate(pat):
        if part == "#":
            # Must consume at least one remaining segment.
            return len(top) >= i + 1
        if i >= len(top):
            return False
        if part == "*":
            continue
        if part != top[i]:
            return False
    return len(pat) == len(top)


def specificity(pattern: str) -> int:
    """Higher is more specific.

    Exact beats ``*`` beats ``#``; within a class, more segments beats fewer.
    """
    parts = _validate(pattern)
    if any(p == "#" for p in parts):
        rank = 0
    elif any(p == "*" for p in parts):
        rank = 1
    else:
        rank = 2
    return rank * 1000 + len(parts)
