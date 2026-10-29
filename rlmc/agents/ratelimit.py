"""Global rate limiting and retry for flaky / rate-capped free endpoints.

Why this is patched onto the client class rather than wrapped at our call sites
-------------------------------------------------------------------------------
``rlm.RLM`` constructs its own clients internally (``_spawn_completion_context``
calls ``get_client`` per completion, and every recursive sub-call does it
again). So wrapping the client object we happen to hold would leave most calls
unthrottled. Patching ``OpenAIClient.completion`` once covers every call the
whole tree makes, including RLM's own.

What it provides
----------------
* a process-wide sliding-window limiter, so N concurrent build nodes cannot
  collectively exceed the account's requests-per-minute cap;
* retry with exponential backoff and jitter on 429 / 5xx / timeouts / connection
  resets, which free tiers produce routinely;
* a bounded final failure: after the last attempt the error is raised, and the
  driver records it as a node error rather than silently scoring the run.

The limiter is deliberately conservative: it counts a request when it STARTS,
not when it finishes, because the cap is on requests admitted per minute.
"""

from __future__ import annotations

import random
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass
class RateLimiter:
    """Process-wide sliding-window request limiter."""

    rpm: int = 40
    window_s: float = 60.0
    # Keep a safety margin: bursts of concurrent sub-calls otherwise land on the
    # boundary of the window and trip the cap anyway.
    safety: float = 0.85

    _times: deque = field(default_factory=deque, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    waits: int = 0
    total_wait_s: float = 0.0
    admitted: int = 0

    @property
    def effective_rpm(self) -> int:
        return max(1, int(self.rpm * self.safety))

    def acquire(self) -> None:
        """Block until a request may be issued."""
        while True:
            with self._lock:
                now = time.monotonic()
                cutoff = now - self.window_s
                while self._times and self._times[0] < cutoff:
                    self._times.popleft()
                if len(self._times) < self.effective_rpm:
                    self._times.append(now)
                    self.admitted += 1
                    return
                sleep_for = self._times[0] + self.window_s - now
                self.waits += 1
            sleep_for = max(0.05, min(sleep_for, self.window_s))
            self.total_wait_s += sleep_for
            time.sleep(sleep_for)

    def stats(self) -> dict[str, Any]:
        return {
            "rpm_cap": self.rpm,
            "effective_rpm": self.effective_rpm,
            "admitted": self.admitted,
            "waits": self.waits,
            "total_wait_s": round(self.total_wait_s, 1),
        }


@dataclass
class RetryPolicy:
    max_attempts: int = 5
    base_delay: float = 3.0
    max_delay: float = 60.0
    jitter: float = 0.3

    def delay_for(self, attempt: int) -> float:
        d = min(self.max_delay, self.base_delay * (2 ** attempt))
        return d * (1.0 + random.uniform(-self.jitter, self.jitter))


# Errors worth retrying on a free endpoint. Matched by class name so this module
# does not have to import the openai exception hierarchy.
_RETRYABLE_NAMES = {
    "RateLimitError",
    "APITimeoutError",
    "APIConnectionError",
    "APIStatusError",
    "InternalServerError",
    "ServiceUnavailableError",
    "APIError",
    "ConnectionError",
    "ReadTimeout",
    "Timeout",
}

_RETRYABLE_TEXT = (
    "429", "rate limit", "too many requests", "timed out", "timeout",
    "502", "503", "504", "overloaded", "capacity", "connection reset",
    "temporarily unavailable", "bad gateway",
)


def is_retryable(exc: BaseException) -> bool:
    name = type(exc).__name__
    if name in _RETRYABLE_NAMES:
        # A 4xx that is not 429 will not succeed on retry.
        status = getattr(exc, "status_code", None)
        if status is not None and 400 <= int(status) < 500 and int(status) != 429:
            return False
        return True
    text = str(exc).lower()
    return any(t in text for t in _RETRYABLE_TEXT)


@dataclass
class CallStats:
    calls: int = 0
    retries: int = 0
    failures: int = 0
    last_error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "calls": self.calls,
            "retries": self.retries,
            "failures": self.failures,
            "last_error": self.last_error,
        }


STATS = CallStats()
LIMITER = RateLimiter()
POLICY = RetryPolicy()
_installed = False
_install_lock = threading.Lock()


def configure(rpm: int | None = None, max_attempts: int | None = None,
              base_delay: float | None = None) -> None:
    if rpm is not None:
        LIMITER.rpm = rpm
    if max_attempts is not None:
        POLICY.max_attempts = max_attempts
    if base_delay is not None:
        POLICY.base_delay = base_delay


def guard(fn: Callable[..., Any], label: str = "") -> Callable[..., Any]:
    """Wrap a callable with the limiter and the retry policy."""

    def wrapped(*args: Any, **kwargs: Any) -> Any:
        last: BaseException | None = None
        for attempt in range(POLICY.max_attempts):
            LIMITER.acquire()
            try:
                result = fn(*args, **kwargs)
                STATS.calls += 1
                return result
            except BaseException as exc:  # noqa: BLE001 - re-raised below
                last = exc
                STATS.last_error = type(exc).__name__ + ": " + str(exc)[:200]
                if not is_retryable(exc) or attempt == POLICY.max_attempts - 1:
                    break
                STATS.retries += 1
                time.sleep(POLICY.delay_for(attempt))
        STATS.failures += 1
        assert last is not None
        raise last

    return wrapped


def install(rpm: int | None = None, max_attempts: int | None = None,
            base_delay: float | None = None) -> bool:
    """Patch rlm's OpenAI client so EVERY call in the tree is governed.

    Idempotent. Returns True if the patch was applied by this call.
    """
    global _installed
    configure(rpm=rpm, max_attempts=max_attempts, base_delay=base_delay)
    with _install_lock:
        if _installed:
            return False
        try:
            from rlm.clients.openai import OpenAIClient
        except ImportError:
            return False
        OpenAIClient.completion = guard(OpenAIClient.completion, "completion")  # type: ignore[method-assign]
        if hasattr(OpenAIClient, "acompletion"):
            # async variant is left alone: awaiting inside a sync guard would
            # block the event loop, and the driver never uses it.
            pass
        _installed = True
        return True


def stats() -> dict[str, Any]:
    return {**STATS.to_dict(), **LIMITER.stats()}


def reset() -> None:
    global STATS
    STATS = CallStats()
    LIMITER._times.clear()
    LIMITER.admitted = 0
    LIMITER.waits = 0
    LIMITER.total_wait_s = 0.0
