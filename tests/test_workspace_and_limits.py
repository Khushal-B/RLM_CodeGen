"""Regression tests for workspace scaffolding races and the rate limiter.

The scaffolding tests exist because of a real, nondeterministic bug: two
concurrent siblings both created ``pkg/__init__.py``, the harness marked it as
scaffolding, and the node that legitimately owned it was then charged
``missing_symbol`` for every symbol it was supposed to export -- but only
sometimes, depending on which sibling finished first. That made CVR
nondeterministic, which would have silently corrupted the H1 comparison.
"""

from __future__ import annotations

import sys
import tempfile
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rlmc.agents import ratelimit
from rlmc.contracts.checker import verify_node
from rlmc.contracts.types import Contract, SymbolSpec
from rlmc.core.workspace import Workspace, ensure_package_files


def _ws() -> Workspace:
    return Workspace(tempfile.mkdtemp(prefix="rlmc_ws_"))


# --- scaffolding vs ownership --------------------------------------------

def test_scaffolding_never_preempts_a_claimed_file():
    ws = _ws()
    ws.grant("core", ["pkg/core.py"])
    ws.grant("api", ["pkg/__init__.py"])       # api owns the package init
    ws.write("core", "pkg/core.py", "X = 1\n")

    # core finishes first and asks for package scaffolding
    created = ensure_package_files(ws, "core", ws.written_paths("core"))
    assert "pkg/__init__.py" not in created

    # api can still write its own file, and it counts as api's product
    ok, _ = ws.write("api", "pkg/__init__.py", "from pkg.core import X\n")
    assert ok
    assert "pkg/__init__.py" in ws.files_of("api")


def test_owner_write_supersedes_scaffolding():
    ws = _ws()
    ws.grant("a", ["pkg/mod.py"])
    ws.write("a", "pkg/mod.py", "X = 1\n")
    ensure_package_files(ws, "a", ws.written_paths("a"))
    assert "pkg/__init__.py" not in ws.files_of("a")   # scaffolding, not product

    ws.grant("a", ["pkg/__init__.py"])
    ws.write("a", "pkg/__init__.py", "from pkg.mod import X\n")
    assert "pkg/__init__.py" in ws.files_of("a")       # now it is product


def test_concurrent_scaffolding_is_deterministic():
    """Ten siblings racing to scaffold the same package must agree."""
    for _ in range(5):
        ws = _ws()
        for i in range(10):
            ws.grant("n%d" % i, ["pkg/m%d.py" % i])
            ws.write("n%d" % i, "pkg/m%d.py" % i, "X = 1\n")

        created: list[list[str]] = []
        lock = threading.Lock()

        def go(i: int) -> None:
            c = ensure_package_files(ws, "n%d" % i, ws.written_paths("n%d" % i))
            with lock:
                created.append(c)

        threads = [threading.Thread(target=go, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Exactly one node creates it, exactly once.
        total = [p for c in created for p in c]
        assert total.count("pkg/__init__.py") == 1


def test_cvr_is_deterministic_under_scaffolding():
    """The bug's actual symptom: a node blamed for a file it did write."""
    for _ in range(5):
        ws = _ws()
        api = Contract(
            node_id="api",
            owns_paths=["pkg/__init__.py"],
            provides=[SymbolSpec("pkg.run", "function", "pkg/__init__.py")],
        )
        ws.grant("api", api.owns_paths)
        ws.grant("core", ["pkg/core.py"])
        ws.write("core", "pkg/core.py", "Y = 1\n")
        ensure_package_files(ws, "core", ws.written_paths("core"))
        ws.write("api", "pkg/__init__.py", "def run():\n    return 1\n")

        report = verify_node(api, ws.files_of("api"), owner="api")
        assert report.ok, [v.brief() for v in report.violations]


def test_is_claimed():
    ws = _ws()
    ws.grant("a", ["pkg/a.py"])
    assert ws.is_claimed("pkg/a.py")
    assert not ws.is_claimed("pkg/b.py")


# --- rate limiter ---------------------------------------------------------

def test_limiter_admits_up_to_the_cap_without_waiting():
    lim = ratelimit.RateLimiter(rpm=60, window_s=60.0, safety=1.0)
    start = time.monotonic()
    for _ in range(10):
        lim.acquire()
    assert time.monotonic() - start < 0.5
    assert lim.admitted == 10


def test_limiter_blocks_past_the_cap():
    lim = ratelimit.RateLimiter(rpm=4, window_s=1.0, safety=1.0)
    for _ in range(4):
        lim.acquire()
    start = time.monotonic()
    lim.acquire()          # must wait for the window to roll
    assert time.monotonic() - start > 0.3
    assert lim.waits >= 1


def test_limiter_is_thread_safe():
    lim = ratelimit.RateLimiter(rpm=100, window_s=60.0, safety=1.0)

    def go() -> None:
        for _ in range(10):
            lim.acquire()

    threads = [threading.Thread(target=go) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert lim.admitted == 50


def test_safety_margin_reduces_effective_rate():
    assert ratelimit.RateLimiter(rpm=40, safety=0.85).effective_rpm == 34


# --- retry ----------------------------------------------------------------

def test_retryable_classification():
    class RateLimitError(Exception):
        pass

    class BadRequestError(Exception):
        status_code = 400

    assert ratelimit.is_retryable(RateLimitError("429 too many requests"))
    assert ratelimit.is_retryable(TimeoutError("request timed out"))
    assert ratelimit.is_retryable(Exception("503 service unavailable"))
    assert not ratelimit.is_retryable(BadRequestError("bad request"))
    assert not ratelimit.is_retryable(ValueError("malformed contract"))


def test_guard_retries_then_succeeds():
    ratelimit.reset()
    ratelimit.configure(rpm=10_000, max_attempts=4, base_delay=0.01)
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise TimeoutError("timed out")
        return "ok"

    assert ratelimit.guard(flaky)() == "ok"
    assert calls["n"] == 3
    assert ratelimit.STATS.retries >= 2


def test_guard_gives_up_and_reraises():
    ratelimit.reset()
    ratelimit.configure(rpm=10_000, max_attempts=2, base_delay=0.01)

    def always():
        raise TimeoutError("timed out")

    with pytest.raises(TimeoutError):
        ratelimit.guard(always)()
    assert ratelimit.STATS.failures >= 1


def test_guard_does_not_retry_a_permanent_error():
    ratelimit.reset()
    ratelimit.configure(rpm=10_000, max_attempts=5, base_delay=0.01)
    calls = {"n": 0}

    def bad():
        calls["n"] += 1
        raise ValueError("contract is malformed")

    with pytest.raises(ValueError):
        ratelimit.guard(bad)()
    assert calls["n"] == 1


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
