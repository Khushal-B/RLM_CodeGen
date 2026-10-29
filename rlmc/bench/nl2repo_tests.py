"""Provisioning upstream test suites for NL2Repo-Bench without Docker.

Why this exists
---------------
NL2Repo-Bench scores a generated repository by running the ORIGINAL upstream
pytest suite. Those suites are not in the benchmark clone: the official harness
runs each task inside a per-task Docker image tagged ``<task>:1.0`` which
carries them. Without Docker there is no suite, and therefore no execution-
verified score.

The suites are, however, public: most of these projects ship their tests inside
their PyPI source distribution. Fetching the sdist for a pinned version and
extracting its ``tests/`` directory reconstructs the same artifact the Docker
image would have provided, from the same upstream source.

Honesty notes -- these matter for how the numbers may be reported
----------------------------------------------------------------
* The version pinned here may not be the exact revision the benchmark authors
  froze. Where the extracted test count differs from the task's
  ``test_case_count.txt``, ``provision`` reports the discrepancy rather than
  hiding it, and the run record keeps both numbers.
* Scores produced this way are valid for comparing ARMS AGAINST EACH OTHER
  (both arms run the identical suite), which is what H1 needs. They should not
  be quoted against the benchmark's published leaderboard numbers without
  reproducing under the official Docker harness.
* The agent never sees these files: they are fetched into the benchmark task
  directory, not into the workspace, and copied in only at evaluation time.
"""

from __future__ import annotations

import io
import json
import shutil
import tarfile
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

PYPI_JSON = "https://pypi.org/pypi/{pkg}/json"

# Curated task -> PyPI project mapping. Only tasks whose upstream sdist actually
# ships its test suite are listed; add more as they are verified. A pinned
# version keeps the suite stable across runs -- "latest" would silently change
# the denominator between experiments.
KNOWN_SOURCES: dict[str, dict[str, str]] = {
    "jsonlines": {"pypi": "jsonlines", "version": "4.0.0"},
    "markupsafe": {"pypi": "MarkupSafe", "version": "3.0.3"},
    "plac": {"pypi": "plac", "version": "1.4.6"},
    "boltons": {"pypi": "boltons", "version": "25.0.0"},
    "cerberus": {"pypi": "Cerberus", "version": "1.3.7"},
    "bleach": {"pypi": "bleach", "version": "6.2.0"},
    "asteval": {"pypi": "asteval", "version": "1.0.6"},
    "records": {"pypi": "records", "version": "0.6.0"},
    "pyperclip": {"pypi": "pyperclip", "version": "1.11.0"},
}


@dataclass
class ProvisionResult:
    task_id: str
    ok: bool
    tests_dir: str | None = None
    files: int = 0
    collected: int = 0          # tests pytest can actually collect
    expected: int = 0           # test_case_count.txt
    note: str = ""

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def _fetch_sdist(pkg: str, version: str | None) -> tuple[bytes, str, str]:
    """Return (raw_bytes, filename, resolved_version) for a project's sdist."""
    meta = json.load(urllib.request.urlopen(PYPI_JSON.format(pkg=pkg), timeout=60))
    resolved = version or meta["info"]["version"]
    releases = meta.get("releases", {})
    urls = releases.get(resolved) or []
    if not urls:
        urls = meta["urls"]
        resolved = meta["info"]["version"]
    sdists = [u for u in urls if u.get("packagetype") == "sdist"]
    if not sdists:
        raise RuntimeError("no sdist published for " + pkg + " " + resolved)
    raw = urllib.request.urlopen(sdists[0]["url"], timeout=180).read()
    return raw, sdists[0]["filename"], resolved


def _members(raw: bytes, filename: str):
    if filename.endswith(".zip"):
        zf = zipfile.ZipFile(io.BytesIO(raw))
        for name in zf.namelist():
            yield name, (lambda n=name: zf.read(n))
    else:
        tf = tarfile.open(fileobj=io.BytesIO(raw))
        for member in tf.getmembers():
            if member.isfile():
                yield member.name, (
                    lambda m=member: (tf.extractfile(m).read() if tf.extractfile(m) else b"")
                )


def provision(
    task_dir: str | Path,
    task_id: str,
    pypi: str | None = None,
    version: str | None = None,
    force: bool = False,
) -> ProvisionResult:
    """Extract a task's upstream ``tests/`` into ``<task_dir>/tests``."""
    task_dir = Path(task_dir)
    dest = task_dir / "tests"

    expected = 0
    count_file = task_dir / "test_case_count.txt"
    if count_file.exists():
        try:
            expected = int(count_file.read_text(encoding="utf-8").strip())
        except ValueError:
            expected = 0

    if dest.exists() and not force:
        n = len(list(dest.rglob("*.py")))
        return ProvisionResult(task_id, True, str(dest), n, 0, expected, "already present")

    source = KNOWN_SOURCES.get(task_id)
    pkg = pypi or (source or {}).get("pypi")
    ver = version or (source or {}).get("version")
    if not pkg:
        return ProvisionResult(
            task_id, False, None, 0, 0, expected,
            "no PyPI mapping known; add one to KNOWN_SOURCES or pass --pypi",
        )

    try:
        raw, filename, resolved = _fetch_sdist(pkg, ver)
    except Exception as exc:
        return ProvisionResult(task_id, False, None, 0, 0, expected,
                               "fetch failed: " + type(exc).__name__ + ": " + str(exc)[:120])

    written = 0
    if dest.exists():
        shutil.rmtree(dest, ignore_errors=True)
    dest.mkdir(parents=True, exist_ok=True)

    for name, read in _members(raw, filename):
        parts = name.replace("\\", "/").split("/")
        # <pkg-version>/tests/... or <pkg-version>/test/...
        if len(parts) < 3 or parts[1] not in ("tests", "test"):
            continue
        rel = "/".join(parts[2:])
        if not rel:
            continue
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            target.write_bytes(read())
            written += 1
        except Exception:
            continue

    if written == 0:
        shutil.rmtree(dest, ignore_errors=True)
        return ProvisionResult(task_id, False, None, 0, 0, expected,
                               pkg + " " + resolved + " sdist ships no tests/ directory")

    note = "from " + pkg + " " + resolved
    return ProvisionResult(task_id, True, str(dest), written, 0, expected, note)


def count_collectable(tests_dir: str | Path, timeout: int = 120) -> int:
    """How many tests pytest can collect from a suite, in isolation.

    Reported so a mismatch against ``test_case_count.txt`` is visible rather
    than silently changing the denominator of every pass rate.
    """
    import re
    import subprocess
    import sys

    try:
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", "--collect-only", "-q",
             "-p", "no:cacheprovider", str(tests_dir)],
            capture_output=True, text=True, timeout=timeout,
        )
    except Exception:
        return 0
    out = (proc.stdout or "") + (proc.stderr or "")
    m = re.search(r"(\d+)\s+tests? collected", out)
    if m:
        return int(m.group(1))
    m = re.search(r"(\d+)/\d+ tests collected", out)
    return int(m.group(1)) if m else 0


def provision_many(
    nl2repo_root: str | Path, task_ids: list[str], force: bool = False
) -> list[ProvisionResult]:
    root = Path(nl2repo_root)
    base = root / "test_files"
    if not base.exists():
        base = root
    results: list[ProvisionResult] = []
    for tid in task_ids:
        d = base / tid
        if not d.exists():
            results.append(ProvisionResult(tid, False, note="task directory not found"))
            continue
        res = provision(d, tid, force=force)
        if res.ok and res.tests_dir:
            res.collected = count_collectable(res.tests_dir)
            if res.expected and res.collected and abs(res.collected - res.expected) > 0:
                res.note += (
                    "; collected " + str(res.collected) + " vs benchmark's "
                    + str(res.expected) + " -- version drift, comparable across "
                    "arms but not against published numbers"
                )
        results.append(res)
    return results
