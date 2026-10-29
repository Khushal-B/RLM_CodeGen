"""Task loading.

One abstraction (``TaskSpec``) with two loaders behind it:

* ``load_smoke_tasks``   -- the local suite in ``rlmc/bench/smoke/``.
* ``load_nl2repo_tasks`` -- the real benchmark, once it is cloned.

Both produce identical ``TaskSpec`` objects, so switching benchmarks is a
command-line flag and nothing in the agents or the evaluator changes.
"""

from __future__ import annotations

import json
from pathlib import Path

from rlmc.agents.base import TaskSpec

SMOKE_DIR = Path(__file__).parent / "smoke"


def load_smoke_tasks(task_ids: list[str] | None = None) -> list[TaskSpec]:
    """Load the local smoke suite.

    Each task is a directory containing ``task.json``, ``requirements.md`` and
    ``tests/``. Deliberately the same shape as an NL2Repo task, so the adapter
    below is a data mapping rather than a code change.
    """
    tasks: list[TaskSpec] = []
    if not SMOKE_DIR.exists():
        return tasks
    for d in sorted(SMOKE_DIR.iterdir()):
        if not d.is_dir() or not (d / "task.json").exists():
            continue
        meta = json.loads((d / "task.json").read_text(encoding="utf-8"))
        tid = meta.get("task_id", d.name)
        if task_ids and tid not in task_ids:
            continue
        req_path = d / meta.get("requirements_file", "requirements.md")
        tasks.append(
            TaskSpec(
                task_id=tid,
                requirements=req_path.read_text(encoding="utf-8"),
                package_name=meta.get("package_name", tid.replace("-", "_")),
                difficulty=meta.get("difficulty", "easy"),
                target_loc=int(meta.get("target_loc", 0)),
                tests_path=str(d / meta.get("tests_dir", "tests")),
                metadata={"source": "smoke", "dir": str(d)},
            )
        )
    return tasks


def load_nl2repo_tasks(
    root: str | Path, task_ids: list[str] | None = None, difficulty: str | None = None
) -> list[TaskSpec]:
    """Load NL2Repo-Bench tasks from a clone of the benchmark repo.

    The real layout (verified against multimodal-art-projection/NL2RepoBench) is
    one directory per task under ``test_files/``:

        test_files/<name>/start.md              the requirements document
        test_files/<name>/test_case_count.txt   expected number of test cases
        test_files/<name>/test_commands.json    ["pip install -e .", "pytest ..."]
        test_files/<name>/test_files.json       ["tests"]

    The upstream test suites themselves are NOT in the repo: the official
    harness runs each task in a per-task Docker image tagged ``<name>:1.0`` that
    carries them. ``tests_path`` is therefore left None unless a local suite has
    been provisioned (see ``rlmc.bench.nl2repo_tests``), and scoring falls back
    to the structural metrics until one is.

    Difficulty is not labelled in the repo, so it is derived from the test-case
    count, which is the closest available proxy for the benchmark's own
    LOC-based grading.
    """
    root = Path(root)
    base = root / "test_files"
    if not base.exists():
        # Tolerate being pointed straight at test_files/.
        base = root
    if not base.exists():
        raise FileNotFoundError("NL2Repo-Bench test_files not found under " + str(root))

    tasks: list[TaskSpec] = []
    for d in sorted(base.iterdir()):
        if not d.is_dir():
            continue
        md = d / "start.md"
        if not md.exists():
            continue
        name = d.name
        if task_ids and name not in task_ids:
            continue

        count = 0
        count_file = d / "test_case_count.txt"
        if count_file.exists():
            try:
                count = int(count_file.read_text(encoding="utf-8").strip())
            except ValueError:
                count = 0

        commands = _read_json(d / "test_commands.json") or []
        test_dirs = _read_json(d / "test_files.json") or ["tests"]
        diff = nl2repo_difficulty(count)
        if difficulty and diff != difficulty.lower():
            continue

        local_tests = d / "tests"

        tasks.append(
            TaskSpec(
                task_id=name,
                requirements=md.read_text(encoding="utf-8", errors="replace"),
                package_name=_package_name_for(name),
                difficulty=diff,
                target_loc=0,
                test_command=commands,
                tests_path=str(local_tests) if local_tests.exists() else None,
                metadata={
                    "source": "nl2repo",
                    "test_case_count": count,
                    "test_dirs": test_dirs,
                    "task_dir": str(d),
                    "docker_image": name + ":1.0",
                },
            )
        )
    return tasks


def nl2repo_difficulty(test_case_count: int) -> str:
    """Proxy for the benchmark's LOC-based grading.

    NL2Repo grades Easy <1.5K LOC / Medium 1.5-4K / Hard >4K, but the clone does
    not ship LOC labels. Test-case count is monotone in repository size across
    the suite, so it is used as the stand-in and the thresholds are recorded
    here rather than buried in a caller.
    """
    if test_case_count <= 0:
        return "unknown"
    if test_case_count < 30:
        return "easy"
    if test_case_count < 100:
        return "medium"
    return "hard"


def _package_name_for(task_name: str) -> str:
    """Importable package name for a task directory name."""
    return task_name.replace("-", "_").replace(".", "_")


def _read_json(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _read_records(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8", errors="replace")
    if path.suffix == ".jsonl":
        out = []
        for line in text.splitlines():
            line = line.strip()
            if line:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    out.append(obj)
        return out
    obj = json.loads(text)
    if isinstance(obj, list):
        return [o for o in obj if isinstance(o, dict)]
    if isinstance(obj, dict):
        for key in ("tasks", "instances", "data", "examples"):
            if isinstance(obj.get(key), list):
                return [o for o in obj[key] if isinstance(o, dict)]
    return []


def _first(d: dict, *keys: str):
    for k in keys:
        if k in d and d[k]:
            return d[k]
    return None


def stratified_subset(
    tasks: list[TaskSpec], easy: int = 5, medium: int = 8, hard: int = 7, seed: int = 0
) -> list[TaskSpec]:
    """Freeze a stratified dev subset.

    The validation plan calls for 5 easy / 8 medium / 7 hard so the predicted
    Easy-vs-Hard crossover is visible, and for the subset to be frozen on day
    one and never tuned against the rest. Selection is by sorted task id under a
    fixed seed, so the same call always returns the same subset.
    """
    import random

    rng = random.Random(seed)
    buckets: dict[str, list[TaskSpec]] = {"easy": [], "medium": [], "hard": []}
    for t in tasks:
        buckets.setdefault(t.difficulty, []).append(t)

    out: list[TaskSpec] = []
    for name, n in (("easy", easy), ("medium", medium), ("hard", hard)):
        pool = sorted(buckets.get(name, []), key=lambda t: t.task_id)
        out.extend(pool if len(pool) <= n else rng.sample(pool, n))
    return sorted(out, key=lambda t: t.task_id)
