"""Execution-based evaluation.

Scoring runs the task's real pytest suite against the generated workspace and
reports the fraction of tests that pass. No LLM judge is involved anywhere.

Two deliberate choices, both from the validation plan:

* **Average test pass rate, not binary success.** Problem B: with n=20 and a
  binary outcome at p=0.25 the standard error is about 7.7 points, so a real
  5-point effect is invisible. A continuous per-task pass rate has far lower
  variance and is what NL2Repo-Bench itself reports.
* **A failed run scores 0, it is not dropped.** Dropping failures inflates the
  arm that crashes more often. Scoring 0 is both the honest choice and the one
  that favours a method claiming to eliminate non-finish.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class EvalResult:
    tests_total: int = 0
    tests_passed: int = 0
    pass_rate: float = 0.0
    all_tests_pass: bool = False
    collected: bool = False       # did pytest manage to collect the suite at all
    error: str | None = None
    stdout_tail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "tests_total": self.tests_total,
            "tests_passed": self.tests_passed,
            "pass_rate": self.pass_rate,
            "all_tests_pass": self.all_tests_pass,
            "collected": self.collected,
            "error": self.error,
            "stdout_tail": self.stdout_tail,
        }


def evaluate_workspace(
    workspace: str | Path,
    tests_path: str | Path,
    timeout: int = 300,
    package_name: str = "",
) -> EvalResult:
    """Run ``tests_path`` against the package in ``workspace``.

    Tests are copied into a scratch directory alongside the generated package so
    the suite cannot be modified by the agent and is identical across arms.
    """
    workspace = Path(workspace).resolve()
    tests_path = Path(tests_path).resolve()

    if not workspace.exists():
        return EvalResult(error="workspace does not exist: " + str(workspace))
    if not tests_path.exists():
        return EvalResult(error="tests not found: " + str(tests_path))

    scratch = Path(tempfile.mkdtemp(prefix="rlmc_eval_"))
    try:
        # Generated package first, then the pristine suite on top.
        for item in workspace.iterdir():
            if item.name in _SKIP:
                continue
            dest = scratch / item.name
            if item.is_dir():
                shutil.copytree(item, dest, dirs_exist_ok=True,
                                ignore=shutil.ignore_patterns(*_SKIP))
            else:
                shutil.copy2(item, dest)

        test_dest = scratch / "tests"
        if tests_path.is_dir():
            shutil.copytree(tests_path, test_dest, dirs_exist_ok=True)
        else:
            test_dest.mkdir(parents=True, exist_ok=True)
            shutil.copy2(tests_path, test_dest / tests_path.name)

        env = dict(os.environ)
        env["PYTHONPATH"] = str(scratch) + os.pathsep + env.get("PYTHONPATH", "")
        env["PYTHONDONTWRITEBYTECODE"] = "1"

        try:
            proc = subprocess.run(
                [sys.executable, "-m", "pytest", "tests", "-q", "--no-header",
                 "-p", "no:cacheprovider"],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(scratch),
                env=env,
            )
        except subprocess.TimeoutExpired:
            return EvalResult(error="pytest timed out after " + str(timeout) + "s")
        except FileNotFoundError:
            return EvalResult(error="pytest is not installed in this interpreter")

        out = (proc.stdout or "") + "\n" + (proc.stderr or "")
        return parse_pytest_output(out)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


_SKIP = ("__pycache__", ".git", ".pytest_cache", ".mypy_cache", ".venv", "node_modules")

# "12 passed, 3 failed, 1 error in 0.42s" -- order and membership vary.
_COUNT_RE = re.compile(r"(\d+)\s+(passed|failed|error|errors|skipped|xfailed|xpassed)")


def parse_pytest_output(text: str) -> EvalResult:
    """Turn pytest's summary line into counts.

    A collection failure (the package does not import at all) is reported as
    ``collected=False`` with a pass rate of 0 rather than as a missing datapoint
    -- that is exactly the non-finish case the thesis cares about measuring.
    """
    counts: dict[str, int] = {}
    for m in _COUNT_RE.finditer(text):
        key = m.group(2).rstrip("s")
        counts[key] = counts.get(key, 0) + int(m.group(1))

    passed = counts.get("passed", 0) + counts.get("xpassed", 0)
    failed = counts.get("failed", 0) + counts.get("error", 0)
    skipped = counts.get("skipped", 0) + counts.get("xfailed", 0)
    total = passed + failed + skipped

    collected = total > 0
    if not collected:
        err = None
        if "ImportError" in text or "ModuleNotFoundError" in text:
            err = "package did not import"
        elif "no tests ran" in text.lower():
            err = "no tests collected"
        elif "INTERNALERROR" in text:
            err = "pytest internal error"
        return EvalResult(
            collected=False,
            error=err or "no test results parsed",
            stdout_tail=text[-1500:],
        )

    denom = passed + failed  # skipped tests are excluded from the denominator
    rate = (passed / denom) if denom else 0.0
    return EvalResult(
        tests_total=total,
        tests_passed=passed,
        pass_rate=rate,
        all_tests_pass=(failed == 0 and passed > 0),
        collected=True,
        stdout_tail=text[-1500:],
    )


def importability(workspace: str | Path, package_name: str, timeout: int = 60) -> tuple[bool, str]:
    """Can the generated package be imported at all? The completion-rate check."""
    workspace = Path(workspace).resolve()
    if not package_name:
        return False, "no package name given"
    code = "import " + package_name + "; print('ok')"
    env = dict(os.environ)
    env["PYTHONPATH"] = str(workspace) + os.pathsep + env.get("PYTHONPATH", "")
    try:
        proc = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True, timeout=timeout,
            cwd=str(workspace), env=env,
        )
    except Exception as exc:
        return False, type(exc).__name__ + ": " + str(exc)
    return proc.returncode == 0, ((proc.stdout or "") + (proc.stderr or ""))[-500:]


def save_eval(record_path: str | Path, result: EvalResult) -> None:
    """Merge an evaluation into an existing run record on disk."""
    p = Path(record_path)
    data = json.loads(p.read_text(encoding="utf-8"))
    data["tests_total"] = result.tests_total
    data["tests_passed"] = result.tests_passed
    data["pass_rate"] = result.pass_rate
    data["all_tests_pass"] = result.all_tests_pass
    data["eval_error"] = result.error
    p.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
