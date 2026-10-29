"""The contract-recursive arm: RLM + contracts + bounded returns."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from rlmc.agents.backend import LMBackend
from rlmc.agents.base import RunRecord, TaskSpec, environment_stamp
from rlmc.config import ArmConfig
from rlmc.contracts.types import Contract
from rlmc.core.driver import BuildDriver
from rlmc.core.node import Budget, BuildNode, NodeKind
from rlmc.core.workspace import Workspace
from rlmc.metrics.telemetry import RunTelemetry, decomposition_shape


class ContractRLMAgent:
    """Runs one task through ``rlm_build``."""

    def __init__(self, config: ArmConfig, progress: bool = False) -> None:
        self.cfg = config
        self.name = config.name
        self.progress = progress

    def run(self, task: TaskSpec, workspace_dir: str, seed: int) -> RunRecord:
        t0 = time.perf_counter()
        cfg = self.cfg.copy_with(seed=seed) if self.cfg.seed is None else self.cfg

        ws = Workspace(workspace_dir)
        ws.reset()
        telemetry = RunTelemetry()
        lm = LMBackend(cfg)
        events: list[dict[str, Any]] = []

        def _on_event(kind: str, data: dict[str, Any]) -> None:
            events.append({"kind": kind, **data})
            # A run against a slow free endpoint can spend 15 minutes on a
            # single call; without this the terminal looks hung.
            if self.progress:
                _print_event(kind, data, t0)

        driver = BuildDriver(
            config=cfg,
            workspace=ws,
            lm=lm,
            telemetry=telemetry,
            on_event=_on_event,
        )

        package = task.package_name or _guess_package(task)
        root_contract = Contract(
            node_id="root",
            boundary="python_stub",
            provides=[],
            requires=[],
            # The root owns the whole tree; children narrow this on decomposition.
            owns_paths=[package, package + "/*", "*.py", "*.toml", "*.txt", "*.cfg", "*.md"],
        )
        ws.grant("root", root_contract.owns_paths)

        root = BuildNode(
            node_id="root",
            kind=NodeKind.REPO,
            title=task.task_id,
            spec=_root_spec(task, package),
            contract=root_contract,
            depth=0,
        )
        budget = Budget(
            tokens=200_000,
            usd=cfg.max_budget_usd,
            iterations=cfg.root_max_iterations,
            depth_remaining=cfg.max_depth,
        )

        error: str | None = None
        result = None
        try:
            result = driver.build(root, budget)
        except Exception as exc:
            error = type(exc).__name__ + ": " + str(exc)

        stats = ws.stats()
        report = result.report if result else None

        record = RunRecord(
            task_id=task.task_id,
            arm=cfg.name,
            seed=seed,
            ok=bool(result and result.ok),
            completed=stats["py_files"] > 0,
            workspace=str(Path(workspace_dir).resolve()),
            wallclock_s=time.perf_counter() - t0,
            cost_usd=telemetry.total_cost_usd,
            cvr=report.cvr if report else 0.0,
            checked=report.checked if report else 0,
            violations=len(report.violations) if report else 0,
            cvr_by_boundary=report.cvr_by_boundary() if report else {},
            root_context_tokens=telemetry.root_context_tokens,
            peak_context_tokens=telemetry.peak_context_tokens,
            max_return_tokens=driver.returns.max_tokens,
            max_return_tokens_unbounded=driver.returns.max_raw_tokens,
            bound_respected=(
                driver.returns.bound_respected(cfg.diagnostic_budget_tokens)
                if cfg.return_mode == "bounded"
                else True
            ),
            loc=stats["loc"],
            files=stats["py_files"],
            nodes=result.node_count() if result else 0,
            max_depth=result.max_depth_reached() if result else 0,
            error=error,
            config=cfg.to_dict(),
            telemetry={
                **telemetry.to_dict(),
                "returns": driver.returns.to_dict(),
                "workspace": stats,
                "shape": decomposition_shape(result),
                "events": events[-200:],
            },
            tree=result.to_dict() if result else {},
            env=environment_stamp(),
            started_at=t0,
        )
        return record


def _print_event(kind: str, data: dict[str, Any], t0: float) -> None:
    """One line per driver event, so a slow run shows a pulse."""
    elapsed = time.perf_counter() - t0
    stamp = "    [" + format(elapsed, "6.0f") + "s] "
    node = str(data.get("node_id", ""))
    if kind == "node_start":
        print(stamp + "-> " + node + " (depth " + str(data.get("depth", 0)) + ")", flush=True)
    elif kind == "node_done":
        print(stamp + "<- " + node + " " + str(data.get("action", ""))
              + (" ok" if data.get("ok") else " FAILED")
              + " payload=" + str(data.get("payload_tokens", 0)) + "t", flush=True)
    elif kind == "composition_rejected":
        print(stamp + "!! composition rejected at " + node + ": "
              + str(data.get("summary", ""))[:110], flush=True)
    elif kind == "repair":
        print(stamp + "~~ repair " + str(data.get("action", "")) + " -> "
              + ", ".join(data.get("targets") or []) + " (" + str(data.get("reason", ""))[:70]
              + ")", flush=True)
    elif kind == "decompose_failed":
        print(stamp + "!! decompose failed at " + node + ": "
              + str(data.get("error", ""))[:110], flush=True)
    elif kind == "node_error":
        print(stamp + "!! " + node + " error: " + str(data.get("error", ""))[:110], flush=True)


def _guess_package(task: TaskSpec) -> str:
    name = (task.package_name or task.task_id or "generated_pkg").strip()
    name = name.replace("-", "_").replace(" ", "_")
    return "".join(ch for ch in name if ch.isalnum() or ch == "_") or "generated_pkg"


def _root_spec(task: TaskSpec, package: str) -> str:
    """The requirements document, plus the packaging constraints.

    Packaging, import consistency and dependency declaration are three of the
    four failure categories NL2Repo-Bench reports, and none of them are
    interesting research -- so they are stated once, explicitly, rather than
    left to be rediscovered by every node.
    """
    return (
        "# Task\n\n"
        "Build a complete, installable Python package from the requirements below, "
        "starting from an empty workspace.\n\n"
        "## Hard constraints\n"
        "- The importable top-level package MUST be named `" + package + "`.\n"
        "- Every module must be inside `" + package + "/` and importable as "
        "`" + package + ".<module>`.\n"
        "- Public API declared in a contract must be importable from the module "
        "path the contract names.\n"
        "- Use only the Python standard library unless the requirements name a "
        "dependency explicitly.\n"
        "- No placeholders: every function a contract declares must be fully "
        "implemented.\n\n"
        "## Requirements document\n\n" + task.requirements
    )
