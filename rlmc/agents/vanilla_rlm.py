"""Baseline arms: unmodified RLM, and a plain single call.

These are the control rows. Neither knows anything about contracts; both write
into the same workspace and produce the same artifact, so the evaluator scores
them identically to the contract arm.

``VanillaRLMAgent`` is the paper's method applied to generation: one RLM with a
REPL, file-writing tools, and no contract system, no decomposition discipline
and no bounded returns. It is the "RLM(depth=k) monolithic" row the plan calls
for, and it is the honest reference point for what the contract machinery
actually buys.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from rlmc.agents.backend import LMBackend
from rlmc.agents.base import RunRecord, TaskSpec, environment_stamp
from rlmc.config import ArmConfig
from rlmc.contracts.imports import SymbolOwnership
from rlmc.contracts.types import Contract
from rlmc.core.tools import NodeToolbox
from rlmc.core.workspace import Workspace, ensure_package_files
from rlmc.metrics.telemetry import RunTelemetry, context_high_water

_VANILLA_PROMPT = """You are building a complete, installable Python package into a
workspace, starting from an empty directory.

Available in the REPL (write ```repl blocks; the REPL persists across turns):
- `spec`: the requirements document.
- `write_file(path, content) -> str`: write a file into the workspace.
- `read_file(path) -> str`: read a file back.
- `list_my_files() -> str`: what you have written so far.
- `run_python(code) -> str`: run a snippet against the workspace to check it works.
- `workspace_tree() -> str`: paths and line counts.
- `llm_query(prompt)` / `llm_query_batched(prompts)`: sub-LLM calls. Use these to
  draft individual modules, then write the results with `write_file`.
- `SHOW_VARS()`, `answer`: submit by setting `answer["content"]` and
  `answer["ready"] = True`.

Plan the package, then build it module by module. Verify with `run_python` that
your package imports and behaves before finishing. Do not finish with any
function left as `pass`, `TODO` or `NotImplementedError`.
"""

_BASE_LM_PROMPT = """You write complete Python packages in a single response.

Return the entire package as a sequence of files, each in this exact format:

### path/to/file.py
```python
<complete file contents>
```

Include every file needed for an installable, working package. No placeholders,
no TODOs, no omitted function bodies.
"""


class VanillaRLMAgent:
    """Unmodified RLM with file tools. No contracts, no bounded returns."""

    def __init__(self, config: ArmConfig) -> None:
        self.cfg = config
        self.name = config.name

    def run(self, task: TaskSpec, workspace_dir: str, seed: int) -> RunRecord:
        t0 = time.perf_counter()
        cfg = self.cfg.copy_with(seed=seed) if self.cfg.seed is None else self.cfg

        ws = Workspace(workspace_dir, enforce=False)  # no ownership discipline
        ws.reset()
        telemetry = RunTelemetry()
        lm = LMBackend(cfg)

        package = task.package_name or "generated_pkg"
        contract = Contract.empty("root")
        toolbox = NodeToolbox(
            node_id="root",
            contract=contract,
            workspace=ws,
            ownership=SymbolOwnership.from_contracts([contract], scope_complete=False),
            contract_mode="none",
        )
        tools = toolbox.as_custom_tools()
        tools.pop("check_contract", None)  # the baseline has no contract to check

        spec = (
            "Build an installable Python package. The importable top-level "
            "package MUST be named `" + package + "`.\n\n" + task.requirements
        )

        error: str | None = None
        trajectory: dict[str, Any] = {}
        try:
            trajectory = lm.run_rlm_node(
                system_prompt=_VANILLA_PROMPT,
                context=spec,
                tools=tools,
                max_iterations=cfg.root_max_iterations,
                node_id="root",
                budget_usd=cfg.max_budget_usd,
            )
            telemetry.record_call(trajectory.get("usage", {}))
        except Exception as exc:
            error = type(exc).__name__ + ": " + str(exc)

        ensure_package_files(ws, "root", ws.written_paths("root"))
        ctx = context_high_water(trajectory)
        telemetry.record_context("root", 0, ctx)
        stats = ws.stats()

        return RunRecord(
            task_id=task.task_id,
            arm=cfg.name,
            seed=seed,
            ok=stats["py_files"] > 0 and error is None,
            completed=stats["py_files"] > 0,
            workspace=str(Path(workspace_dir).resolve()),
            wallclock_s=time.perf_counter() - t0,
            cost_usd=telemetry.total_cost_usd,
            root_context_tokens=ctx,
            peak_context_tokens=ctx,
            loc=stats["loc"],
            files=stats["py_files"],
            nodes=1,
            max_depth=0,
            error=error,
            config=cfg.to_dict(),
            telemetry={**telemetry.to_dict(), "workspace": stats,
                       "iterations": trajectory.get("iterations", 0),
                       "subcalls": trajectory.get("subcalls", 0)},
            env=environment_stamp(),
            started_at=t0,
        )


class BaseLMAgent:
    """Single completion, no REPL, no recursion. The lower bound."""

    def __init__(self, config: ArmConfig) -> None:
        self.cfg = config
        self.name = config.name

    def run(self, task: TaskSpec, workspace_dir: str, seed: int) -> RunRecord:
        t0 = time.perf_counter()
        cfg = self.cfg.copy_with(seed=seed) if self.cfg.seed is None else self.cfg

        ws = Workspace(workspace_dir, enforce=False)
        ws.reset()
        telemetry = RunTelemetry()
        lm = LMBackend(cfg)

        package = task.package_name or "generated_pkg"
        user = (
            "The importable top-level package MUST be named `" + package + "`.\n\n"
            + task.requirements
        )

        error: str | None = None
        text = ""
        try:
            text, usage = lm.complete(_BASE_LM_PROMPT, user, node_id="root")
            telemetry.record_call(usage)
        except Exception as exc:
            error = type(exc).__name__ + ": " + str(exc)

        for path, content in parse_file_blocks(text).items():
            ws.write("root", path, content)
        ensure_package_files(ws, "root", ws.written_paths("root"))

        ctx = 0
        try:
            from rlmc.metrics.telemetry import count_tokens

            ctx = count_tokens(_BASE_LM_PROMPT + user)
        except Exception:
            pass
        telemetry.record_context("root", 0, ctx)
        stats = ws.stats()

        return RunRecord(
            task_id=task.task_id,
            arm=cfg.name,
            seed=seed,
            ok=stats["py_files"] > 0 and error is None,
            completed=stats["py_files"] > 0,
            workspace=str(Path(workspace_dir).resolve()),
            wallclock_s=time.perf_counter() - t0,
            cost_usd=telemetry.total_cost_usd,
            root_context_tokens=ctx,
            peak_context_tokens=ctx,
            loc=stats["loc"],
            files=stats["py_files"],
            nodes=1,
            max_depth=0,
            error=error,
            config=cfg.to_dict(),
            telemetry={**telemetry.to_dict(), "workspace": stats},
            env=environment_stamp(),
            started_at=t0,
        )


def parse_file_blocks(text: str) -> dict[str, str]:
    """Extract ``### path`` + fenced-code pairs from a single-shot response."""
    import re

    out: dict[str, str] = {}
    if not text:
        return out
    pattern = re.compile(
        r"^#{1,4}\s*`?([\w./\\-]+\.\w+)`?\s*\n+```[a-zA-Z]*\n(.*?)\n```",
        re.DOTALL | re.MULTILINE,
    )
    for match in pattern.finditer(text):
        path = match.group(1).replace("\\", "/").strip("/")
        out[path] = match.group(2)
    return out
