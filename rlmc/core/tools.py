"""Custom tools injected into a build node's REPL.

These are handed to ``RLM(custom_tools=...)``, which puts them straight into the
REPL globals (rlm/environments/local_repl.py). Each tool is a closure bound to
one node, so ownership is resolved per call without the model being able to
address another node's files.

The important tool is ``check_contract``: it gives a node the *same*
deterministic checker the harness will use to accept or reject its work. That
turns verification from something done to the node into something the node can
do to itself, which is what lets a node converge before returning rather than
burning a repair round.
"""

from __future__ import annotations

import io
import subprocess
import sys
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass, field
from typing import Any

from rlmc.contracts.checker import verify_node
from rlmc.contracts.imports import SymbolOwnership
from rlmc.contracts.types import CheckReport, Contract
from rlmc.core.workspace import Workspace

MAX_TOOL_OUTPUT = 4000


@dataclass
class NodeToolbox:
    """Tools for one node, plus the counters the harness reads back."""

    node_id: str
    contract: Contract
    workspace: Workspace
    ownership: SymbolOwnership
    contract_mode: str = "typed"
    strict_undeclared_imports: bool = True

    writes: int = 0
    checks: int = 0
    last_report: CheckReport | None = None
    denied: list[str] = field(default_factory=list)

    # ---- file io ------------------------------------------------------
    def write_file(self, path: str, content: str) -> str:
        """Write a file this node owns."""
        if not isinstance(content, str):
            content = str(content)
        ok, msg = self.workspace.write(self.node_id, path, content)
        if ok:
            self.writes += 1
        else:
            self.denied.append(path)
        return msg

    def read_file(self, path: str) -> str:
        """Read back a file this node wrote."""
        ok, msg = self.workspace.read(self.node_id, path)
        return msg if ok else msg

    def list_my_files(self) -> str:
        paths = self.workspace.written_paths(self.node_id)
        if not paths:
            return "You have not written any files yet."
        return "\n".join(paths)

    # ---- verification -------------------------------------------------
    def check_contract(self) -> str:
        """Run the deterministic contract checker over this node's files."""
        self.checks += 1
        files = self.workspace.files_of(self.node_id)
        if self.contract_mode == "none":
            # No declared interface to check against; report structure only so
            # the arm still gets syntax feedback and the comparison stays fair.
            broken = _syntax_errors(files)
            if broken:
                return "Syntax errors:\n" + "\n".join(broken)
            return "No contract declared for this run. Files parse: " + str(len(files))

        report = verify_node(
            self.contract,
            files,
            ownership=self.ownership,
            owner=self.node_id,
            extra_boundaries=["import_graph"],
        )
        self.last_report = report
        if report.ok:
            return (
                "CONTRACT OK -- " + str(report.checked) + " element(s) checked, "
                "0 violations."
            )
        lines = [
            "CONTRACT VIOLATIONS (" + str(len(report.violations)) + " of "
            + str(report.checked) + " checked, CVR="
            + str(round(report.cvr, 3)) + "):"
        ]
        for v in report.violations[:40]:
            lines.append("- " + v.kind + ": " + v.subject + " -- " + v.detail)
        if len(report.violations) > 40:
            lines.append("... +" + str(len(report.violations) - 40) + " more")
        return "\n".join(lines)

    def run_python(self, code: str, timeout: int = 60) -> str:
        """Execute a snippet in a subprocess rooted at the workspace.

        Runs out-of-process so an import error or a hang in generated code
        cannot take down the harness.
        """
        try:
            proc = subprocess.run(
                [sys.executable, "-c", code],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(self.workspace.root),
            )
        except subprocess.TimeoutExpired:
            return "TIMEOUT after " + str(timeout) + "s"
        except Exception as exc:
            return "failed to run: " + type(exc).__name__ + ": " + str(exc)
        out = (proc.stdout or "") + (proc.stderr or "")
        if len(out) > MAX_TOOL_OUTPUT:
            out = out[:MAX_TOOL_OUTPUT] + "\n... [truncated]"
        return out.strip() or ("exit " + str(proc.returncode) + ", no output")

    def workspace_tree(self) -> str:
        """Structure of the whole workspace. Paths and line counts only."""
        return self.workspace.tree()

    # ---- wiring -------------------------------------------------------
    def as_custom_tools(self) -> dict[str, Any]:
        """The dict handed to ``RLM(custom_tools=...)``."""
        tools: dict[str, Any] = {
            "write_file": {
                "tool": self.write_file,
                "description": (
                    "write_file(path, content) -> str. Write a file this node owns. "
                    "Refuses paths owned by other nodes."
                ),
            },
            "read_file": {
                "tool": self.read_file,
                "description": (
                    "read_file(path) -> str. Read back a file YOU wrote. You cannot "
                    "read another node's files."
                ),
            },
            "list_my_files": {
                "tool": self.list_my_files,
                "description": "list_my_files() -> str. Paths you have written.",
            },
            "run_python": {
                "tool": self.run_python,
                "description": (
                    "run_python(code) -> str. Run a snippet in the workspace to check "
                    "your code imports and behaves. Use it before finishing."
                ),
            },
            "workspace_tree": {
                "tool": self.workspace_tree,
                "description": (
                    "workspace_tree() -> str. Paths and line counts across the whole "
                    "workspace. Structure only, never file contents."
                ),
            },
        }
        tools["check_contract"] = {
            "tool": self.check_contract,
            "description": (
                "check_contract() -> str. Run the deterministic checker over your "
                "files and list every contract violation. This is the same checker "
                "that decides whether your work is accepted. Call it before you finish."
            ),
        }
        return tools


def _syntax_errors(files: dict[str, str]) -> list[str]:
    import ast

    out: list[str] = []
    for path, source in sorted(files.items()):
        if not path.endswith(".py"):
            continue
        try:
            ast.parse(source)
        except SyntaxError as e:
            out.append(path + ": line " + str(e.lineno) + ": " + str(e.msg))
    return out


def capture_stdio(fn: Any, *args: Any, **kwargs: Any) -> tuple[Any, str]:
    """Run ``fn`` capturing stdout/stderr. Used by the harness, not by nodes."""
    buf = io.StringIO()
    with redirect_stdout(buf), redirect_stderr(buf):
        result = fn(*args, **kwargs)
    return result, buf.getvalue()
