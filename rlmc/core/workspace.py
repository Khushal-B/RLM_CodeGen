"""The workspace: the repository as an external object.

The build target lives on disk, never in a model's context. A node interacts
with it only through the narrow API here, and the API enforces the two
structural rules that the rest of the design depends on:

1. A node may WRITE only files listed in its contract's ``owns_paths``.
2. A node may READ only files it owns. Sibling code is not readable -- a node
   is given its siblings' *contracts* instead. This is the filesystem-level
   half of the invariant; the payload module is the return-channel half.

Both refusals are recorded rather than silently ignored, so an arm that tries
to reach around the invariant shows up as a measurement
(``Workspace.denied_reads`` / ``denied_writes``) instead of quietly succeeding.
"""

from __future__ import annotations

import fnmatch
import os
import shutil
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class AccessDenial:
    node_id: str
    path: str
    op: str      # "read" | "write"
    reason: str


class Workspace:
    """A directory the build writes into, with per-node access control."""

    def __init__(self, root: str | Path, enforce: bool = True) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.enforce = enforce
        # node_id -> set of owned path patterns (may contain globs)
        self._ownership: dict[str, set[str]] = {}
        self._writes: dict[str, set[str]] = {}   # node_id -> paths actually written
        # Scaffolding the harness created on a node's behalf (empty __init__.py
        # and the like). Excluded from files_of(), so it is never scored as the
        # node's product nor flagged as an unowned write.
        self._harness_paths: set[str] = set()
        self._lock = threading.Lock()
        self.denied_reads: list[AccessDenial] = []
        self.denied_writes: list[AccessDenial] = []

    # ---- ownership ----------------------------------------------------
    def grant(self, node_id: str, paths: list[str]) -> None:
        """Give a node write access to a set of paths (globs allowed)."""
        self._ownership.setdefault(node_id, set()).update(_norm(p) for p in paths)

    def owns(self, node_id: str, path: str) -> bool:
        patterns = self._ownership.get(node_id)
        if not patterns:
            return False
        p = _norm(path)
        for pattern in patterns:
            if p == pattern or fnmatch.fnmatch(p, pattern):
                return True
            # A directory grant covers everything beneath it.
            if not pattern.endswith(".py") and p.startswith(pattern.rstrip("/") + "/"):
                return True
        return False

    # ---- io -----------------------------------------------------------
    def write(self, node_id: str, path: str, content: str) -> tuple[bool, str]:
        """Write a file on behalf of a node. Returns ``(ok, message)``."""
        p = _norm(path)
        if not p or p.startswith("..") or Path(p).is_absolute():
            self.denied_writes.append(AccessDenial(node_id, p, "write", "path escapes workspace"))
            return False, "refused: path escapes the workspace"
        if self.enforce and not self.owns(node_id, p):
            self.denied_writes.append(
                AccessDenial(node_id, p, "write", "not in owns_paths")
            )
            return False, (
                "refused: '" + p + "' is not owned by this node. Owned paths: "
                + ", ".join(sorted(self._ownership.get(node_id, set())) or ["<none>"])
            )
        target = self.root / p
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        self._writes.setdefault(node_id, set()).add(p)
        # A real write supersedes harness scaffolding at the same path: the file
        # now carries the node's own content and must be scored as its product.
        self._harness_paths.discard(p)
        return True, "wrote " + p + " (" + str(content.count(chr(10)) + 1) + " lines)"

    def is_claimed(self, path: str) -> bool:
        """True if any node has been granted this exact path."""
        p = _norm(path)
        return any(p in patterns for patterns in self._ownership.values())

    def read(self, node_id: str, path: str) -> tuple[bool, str]:
        """Read a file on behalf of a node, subject to ownership."""
        p = _norm(path)
        if self.enforce and not self.owns(node_id, p):
            self.denied_reads.append(AccessDenial(node_id, p, "read", "not owned"))
            return False, (
                "refused: '" + p + "' belongs to another node. You are given sibling "
                "contracts, not sibling source -- code against the declared interface."
            )
        target = self.root / p
        if not target.exists():
            return False, "not found: " + p
        return True, target.read_text(encoding="utf-8", errors="replace")

    # ---- inspection (harness side, not exposed to nodes) --------------
    def files_of(self, node_id: str) -> dict[str, str]:
        """Every file this node wrote, as ``{path: source}``."""
        out: dict[str, str] = {}
        for p in sorted(self._writes.get(node_id, set())):
            if p in self._harness_paths:
                continue
            target = self.root / p
            if target.exists():
                out[p] = target.read_text(encoding="utf-8", errors="replace")
        return out

    def files_of_many(self, node_ids: list[str]) -> dict[str, dict[str, str]]:
        return {n: self.files_of(n) for n in node_ids}

    def all_files(self, include: str = "*.py") -> dict[str, str]:
        """Every file in the workspace, regardless of writer."""
        out: dict[str, str] = {}
        for path in sorted(self.root.rglob("*")):
            if not path.is_file():
                continue
            rel = _norm(str(path.relative_to(self.root)))
            if any(part in _SKIP_DIRS for part in Path(rel).parts):
                continue
            if include and not fnmatch.fnmatch(rel, include):
                continue
            try:
                out[rel] = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
        return out

    def written_paths(self, node_id: str | None = None) -> list[str]:
        if node_id is not None:
            return sorted(self._writes.get(node_id, set()))
        seen: set[str] = set()
        for paths in self._writes.values():
            seen.update(paths)
        return sorted(seen)

    def tree(self, limit: int = 200) -> str:
        """Compact listing. Structure only -- never file contents."""
        rows: list[str] = []
        for path in sorted(self.root.rglob("*")):
            if not path.is_file():
                continue
            rel = _norm(str(path.relative_to(self.root)))
            if any(part in _SKIP_DIRS for part in Path(rel).parts):
                continue
            try:
                lines = path.read_text(encoding="utf-8", errors="replace").count("\n") + 1
            except OSError:
                lines = 0
            rows.append(rel + "  (" + str(lines) + " lines)")
            if len(rows) >= limit:
                rows.append("... truncated")
                break
        return "\n".join(rows) if rows else "<empty workspace>"

    def stats(self) -> dict[str, Any]:
        files = self.all_files(include="*")
        py = {k: v for k, v in files.items() if k.endswith(".py")}
        return {
            "files": len(files),
            "py_files": len(py),
            "loc": sum(v.count("\n") + 1 for v in py.values()),
            "denied_writes": len(self.denied_writes),
            "denied_reads": len(self.denied_reads),
        }

    def reset(self) -> None:
        if self.root.exists():
            shutil.rmtree(self.root, ignore_errors=True)
        self.root.mkdir(parents=True, exist_ok=True)
        self._writes.clear()
        self._harness_paths.clear()
        self.denied_reads.clear()
        self.denied_writes.clear()


_SKIP_DIRS = {"__pycache__", ".git", ".pytest_cache", ".mypy_cache", ".venv", "node_modules"}


def _norm(path: str) -> str:
    return str(path).replace("\\", "/").strip("/")


def ensure_package_files(workspace: Workspace, node_id: str, paths: list[str]) -> list[str]:
    """Create missing ``__init__.py`` files for any package directory in ``paths``.

    Packaging omissions are one of NL2Repo-Bench's named failure categories and
    are pure boilerplate, so the harness handles them rather than spending model
    turns on them.

    Runs under the workspace lock: sibling nodes build concurrently and would
    otherwise race to create the same ``__init__.py``, with both recording a
    write and both then being charged an unowned-write violation for it.
    Anything created here is marked as harness scaffolding, so it is excluded
    from the node's product and never scored against its contract.

    Returns the paths created.
    """
    created: list[str] = []
    dirs: set[str] = set()
    for p in paths:
        parent = os.path.dirname(_norm(p))
        while parent:
            dirs.add(parent)
            parent = os.path.dirname(parent)

    with workspace._lock:
        for d in sorted(dirs):
            if d in _SKIP_DIRS:
                continue
            init = d + "/__init__.py"
            if (workspace.root / init).exists() or init in workspace._harness_paths:
                continue
            # Never pre-create a file some node's contract claims. Siblings run
            # concurrently, so a node whose own __init__.py has not been written
            # YET would otherwise have an empty one created under it, get it
            # marked as harness scaffolding, and then be charged missing_symbol
            # for every symbol it was supposed to export -- nondeterministically,
            # depending on which sibling finished first.
            if workspace.is_claimed(init):
                continue
            workspace.grant(node_id, [init])
            target = workspace.root / init
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("", encoding="utf-8")
            workspace._writes.setdefault(node_id, set()).add(init)
            workspace._harness_paths.add(init)
            created.append(init)
    return created
