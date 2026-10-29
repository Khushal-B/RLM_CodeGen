"""Contract composition: do the child contracts add up to the parent's?

This is checked at DECOMPOSE time, before any child runs. It is cheap (pure
Python over declarations, no code exists yet) and it catches the most expensive
class of failure early: a decomposition that could not possibly satisfy the
parent no matter how well each child implements its part.

The four rules:

C1  Coverage      -- every symbol the parent must provide is provided by exactly
                     one child.
C2  Exclusivity   -- no two children claim the same symbol or the same file.
C3  Closure       -- every symbol a child requires is provided by a sibling, by
                     the parent's own requires (i.e. imported from outside the
                     subtree), or by a third-party/stdlib module.
C4  Acyclicity    -- the sibling dependency graph implied by requires is a DAG.

A decomposition that fails composition is rejected and re-requested from the
model. That retry is bounded; on exhaustion the node falls back to IMPLEMENT.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from rlmc.contracts.imports import SymbolOwnership, check_acyclic
from rlmc.contracts.types import Contract, Violation


@dataclass
class CompositionResult:
    ok: bool
    violations: list[Violation] = field(default_factory=list)
    # Diagnostics small enough to hand back to the model on a retry.
    missing: list[str] = field(default_factory=list)
    duplicated: list[str] = field(default_factory=list)
    unresolved: list[str] = field(default_factory=list)
    path_conflicts: list[str] = field(default_factory=list)

    def summary(self, limit: int = 6) -> str:
        parts: list[str] = []
        if self.missing:
            parts.append("uncovered parent symbols: " + ", ".join(self.missing[:limit]))
        if self.duplicated:
            parts.append("claimed by >1 child: " + ", ".join(self.duplicated[:limit]))
        if self.unresolved:
            parts.append("required but nobody provides: " + ", ".join(self.unresolved[:limit]))
        if self.path_conflicts:
            parts.append("file owned by >1 child: " + ", ".join(self.path_conflicts[:limit]))
        cycles = [v.subject for v in self.violations if v.kind == "import_cycle"]
        if cycles:
            parts.append("dependency cycles: " + "; ".join(cycles[:limit]))
        return "; ".join(parts) if parts else "composition ok"

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "violations": [v.to_dict() for v in self.violations],
            "missing": self.missing,
            "duplicated": self.duplicated,
            "unresolved": self.unresolved,
            "path_conflicts": self.path_conflicts,
            "summary": self.summary(),
        }


def check_composition(
    parent: Contract,
    children: list[Contract],
    external_modules: set[str] | None = None,
) -> CompositionResult:
    """Verify C1-C4 for a proposed decomposition.

    ``external_modules`` names modules known to live outside the subtree
    (third-party dependencies, stdlib); requirements resolving into them are
    accepted rather than reported as unresolved.
    """
    external_modules = external_modules or set()
    result = CompositionResult(ok=True)
    ownership = SymbolOwnership.from_contracts(children)

    # ---- C1 coverage ----------------------------------------------------
    child_provided: dict[str, list[str]] = {}
    for c in children:
        for sym in c.provides:
            child_provided.setdefault(sym.qualname, []).append(c.node_id)

    for sym in parent.provides:
        if sym.qualname not in child_provided:
            result.missing.append(sym.qualname)
            result.violations.append(
                Violation(
                    kind="missing_symbol",
                    boundary="python_stub",
                    subject=sym.qualname,
                    detail="parent must provide it but no child contract does",
                    owner=parent.node_id,
                    cross_node=True,
                )
            )

    # ---- C2 exclusivity -------------------------------------------------
    for qualname, owners in child_provided.items():
        if len(owners) > 1:
            result.duplicated.append(qualname)
            result.violations.append(
                Violation(
                    kind="kind_mismatch",
                    boundary="python_stub",
                    subject=qualname,
                    detail="claimed by multiple children: " + ", ".join(sorted(owners)),
                    owner=parent.node_id,
                    cross_node=True,
                )
            )

    path_owners: dict[str, list[str]] = {}
    for c in children:
        for p in c.owns_paths:
            path_owners.setdefault(_norm_path(p), []).append(c.node_id)
    for path, owners in path_owners.items():
        if len(owners) > 1:
            result.path_conflicts.append(path)
            result.violations.append(
                Violation(
                    kind="unowned_write",
                    boundary="import_graph",
                    subject=path,
                    detail="file claimed by multiple children: " + ", ".join(sorted(owners)),
                    owner=parent.node_id,
                    cross_node=True,
                )
            )

    # ---- C3 closure -----------------------------------------------------
    sibling_symbols = set(child_provided)
    sibling_modules = ownership.known_modules()
    parent_requires = set(parent.requires)
    for c in children:
        for req in c.requires:
            if req in sibling_symbols or req in parent_requires:
                continue
            if ownership.owner_of_symbol(req) is not None:
                continue
            root = req.split(".", 1)[0]
            if root in external_modules or req in external_modules:
                continue
            if any(req.startswith(m + ".") for m in sibling_modules):
                continue
            result.unresolved.append(req)
            result.violations.append(
                Violation(
                    kind="unresolved_require",
                    boundary="import_graph",
                    subject=req,
                    detail="required by " + c.node_id + " but provided by nobody",
                    owner=parent.node_id,
                    cross_node=True,
                )
            )

    # ---- C4 acyclicity --------------------------------------------------
    cycle_report = check_acyclic(children, ownership, parent.node_id)
    result.violations.extend(cycle_report.violations)

    result.ok = not result.violations
    return result


def _norm_path(path: str) -> str:
    return path.replace("\\", "/").strip("/")


def derive_child_requires(children: list[Contract]) -> None:
    """Best-effort: infer missing ``requires`` edges from declared annotations.

    Not used for scoring -- it only softens the retry loop when a model declares
    a sensible decomposition but forgets to list an obvious dependency. Left
    off by default in the driver so that the H3 arms are not quietly helped.
    """
    ownership = SymbolOwnership.from_contracts(children)
    for c in children:
        declared = set(c.requires)
        for sym in c.provides:
            for p in sym.params:
                if not p.annotation:
                    continue
                owner = ownership.owner_of_symbol(p.annotation)
                if owner and owner != c.node_id and p.annotation not in declared:
                    c.requires.append(p.annotation)
                    declared.add(p.annotation)
