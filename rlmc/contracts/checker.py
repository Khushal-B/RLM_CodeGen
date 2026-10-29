"""Top-level verification entry point.

``verify_node`` is what the driver calls after a node finishes. It runs every
applicable boundary checker over the files the node owns, merges the reports,
and returns a single ``CheckReport`` carrying the CVR numerator/denominator
broken out by boundary type.

Verification never involves a language model. That is the property that lets a
parent trust a child's one-bit return without reading its source, and it is the
answer to the RLM paper's depth-degradation result (Table 1): plain RLM feeds
unverified natural language into the parent's reasoning, so a child's error
contaminates the parent immediately; here every recursion boundary is gated on
execution and static analysis.
"""

from __future__ import annotations

from rlmc.contracts.imports import SymbolOwnership, check_acyclic
from rlmc.contracts.registry import get_checker
from rlmc.contracts.types import CheckReport, Contract, Violation


def verify_node(
    contract: Contract,
    files: dict[str, str],
    ownership: SymbolOwnership | None = None,
    owner: str | None = None,
    extra_boundaries: list[str] | None = None,
) -> CheckReport:
    """Verify one node's written files against its contract.

    Parameters
    ----------
    contract:
        The node's own contract.
    files:
        ``{workspace_relative_path: source}`` for the files this node owns.
    ownership:
        Symbol ownership across the sibling set, used by the import checker.
        When omitted, ownership is derived from this contract alone, which makes
        every cross-boundary import look external -- correct for a leaf checked
        in isolation.
    extra_boundaries:
        Additional boundary types to run beyond the contract's declared one.
        The driver passes ``["import_graph"]`` for Python nodes so both
        denominators are populated.
    """
    if ownership is None:
        # No sibling set in scope: cross-node imports look external and
        # requires-resolution is skipped rather than falsely failed.
        ownership = SymbolOwnership.from_contracts([contract], scope_complete=False)
    owner = owner or contract.node_id

    boundaries = [contract.boundary]
    for b in extra_boundaries or []:
        if b not in boundaries:
            boundaries.append(b)

    merged = CheckReport(node_id=contract.node_id)
    for boundary in boundaries:
        checker = get_checker(boundary)  # type: ignore[arg-type]
        try:
            report = checker.check(contract, files, ownership, owner=owner)
        except Exception as exc:  # a broken checker must not look like a pass
            report = CheckReport(node_id=contract.node_id)
            report.errors.append(
                "checker '" + str(boundary) + "' raised " + type(exc).__name__ + ": " + str(exc)
            )
        merged.merge(report)

    # A node that wrote outside its lane is a violation regardless of boundary.
    merged.merge(check_ownership(contract, files, owner))
    return merged


def check_ownership(contract: Contract, files: dict[str, str], owner: str) -> CheckReport:
    """Flag any written file the contract does not claim.

    The workspace refuses unowned writes at call time, so this should normally
    be empty; it exists so that a bug in the workspace shows up as a measured
    violation rather than as a silent invariant breach.
    """
    report = CheckReport(node_id=contract.node_id)
    if not contract.owns_paths:
        return report
    allowed = {p.replace("\\", "/").strip("/") for p in contract.owns_paths}
    for path in files:
        norm = path.replace("\\", "/").strip("/")
        report.record("import_graph", 1)
        if norm not in allowed:
            report.add(
                Violation(
                    kind="unowned_write",
                    boundary="import_graph",
                    subject=norm,
                    detail="written but not listed in owns_paths",
                    owner=owner,
                    path=norm,
                )
            )
    return report


def verify_subtree(
    parent: Contract,
    children: list[Contract],
    files_by_node: dict[str, dict[str, str]],
) -> CheckReport:
    """Verify a whole decomposition level: each child, then the sibling graph.

    ``files_by_node`` maps node_id to the files that node wrote. The parent is
    verified against the union, since its contract is the composition of its
    children's.
    """
    ownership = SymbolOwnership.from_contracts(children)
    merged = CheckReport(node_id=parent.node_id)

    for child in children:
        files = files_by_node.get(child.node_id, {})
        merged.merge(
            verify_node(
                child,
                files,
                ownership=ownership,
                owner=child.node_id,
                extra_boundaries=["import_graph"],
            )
        )

    merged.merge(check_acyclic(children, ownership, parent.node_id))

    # The parent is checked only on the RESIDUAL: symbols it must provide that
    # no child contract claimed. Re-checking the union of all files here would
    # double-count every violation the children were already charged for, which
    # would inflate CVR -- the headline metric -- by roughly the branching
    # factor. Anything a child did claim is that child's business, and
    # composition (rule C1) is what guarantees the claims cover the parent.
    claimed = {sym.qualname for c in children for sym in c.provides}
    residual = [s for s in parent.provides if s.qualname not in claimed]
    if residual:
        all_files: dict[str, str] = {}
        for files in files_by_node.values():
            all_files.update(files)
        residual_contract = Contract(
            node_id=parent.node_id,
            boundary=parent.boundary,
            provides=residual,
            requires=[],
            owns_paths=[],  # empty: ownership is a per-child concern
        )
        merged.merge(
            verify_node(
                residual_contract, all_files, ownership=ownership, owner=parent.node_id
            )
        )

    merged.dedupe()
    return merged
