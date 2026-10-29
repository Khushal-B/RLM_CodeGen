"""Boundary-type checker registry.

Every boundary type resolves to a ``BoundaryChecker`` with one method. Two are
fully implemented (``python_stub``, ``import_graph``); two are registered but
report ``unsupported`` until a service-oriented benchmark needs them
(``openapi``, ``sql_ddl``). Registering them now means the CVR-by-boundary
table, the contract schema and the blame router already have the right shape,
and filling them in later touches only this file.

An unsupported boundary contributes 0 to both the CVR numerator and denominator
-- it is never silently scored as passing.
"""

from __future__ import annotations

from typing import Any, Protocol

from rlmc.contracts.imports import SymbolOwnership, check_import_contract
from rlmc.contracts.stubs import check_stub_contract
from rlmc.contracts.types import BoundaryType, CheckReport, Contract, Violation


class BoundaryChecker(Protocol):
    """Deterministic verifier for one boundary type."""

    name: str
    supported: bool

    def check(
        self,
        contract: Contract,
        files: dict[str, str],
        ownership: SymbolOwnership,
        owner: str | None = None,
    ) -> CheckReport: ...


class PythonStubChecker:
    name = "python_stub"
    supported = True

    def check(
        self,
        contract: Contract,
        files: dict[str, str],
        ownership: SymbolOwnership,
        owner: str | None = None,
    ) -> CheckReport:
        return check_stub_contract(contract, files, owner=owner)


class ImportGraphChecker:
    name = "import_graph"
    supported = True

    def __init__(self, strict_undeclared: bool = True) -> None:
        self.strict_undeclared = strict_undeclared

    def check(
        self,
        contract: Contract,
        files: dict[str, str],
        ownership: SymbolOwnership,
        owner: str | None = None,
    ) -> CheckReport:
        return check_import_contract(
            contract,
            files,
            ownership,
            owner=owner,
            strict_undeclared=self.strict_undeclared,
        )


class _UnsupportedChecker:
    """Placeholder for a declared-but-unimplemented boundary type.

    Records the boundary as present so it appears in reports, but contributes
    nothing to CVR and emits a checker error rather than a violation, so an
    unimplemented boundary can never look like a pass.
    """

    supported = False

    def __init__(self, name: str, note: str) -> None:
        self.name = name
        self.note = note

    def check(
        self,
        contract: Contract,
        files: dict[str, str],
        ownership: SymbolOwnership,
        owner: str | None = None,
    ) -> CheckReport:
        report = CheckReport(node_id=contract.node_id)
        report.errors.append(
            "boundary '" + self.name + "' is declared but not implemented: " + self.note
        )
        return report


_REGISTRY: dict[str, Any] = {
    "python_stub": PythonStubChecker(),
    "import_graph": ImportGraphChecker(),
    "openapi": _UnsupportedChecker(
        "openapi",
        "needs a service-boundary benchmark; check = schema validation + "
        "generated client compiles",
    ),
    "sql_ddl": _UnsupportedChecker(
        "sql_ddl",
        "needs a data-boundary benchmark; check = migration applies + ORM model "
        "matches DDL",
    ),
}


def get_checker(boundary: BoundaryType) -> Any:
    checker = _REGISTRY.get(boundary)
    if checker is None:
        return _UnsupportedChecker(str(boundary), "unknown boundary type")
    return checker


def register_checker(boundary: str, checker: Any) -> None:
    """Install or replace a checker. Used by tests and by future boundaries."""
    _REGISTRY[boundary] = checker


def supported_boundaries() -> list[str]:
    return [name for name, c in _REGISTRY.items() if getattr(c, "supported", False)]


def unsupported_violation(boundary: str, node_id: str) -> Violation:
    return Violation(
        kind="schema_invalid",
        boundary=boundary,  # type: ignore[arg-type]
        subject=boundary,
        detail="no checker implemented for this boundary type",
        owner=node_id,
    )
