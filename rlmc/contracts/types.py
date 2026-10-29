"""Contract type system.

A contract is the machine-checkable interface a build node must expose to, and
may consume from, its siblings. It is the object that carries a constraint
across sibling branches of the recursion -- which is what makes output-side
decomposition possible at all (see Thesis_Validation_Plan sec. 2.8).

Design notes
------------
* ``Contract`` is deliberately boring and declarative. All intelligence lives in
  the checkers (``rlmc.contracts.checker``) and the composition rules
  (``rlmc.contracts.compose``), both of which are deterministic Python. No LLM
  is ever in the verification loop.
* ``Contract.hash()`` digests the *typed* content only. It is what a child
  returns to its parent under the bounded-return invariant, so it must not
  depend on prose, ordering, or formatting.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

# Boundary types. Only the first two are checkable against NL2Repo-Bench (pure
# Python libraries); the other two are declared here so the CVR-by-boundary
# table and the checker registry already have shape for service-oriented
# benchmarks.
BoundaryType = Literal["python_stub", "import_graph", "openapi", "sql_ddl"]

BOUNDARY_TYPES: tuple[BoundaryType, ...] = (
    "python_stub",
    "import_graph",
    "openapi",
    "sql_ddl",
)

SymbolKind = Literal["function", "class", "constant"]


@dataclass(frozen=True)
class Param:
    """A single parameter in a declared signature."""

    name: str
    annotation: str | None = None
    has_default: bool = False
    kind: Literal["positional", "keyword", "var_positional", "var_keyword"] = "positional"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Param:
        return cls(
            name=str(d.get("name", "")),
            annotation=d.get("annotation"),
            has_default=bool(d.get("has_default", False)),
            kind=d.get("kind", "positional"),
        )


@dataclass
class SymbolSpec:
    """One declared public symbol: a function, class, or module constant.

    ``qualname`` is dotted and rooted at the package (e.g. ``mypkg.parser.tokenize``).
    ``path`` is the file that must define it, relative to the workspace root.
    """

    qualname: str
    kind: SymbolKind
    path: str
    params: list[Param] = field(default_factory=list)
    returns: str | None = None
    doc: str | None = None
    # Methods of a class, as nested SymbolSpecs with kind="function".
    methods: list[SymbolSpec] = field(default_factory=list)

    @property
    def name(self) -> str:
        return self.qualname.rsplit(".", 1)[-1]

    @property
    def module(self) -> str:
        return self.qualname.rsplit(".", 1)[0] if "." in self.qualname else ""

    def signature(self) -> str:
        """Render as a Python signature string (used when emitting .pyi)."""
        parts: list[str] = []
        for p in self.params:
            if p.kind == "var_positional":
                s = "*" + p.name
            elif p.kind == "var_keyword":
                s = "**" + p.name
            else:
                s = p.name
            if p.annotation:
                s += ": " + p.annotation
            if p.has_default:
                s += " = ..."
            parts.append(s)
        ret = " -> " + self.returns if self.returns else ""
        return "(" + ", ".join(parts) + ")" + ret

    def arity(self) -> tuple[int, int, bool, bool]:
        """(required_count, total_named_count, has_varargs, has_kwargs)."""
        named = [p for p in self.params if p.kind in ("positional", "keyword")]
        required = [p for p in named if not p.has_default]
        return (
            len(required),
            len(named),
            any(p.kind == "var_positional" for p in self.params),
            any(p.kind == "var_keyword" for p in self.params),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "qualname": self.qualname,
            "kind": self.kind,
            "path": self.path,
            "params": [p.to_dict() for p in self.params],
            "returns": self.returns,
            "doc": self.doc,
            "methods": [m.to_dict() for m in self.methods],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SymbolSpec:
        return cls(
            qualname=str(d.get("qualname", "")),
            kind=d.get("kind", "function"),
            path=str(d.get("path", "")),
            params=[Param.from_dict(p) for p in d.get("params") or []],
            returns=d.get("returns"),
            doc=d.get("doc"),
            methods=[SymbolSpec.from_dict(m) for m in d.get("methods") or []],
        )

    def canonical(self) -> dict[str, Any]:
        """Hash-stable projection: identity and shape, never prose."""
        return {
            "qualname": self.qualname,
            "kind": self.kind,
            "path": self.path,
            "params": [
                {
                    "name": p.name,
                    "annotation": p.annotation,
                    "has_default": p.has_default,
                    "kind": p.kind,
                }
                for p in self.params
            ],
            "returns": self.returns,
            "methods": [m.canonical() for m in sorted(self.methods, key=lambda s: s.qualname)],
        }


@dataclass
class Contract:
    """The machine-checkable interface for one build node.

    Attributes
    ----------
    node_id:
        Identity of the owning node in the build tree.
    boundary:
        Which checker validates this contract.
    provides:
        Symbols this node MUST expose. Checked by AST walk against the files the
        node wrote. Every entry is one CVR denominator element.
    requires:
        Qualified symbols this node MAY consume from siblings. Checked as
        reference resolution against the sibling *contracts*, never against
        sibling source -- the node never sees sibling source.
    owns_paths:
        Files this node may write. Enforced by the workspace, so the invariant
        holds at the filesystem level and not only by prompting.
    prose:
        Free-text interface description. Populated ONLY in contract_mode="prose";
        deliberately excluded from the hash and from every checker, so the H3
        arms differ in exactly one respect.
    """

    node_id: str
    boundary: BoundaryType = "python_stub"
    provides: list[SymbolSpec] = field(default_factory=list)
    requires: list[str] = field(default_factory=list)
    owns_paths: list[str] = field(default_factory=list)
    prose: str | None = None
    # Free-form extras for non-Python boundaries (OpenAPI doc, DDL text, ...).
    payload: dict[str, Any] = field(default_factory=dict)

    # ---- identity -------------------------------------------------------
    def canonical(self) -> dict[str, Any]:
        return {
            "boundary": self.boundary,
            "provides": [s.canonical() for s in sorted(self.provides, key=lambda s: s.qualname)],
            "requires": sorted(self.requires),
            "owns_paths": sorted(self.owns_paths),
            "payload": self.payload,
        }

    def hash(self) -> str:
        blob = json.dumps(self.canonical(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]

    # ---- convenience ----------------------------------------------------
    def provided_names(self) -> set[str]:
        return {s.qualname for s in self.provides}

    def symbol(self, qualname: str) -> SymbolSpec | None:
        for s in self.provides:
            if s.qualname == qualname:
                return s
        return None

    def is_empty(self) -> bool:
        return not self.provides and not self.requires and not self.prose

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "boundary": self.boundary,
            "provides": [s.to_dict() for s in self.provides],
            "requires": list(self.requires),
            "owns_paths": list(self.owns_paths),
            "prose": self.prose,
            "payload": self.payload,
            "hash": self.hash(),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Contract:
        return cls(
            node_id=str(d.get("node_id", "")),
            boundary=d.get("boundary", "python_stub"),
            provides=[SymbolSpec.from_dict(s) for s in d.get("provides") or []],
            requires=[str(r) for r in d.get("requires") or []],
            owns_paths=[str(p) for p in d.get("owns_paths") or []],
            prose=d.get("prose"),
            payload=d.get("payload") or {},
        )

    @classmethod
    def empty(cls, node_id: str) -> Contract:
        return cls(node_id=node_id, provides=[], requires=[], owns_paths=[])


# ---------------------------------------------------------------------------
# Violations
# ---------------------------------------------------------------------------

ViolationKind = Literal[
    "missing_symbol",       # declared in provides, absent from the written code
    "arity_mismatch",       # exists but wrong number of required/optional params
    "annotation_mismatch",  # exists but annotation differs from the declaration
    "kind_mismatch",        # declared a class, got a function (or vice versa)
    "missing_method",       # class exists, declared method does not
    "unresolved_require",   # consumes a symbol no sibling contract provides
    "undeclared_import",    # imports across a boundary it never declared
    "import_cycle",         # cyclic package dependency
    "unowned_write",        # wrote a file outside owns_paths
    "syntax_error",         # the file does not parse
    "schema_invalid",       # non-Python boundary payload failed validation
]


@dataclass
class Violation:
    """One deterministic contract failure.

    ``owner`` is the node the violation is attributed to. It is resolved by
    ``rlmc.core.blame`` via a lookup in the contract graph -- never guessed by a
    model -- which is what makes repair routing possible without the parent ever
    reading child source.
    """

    kind: ViolationKind
    boundary: BoundaryType
    subject: str                 # qualname, import edge, or path
    detail: str = ""
    owner: str | None = None     # node_id held responsible
    path: str | None = None      # file it was observed in
    # True when the violation spans more than one node and therefore indicts the
    # decomposition itself rather than any single child.
    cross_node: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def brief(self) -> str:
        """One-line rendering, used inside the bounded diagnostic payload."""
        s = self.kind + ":" + self.subject
        if self.detail:
            s += " (" + self.detail + ")"
        return s


@dataclass
class CheckReport:
    """Result of checking one node's output against its contract."""

    node_id: str
    checked: int = 0                      # CVR denominator
    violations: list[Violation] = field(default_factory=list)
    by_boundary_checked: dict[str, int] = field(default_factory=dict)
    by_boundary_violations: dict[str, int] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)   # checker-internal failures

    @property
    def ok(self) -> bool:
        return not self.violations

    @property
    def cvr(self) -> float:
        """Contract Violation Rate: violations / checked elements."""
        if self.checked <= 0:
            return 0.0
        return len(self.violations) / self.checked

    def cvr_by_boundary(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for b, n in self.by_boundary_checked.items():
            if n > 0:
                out[b] = self.by_boundary_violations.get(b, 0) / n
        return out

    def merge(self, other: CheckReport) -> CheckReport:
        self.checked += other.checked
        self.violations.extend(other.violations)
        self.errors.extend(other.errors)
        for b, n in other.by_boundary_checked.items():
            self.by_boundary_checked[b] = self.by_boundary_checked.get(b, 0) + n
        for b, n in other.by_boundary_violations.items():
            self.by_boundary_violations[b] = self.by_boundary_violations.get(b, 0) + n
        return self

    def dedupe(self) -> CheckReport:
        """Drop violations identical in kind, subject, owner and path.

        A safety net: the same defect must never be counted twice in the CVR
        numerator just because two passes observed it.
        """
        seen: set[tuple[str, str, str | None, str | None]] = set()
        kept: list[Violation] = []
        by_boundary: dict[str, int] = {}
        for v in self.violations:
            key = (v.kind, v.subject, v.owner, v.path)
            if key in seen:
                continue
            seen.add(key)
            kept.append(v)
            by_boundary[v.boundary] = by_boundary.get(v.boundary, 0) + 1
        self.violations = kept
        self.by_boundary_violations = by_boundary
        return self

    def record(self, boundary: BoundaryType, checked: int) -> None:
        self.checked += checked
        self.by_boundary_checked[boundary] = self.by_boundary_checked.get(boundary, 0) + checked

    def add(self, violation: Violation) -> None:
        self.violations.append(violation)
        b = violation.boundary
        self.by_boundary_violations[b] = self.by_boundary_violations.get(b, 0) + 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "ok": self.ok,
            "checked": self.checked,
            "violations": [v.to_dict() for v in self.violations],
            "cvr": self.cvr,
            "by_boundary_checked": self.by_boundary_checked,
            "by_boundary_violations": self.by_boundary_violations,
            "cvr_by_boundary": self.cvr_by_boundary(),
            "errors": self.errors,
        }
