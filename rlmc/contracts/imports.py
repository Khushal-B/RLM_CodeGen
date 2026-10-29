"""``import_graph`` boundary: cross-node import resolution and cycle detection.

Checks three things over the files a node wrote, all deterministic:

1. Every symbol in ``contract.requires`` resolves to something a sibling
   contract actually ``provides``. This is the check that makes the invariant
   survivable: a node cannot look at sibling source, so its only guarantee that
   ``from pkg.parser import tokenize`` will work is that some sibling declared
   ``pkg.parser.tokenize``.
2. Every import that crosses a node boundary was declared in ``requires``. An
   undeclared cross-boundary import means the node reached around the contract.
3. The node-level dependency graph is acyclic.

Intra-node imports (a node importing its own modules) are unconstrained.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field

from rlmc.contracts.stubs import path_to_module
from rlmc.contracts.types import CheckReport, Contract, Violation


@dataclass
class ImportEdge:
    """One observed import statement."""

    from_module: str
    target: str          # dotted module or module.symbol
    symbol: str | None   # the imported name, when ``from X import Y``
    path: str
    level: int = 0       # relative-import depth (0 = absolute)

    @property
    def qualname(self) -> str:
        return self.target + "." + self.symbol if self.symbol else self.target


def extract_imports(files: dict[str, str]) -> list[ImportEdge]:
    """Collect every import statement in ``{relative_path: source}``."""
    edges: list[ImportEdge] = []
    for path, source in files.items():
        if not path.endswith(".py"):
            continue
        module = path_to_module(path)
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue  # reported by the stub checker; not double-counted here
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    edges.append(
                        ImportEdge(
                            from_module=module, target=alias.name, symbol=None, path=path
                        )
                    )
            elif isinstance(node, ast.ImportFrom):
                target = node.module or ""
                if node.level:
                    target = _resolve_relative(module, target, node.level)
                for alias in node.names:
                    edges.append(
                        ImportEdge(
                            from_module=module,
                            target=target,
                            symbol=alias.name,
                            path=path,
                            level=node.level or 0,
                        )
                    )
    return edges


def _resolve_relative(current_module: str, target: str, level: int) -> str:
    """Turn ``from ..x import y`` inside ``a.b.c`` into an absolute module."""
    parts = current_module.split(".") if current_module else []
    # level 1 == current package, so drop (level - 1) trailing components after
    # first dropping the module's own name.
    base = parts[: max(0, len(parts) - level)]
    if target:
        base = base + target.split(".")
    return ".".join(base)


@dataclass
class SymbolOwnership:
    """Which node provides which qualified symbol and which module.

    Built once per decomposition from the child contracts; it is the lookup that
    lets the blame router attribute a violation to a node deterministically.

    ``scope_complete`` records whether this map covers the node's whole sibling
    set. Requires-resolution is a composition-level property: asking "does
    anybody provide X?" is only meaningful when every sibling is present. A map
    built from a single contract (a leaf checked in isolation) sets the flag
    False, and the import checker then skips that rule rather than reporting a
    false ``unresolved_require`` for every legitimate cross-node dependency.
    """

    symbol_to_node: dict[str, str] = field(default_factory=dict)
    module_to_node: dict[str, str] = field(default_factory=dict)
    scope_complete: bool = True

    @classmethod
    def from_contracts(
        cls, contracts: list[Contract], scope_complete: bool = True
    ) -> SymbolOwnership:
        own = cls(scope_complete=scope_complete)
        for c in contracts:
            for sym in c.provides:
                own.symbol_to_node[sym.qualname] = c.node_id
                if sym.module:
                    own.module_to_node.setdefault(sym.module, c.node_id)
            for path in c.owns_paths:
                if path.endswith(".py"):
                    own.module_to_node.setdefault(path_to_module(path), c.node_id)
        return own

    def owner_of_symbol(self, qualname: str) -> str | None:
        if qualname in self.symbol_to_node:
            return self.symbol_to_node[qualname]
        # Fall back to the longest module prefix that we know an owner for.
        parts = qualname.split(".")
        for i in range(len(parts) - 1, 0, -1):
            prefix = ".".join(parts[:i])
            if prefix in self.module_to_node:
                return self.module_to_node[prefix]
        return None

    def owner_of_module(self, module: str) -> str | None:
        parts = module.split(".")
        for i in range(len(parts), 0, -1):
            prefix = ".".join(parts[:i])
            if prefix in self.module_to_node:
                return self.module_to_node[prefix]
        return None

    def all_provided(self) -> set[str]:
        return set(self.symbol_to_node)

    def known_modules(self) -> set[str]:
        return set(self.module_to_node)


def check_import_contract(
    contract: Contract,
    files: dict[str, str],
    ownership: SymbolOwnership,
    owner: str | None = None,
    strict_undeclared: bool = True,
) -> CheckReport:
    """Check requires-resolution and undeclared cross-boundary imports."""
    owner = owner or contract.node_id
    report = CheckReport(node_id=contract.node_id)

    # (1) Every declared requirement must be satisfied by some sibling contract.
    # Only meaningful with the full sibling set in scope (see SymbolOwnership).
    provided = ownership.all_provided()
    known_modules = ownership.known_modules()
    for req in contract.requires if ownership.scope_complete else []:
        report.record("import_graph", 1)
        if req in provided:
            continue
        if ownership.owner_of_symbol(req) is not None:
            continue
        if req in known_modules:
            continue
        report.add(
            Violation(
                kind="unresolved_require",
                boundary="import_graph",
                subject=req,
                detail="no sibling contract provides this symbol",
                owner=owner,
                # Nothing in the tree provides it, so the decomposition itself
                # is at fault rather than any single child.
                cross_node=True,
            )
        )

    # (2) Every cross-boundary import must have been declared.
    declared = set(contract.requires)
    declared_modules = {r.rsplit(".", 1)[0] for r in contract.requires if "." in r}
    own_modules = {path_to_module(p) for p in contract.owns_paths if p.endswith(".py")}

    for edge in extract_imports(files):
        target_owner = ownership.owner_of_module(edge.target)
        if target_owner is None:
            continue  # third-party or stdlib: outside the contract system
        if target_owner == owner:
            continue  # intra-node imports are unconstrained
        report.record("import_graph", 1)
        if edge.qualname in declared or edge.target in declared or edge.target in declared_modules:
            continue
        if edge.target in own_modules:
            continue
        if not strict_undeclared:
            continue
        report.add(
            Violation(
                kind="undeclared_import",
                boundary="import_graph",
                subject=edge.qualname,
                detail=(
                    "imports across boundary into node '" + target_owner
                    + "' without declaring it in requires"
                ),
                owner=owner,
                path=edge.path,
            )
        )

    return report


def node_dependency_graph(
    contracts: list[Contract], ownership: SymbolOwnership
) -> dict[str, set[str]]:
    """Build the node-level dependency graph implied by declared requires."""
    graph: dict[str, set[str]] = {c.node_id: set() for c in contracts}
    for c in contracts:
        for req in c.requires:
            target = ownership.owner_of_symbol(req)
            if target is not None and target != c.node_id:
                graph.setdefault(c.node_id, set()).add(target)
    return graph


def find_cycles(graph: dict[str, set[str]]) -> list[list[str]]:
    """Return simple cycles in a node dependency graph (DFS, first hit per root)."""
    cycles: list[list[str]] = []
    seen_signatures: set[frozenset[str]] = set()
    colour: dict[str, int] = {n: 0 for n in graph}  # 0 white, 1 grey, 2 black
    stack: list[str] = []

    def visit(node: str) -> None:
        colour[node] = 1
        stack.append(node)
        for nxt in sorted(graph.get(node, ())):
            if colour.get(nxt, 0) == 0:
                visit(nxt)
            elif colour.get(nxt) == 1:
                idx = stack.index(nxt)
                cycle = stack[idx:] + [nxt]
                sig = frozenset(cycle)
                if sig not in seen_signatures:
                    seen_signatures.add(sig)
                    cycles.append(cycle)
        stack.pop()
        colour[node] = 2

    for node in sorted(graph):
        if colour.get(node, 0) == 0:
            visit(node)
    return cycles


def check_acyclic(
    contracts: list[Contract], ownership: SymbolOwnership, parent_id: str
) -> CheckReport:
    """Confirm the sibling dependency graph is acyclic.

    A cycle is inherently cross-node, so it is attributed to the parent: the
    decomposition is wrong, not any one child.
    """
    report = CheckReport(node_id=parent_id)
    graph = node_dependency_graph(contracts, ownership)
    report.record("import_graph", max(1, len(graph)))
    for cycle in find_cycles(graph):
        report.add(
            Violation(
                kind="import_cycle",
                boundary="import_graph",
                subject=" -> ".join(cycle),
                detail="cyclic dependency between sibling nodes",
                owner=parent_id,
                cross_node=True,
            )
        )
    return report
