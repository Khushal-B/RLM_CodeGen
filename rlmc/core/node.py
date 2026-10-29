"""Build nodes and results: the shape of ``rlm_build(node, contract, budget)``."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from rlmc.contracts.types import CheckReport, Contract
from rlmc.core.payload import BoundedPayload


class NodeKind(str, Enum):
    REPO = "repo"
    PACKAGE = "package"
    MODULE = "module"
    FILE = "file"
    FUNCTION = "function"


class Action(str, Enum):
    """The action a node takes. Exactly one per visit."""

    IMPLEMENT = "IMPLEMENT"
    DECOMPOSE = "DECOMPOSE"
    ESCALATE = "ESCALATE"
    RETURN = "RETURN"


@dataclass
class Budget:
    """Token/cost allowance, split among children on decomposition."""

    tokens: int = 200_000
    usd: float | None = None
    iterations: int = 12
    depth_remaining: int = 2

    def split(self, n: int) -> list[Budget]:
        """Divide this budget among ``n`` children.

        Children share the parent's allowance rather than each inheriting it, so
        decomposition cannot multiply cost without bound -- the cost-explosion
        risk the RLM paper flags in its limitations.
        """
        if n <= 0:
            return []
        share_tokens = max(1_000, self.tokens // n)
        share_usd = (self.usd / n) if self.usd is not None else None
        return [
            Budget(
                tokens=share_tokens,
                usd=share_usd,
                iterations=self.iterations,
                depth_remaining=max(0, self.depth_remaining - 1),
            )
            for _ in range(n)
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "tokens": self.tokens,
            "usd": self.usd,
            "iterations": self.iterations,
            "depth_remaining": self.depth_remaining,
        }


@dataclass
class BuildNode:
    """A unit of work in the build tree."""

    node_id: str
    kind: NodeKind = NodeKind.MODULE
    title: str = ""
    spec: str = ""                # the slice of the requirements this node owns
    contract: Contract = field(default_factory=lambda: Contract.empty("unnamed"))
    depth: int = 0
    parent_id: str | None = None
    children: list[BuildNode] = field(default_factory=list)

    def descendants(self) -> list[BuildNode]:
        out: list[BuildNode] = []
        for c in self.children:
            out.append(c)
            out.extend(c.descendants())
        return out

    def to_dict(self, include_children: bool = True) -> dict[str, Any]:
        d: dict[str, Any] = {
            "node_id": self.node_id,
            "kind": self.kind.value,
            "title": self.title,
            "depth": self.depth,
            "parent_id": self.parent_id,
            "contract": self.contract.to_dict(),
        }
        if include_children:
            d["children"] = [c.to_dict() for c in self.children]
        return d


@dataclass
class BuildResult:
    """What ``rlm_build`` returns for one node.

    ``payload`` is what the PARENT sees. Everything else on this object is
    harness-side telemetry: it is recorded in the run manifest but is never
    placed into any model's context. Keeping that distinction explicit is what
    lets the run be audited for the invariant after the fact.
    """

    node_id: str
    ok: bool
    action: Action
    contract_hash: str
    payload: BoundedPayload
    report: CheckReport | None = None

    # ---- telemetry (harness only) ----
    depth: int = 0
    files_written: list[str] = field(default_factory=list)
    loc: int = 0
    children: list[BuildResult] = field(default_factory=list)
    repair_rounds: int = 0
    decompose_retries: int = 0
    escalated: bool = False
    iterations_used: int = 0
    subcalls: int = 0
    cost_usd: float = 0.0
    wallclock_s: float = 0.0
    root_context_tokens: int = 0      # high-water mark for THIS node's own context
    error: str | None = None
    notes: list[str] = field(default_factory=list)

    def subtree(self) -> list[BuildResult]:
        out: list[BuildResult] = []
        for c in self.children:
            out.append(c)
            out.extend(c.subtree())
        return out

    def total_cost(self) -> float:
        return self.cost_usd + sum(c.total_cost() for c in self.children)

    def total_subcalls(self) -> int:
        return self.subcalls + sum(c.total_subcalls() for c in self.children)

    def max_depth_reached(self) -> int:
        return max([self.depth] + [c.max_depth_reached() for c in self.children])

    def node_count(self) -> int:
        return 1 + sum(c.node_count() for c in self.children)

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "ok": self.ok,
            "action": self.action.value,
            "contract_hash": self.contract_hash,
            "payload": self.payload.to_dict(),
            "depth": self.depth,
            "files_written": self.files_written,
            "loc": self.loc,
            "repair_rounds": self.repair_rounds,
            "decompose_retries": self.decompose_retries,
            "escalated": self.escalated,
            "iterations_used": self.iterations_used,
            "subcalls": self.subcalls,
            "cost_usd": self.cost_usd,
            "wallclock_s": self.wallclock_s,
            "root_context_tokens": self.root_context_tokens,
            "error": self.error,
            "notes": self.notes,
            "report": self.report.to_dict() if self.report else None,
            "children": [c.to_dict() for c in self.children],
        }
