"""Credit assignment across a recursion tree with externalised artifacts.

This is Problem D of the validation plan -- the repair/escalation channel that
both existing thesis documents leave as a single word.

The problem
-----------
Node N's contract check fails after children C1..Ck have been built. Something
must decide what to re-run. CodeTeam solves the analogous problem by routing QA
failures back to "the responsible developer" using file ownership -- but it can
only do that because its agents share file contents. Here they deliberately do
not, so that solution is unavailable.

The mechanism
-------------
The contract graph is already an index from symbol to owning node. So blame is
a lookup, not an inference:

* A violation naming a symbol some child DECLARED  -> that child. The child
  promised it and did not deliver, and no source needs to be read to know that.
* A violation that spans two or more children, or names a symbol nobody
  declared -> the DECOMPOSITION is at fault, not any child. The parent
  re-decomposes.

That two-way split falls directly out of the contract, which is why contracts
here are load-bearing rather than decorative: they are what makes repair
possible at all once the artifact is externalised.

The ladder, in order:
    1. RETRY_CHILD     one child owns every violation -> rebuild that child
    2. RETRY_CHILDREN  several children own disjoint violations -> rebuild each
    3. REDECOMPOSE     violations are cross-node -> the contract split is wrong
    4. ESCALATE        retries exhausted -> same decomposition, stronger tier
    5. GIVE_UP         budget exhausted -> return ok=False with a bounded note
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from rlmc.contracts.types import CheckReport, Violation


class RepairAction(str, Enum):
    RETRY_CHILD = "retry_child"
    RETRY_CHILDREN = "retry_children"
    REDECOMPOSE = "redecompose"
    ESCALATE = "escalate"
    GIVE_UP = "give_up"


@dataclass
class RepairPlan:
    """What to do about a failed contract check, and why."""

    action: RepairAction
    targets: list[str] = field(default_factory=list)      # node_ids to rebuild
    reason: str = ""
    # Per-target bounded feedback: node_id -> violation briefs it must fix.
    feedback: dict[str, list[str]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action.value,
            "targets": self.targets,
            "reason": self.reason,
            "feedback": self.feedback,
        }


def attribute(
    report: CheckReport, known_children: list[str]
) -> tuple[dict[str, list[Violation]], list[Violation]]:
    """Split violations into per-child buckets and a cross-node remainder.

    A violation is charged to a child only when that child is a real member of
    the current sibling set; anything else (unknown owner, explicitly cross-node,
    or no owner at all) lands in the remainder and indicts the decomposition.
    """
    children = set(known_children)
    by_child: dict[str, list[Violation]] = defaultdict(list)
    cross: list[Violation] = []

    for v in report.violations:
        if v.cross_node or not v.owner or v.owner not in children:
            cross.append(v)
        else:
            by_child[v.owner].append(v)
    return dict(by_child), cross


def plan_repair(
    report: CheckReport,
    known_children: list[str],
    *,
    repair_rounds_used: int,
    max_repair_rounds: int,
    decompose_retries_used: int,
    max_decompose_retries: int,
    can_escalate: bool,
    feedback_limit: int = 8,
) -> RepairPlan:
    """Choose the next repair step. Deterministic; no model involved."""
    if report.ok:
        return RepairPlan(action=RepairAction.GIVE_UP, reason="nothing to repair")

    by_child, cross = attribute(report, known_children)

    # Cross-node violations mean the split itself is wrong. Fixing individual
    # children cannot help, because each child is faithfully implementing a
    # contract that does not add up.
    if cross:
        if decompose_retries_used < max_decompose_retries:
            return RepairPlan(
                action=RepairAction.REDECOMPOSE,
                targets=[],
                reason=(
                    str(len(cross)) + " cross-node violation(s): the decomposition does "
                    "not compose. " + _summarise(cross, feedback_limit)
                ),
                feedback={"__parent__": [v.brief() for v in cross[:feedback_limit]]},
            )
        if can_escalate:
            return RepairPlan(
                action=RepairAction.ESCALATE,
                targets=list(by_child) or list(known_children),
                reason="re-decomposition budget exhausted with cross-node violations",
                feedback={
                    k: [v.brief() for v in vs[:feedback_limit]] for k, vs in by_child.items()
                },
            )
        return RepairPlan(
            action=RepairAction.GIVE_UP,
            reason="cross-node violations remain and no repair budget left",
        )

    # Every violation is attributable. Rebuild exactly the guilty children.
    if by_child and repair_rounds_used < max_repair_rounds:
        targets = sorted(by_child)
        action = RepairAction.RETRY_CHILD if len(targets) == 1 else RepairAction.RETRY_CHILDREN
        return RepairPlan(
            action=action,
            targets=targets,
            reason=(
                "violations attributed to " + str(len(targets)) + " child node(s) via the "
                "contract graph"
            ),
            feedback={
                k: [v.brief() for v in vs[:feedback_limit]] for k, vs in by_child.items()
            },
        )

    if can_escalate:
        return RepairPlan(
            action=RepairAction.ESCALATE,
            targets=sorted(by_child) or list(known_children),
            reason="repair rounds exhausted; retrying at a stronger tier",
            feedback={
                k: [v.brief() for v in vs[:feedback_limit]] for k, vs in by_child.items()
            },
        )

    return RepairPlan(
        action=RepairAction.GIVE_UP,
        reason="repair budget exhausted with " + str(len(report.violations)) + " violation(s)",
    )


def _summarise(violations: list[Violation], limit: int) -> str:
    return "; ".join(v.brief() for v in violations[:limit])
