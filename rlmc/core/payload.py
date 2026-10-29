"""The return payload: where the H1 invariant is enforced.

This module is the whole of H1. A child node's result is converted here into
the text its parent will see, and the conversion is the only difference between
the two arms of the headline ablation.

Why this lives in Python and not in a prompt
--------------------------------------------
RLM's in-REPL ``rlm_query`` returns the child's entire response string into the
parent's REPL (rlm/environments/local_repl.py). Nothing stops the model from
printing it. If the bounded-return invariant were only an instruction in the
system prompt, a single disobedient turn would silently void the experiment and
no reviewer should believe the result. Enforcing it in code makes the bound
auditable: ``BoundedPayload.enforced`` records the pre- and post-truncation
sizes for every single return, and the run manifest carries the maximum
observed payload so the claim "no node ever returned more than C tokens" is a
measurement rather than an assertion.

The invariant is bounded, not absolute
--------------------------------------
Per Problem C of the validation plan: "no source ever returns" is
unimplementable, because the repair loop then has no information channel at
all. The implementable form is: a node returns at most C tokens, where C is a
constant independent of subtree size. That still yields O(1) root growth, which
is the property H1 actually claims.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from rlmc.contracts.types import CheckReport, Violation

# Rough tokens-per-character for code and English. Used only to size the
# bound; exact tokenisation is applied in metrics when a tokenizer is present.
CHARS_PER_TOKEN = 4


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, len(text) // CHARS_PER_TOKEN)


def budget_chars(budget_tokens: int) -> int:
    return max(0, budget_tokens) * CHARS_PER_TOKEN


@dataclass
class BoundedPayload:
    """What a parent actually receives from a child.

    ``text`` is the rendered payload. ``raw_tokens`` is what an unbounded return
    would have cost, so the two arms can be compared directly on the same run.
    """

    text: str
    tokens: int
    raw_tokens: int
    truncated: bool
    mode: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "tokens": self.tokens,
            "raw_tokens": self.raw_tokens,
            "truncated": self.truncated,
            "mode": self.mode,
        }


@dataclass
class ReturnRecord:
    """Audit trail of every payload returned anywhere in the tree."""

    entries: list[dict[str, Any]] = field(default_factory=list)

    def record(self, node_id: str, depth: int, payload: BoundedPayload) -> None:
        self.entries.append(
            {
                "node_id": node_id,
                "depth": depth,
                **payload.to_dict(),
            }
        )

    @property
    def max_tokens(self) -> int:
        return max((e["tokens"] for e in self.entries), default=0)

    @property
    def max_raw_tokens(self) -> int:
        return max((e["raw_tokens"] for e in self.entries), default=0)

    @property
    def total_tokens(self) -> int:
        return sum(e["tokens"] for e in self.entries)

    def bound_respected(self, budget_tokens: int) -> bool:
        """True iff no return anywhere in the tree exceeded the constant C."""
        return all(e["tokens"] <= budget_tokens for e in self.entries)

    def to_dict(self) -> dict[str, Any]:
        return {
            "count": len(self.entries),
            "max_tokens": self.max_tokens,
            "max_raw_tokens": self.max_raw_tokens,
            "total_tokens": self.total_tokens,
            "entries": self.entries,
        }


def build_payload(
    *,
    node_id: str,
    ok: bool,
    contract_hash: str,
    report: CheckReport | None,
    files: dict[str, str],
    return_mode: str,
    budget_tokens: int,
    note: str = "",
) -> BoundedPayload:
    """Render a child's result for its parent.

    ``bounded``     -> (ok, hash, counts, <= C tokens of violation briefs)
    ``full_source`` -> every file the subtree wrote, concatenated

    ``raw_tokens`` is always computed against the full-source rendering, so a
    bounded run still reports what the unbounded arm would have cost. That is
    the number behind the root-context-high-water-mark plot.
    """
    full_text = render_full_source(files)
    raw_tokens = estimate_tokens(full_text)

    if return_mode == "full_source":
        return BoundedPayload(
            text=full_text,
            tokens=raw_tokens,
            raw_tokens=raw_tokens,
            truncated=False,
            mode="full_source",
        )

    text = render_bounded(
        node_id=node_id,
        ok=ok,
        contract_hash=contract_hash,
        report=report,
        files=files,
        budget_tokens=budget_tokens,
        note=note,
    )
    tokens = estimate_tokens(text)
    return BoundedPayload(
        text=text,
        tokens=tokens,
        raw_tokens=raw_tokens,
        truncated=tokens < raw_tokens,
        mode="bounded",
    )


def render_full_source(files: dict[str, str]) -> str:
    parts: list[str] = []
    for path in sorted(files):
        parts.append("### " + path + "\n" + files[path])
    return "\n\n".join(parts)


def render_bounded(
    *,
    node_id: str,
    ok: bool,
    contract_hash: str,
    report: CheckReport | None,
    files: dict[str, str],
    budget_tokens: int,
    note: str = "",
) -> str:
    """Build the bounded payload and hard-truncate it to the constant C.

    The header (ok / hash / counts) is never truncated -- it is fixed size by
    construction. Only the diagnostic tail is trimmed, so the parent always has
    a well-formed result to branch on even at a very small C.
    """
    header = {
        "node": node_id,
        "ok": ok,
        "contract_hash": contract_hash,
        "files": len(files),
        "loc": sum(f.count("\n") + 1 for f in files.values()),
    }
    if report is not None:
        header["checked"] = report.checked
        header["violations"] = len(report.violations)
        header["cvr"] = round(report.cvr, 4)

    head = json.dumps(header, separators=(",", ":"), sort_keys=True)

    briefs: list[str] = []
    if report is not None and report.violations:
        briefs = [_violation_brief(v) for v in report.violations]
    if note:
        briefs.insert(0, note)

    limit = budget_chars(budget_tokens)
    out = head
    for brief in briefs:
        candidate = out + "\n" + brief
        if len(candidate) > limit:
            remaining = len(briefs) - briefs.index(brief)
            tail = "\n[+" + str(remaining) + " more]"
            if len(out) + len(tail) <= limit:
                out += tail
            break
        out = candidate

    # Final hard bound. The header alone can exceed a pathologically small C;
    # truncating here keeps the guarantee absolute rather than best-effort.
    if len(out) > limit:
        out = out[:limit]
    return out


def _violation_brief(v: Violation) -> str:
    """One line per violation, owner-tagged so the parent can route repair."""
    owner = v.owner or "?"
    line = owner + " " + v.kind + " " + v.subject
    if v.cross_node:
        line += " [cross-node]"
    return line[:200]
