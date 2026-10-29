"""Run telemetry: the numbers that become the figures.

Five things are measured, matching the metric list in the validation plan:

* root context high-water mark vs target LOC -- the plot that makes the
  mechanism visible. This method should be flat; every baseline should grow.
* contract violation rate, broken out by boundary type -- the new metric.
* completion rate -- did the run terminate with an installable package.
* cost and wall-clock -- so every table row can be cost-matched.
* decomposition shape (node count, depth, children per node) -- the trajectory
  that shows whether decomposition size is collapsing, which is DecompRL's
  documented RL failure mode.

``context_high_water`` reads RLM's own logger trajectory. ``RLMLogger`` records
each iteration's full message history (rlm/logger/rlm_logger.py), so the root's
context size per turn is recoverable exactly rather than estimated.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from rlmc.core.payload import estimate_tokens

try:  # optional; falls back to a character proxy
    import tiktoken

    _ENC = tiktoken.get_encoding("cl100k_base")
except Exception:  # pragma: no cover - tiktoken is not a hard dependency
    _ENC = None


def count_tokens(text: str) -> int:
    if not text:
        return 0
    if _ENC is not None:
        try:
            return len(_ENC.encode(text))
        except Exception:
            pass
    return estimate_tokens(text)


def messages_tokens(messages: Any) -> int:
    """Token count of a message list, a string, or anything str()-able."""
    if messages is None:
        return 0
    if isinstance(messages, str):
        return count_tokens(messages)
    if isinstance(messages, list):
        total = 0
        for m in messages:
            if isinstance(m, dict):
                total += count_tokens(str(m.get("content", "")))
            else:
                total += count_tokens(str(m))
        return total
    if isinstance(messages, dict):
        return count_tokens(str(messages.get("content", messages)))
    return count_tokens(str(messages))


def context_high_water(trajectory: dict[str, Any] | None) -> int:
    """Peak context size of one node's own conversation, in tokens.

    Takes the max over iterations of that iteration's full prompt. This is the
    per-node quantity; the run-level high-water mark is the max over nodes, and
    for the H1 plot what matters is that the ROOT node's value does not grow
    with target repository size.
    """
    if not trajectory or not isinstance(trajectory, dict):
        return 0
    # Accept either the logger trajectory or the run_rlm_node wrapper, whose
    # "iterations" is a count rather than a list.
    iterations = trajectory.get("iterations")
    if not isinstance(iterations, list):
        iterations = (trajectory.get("trajectory") or {}).get("iterations")
    peak = 0
    for it in iterations if isinstance(iterations, list) else []:
        if isinstance(it, dict):
            peak = max(peak, messages_tokens(it.get("prompt")))
    return max(peak, int(trajectory.get("context_tokens", 0) or 0))


@dataclass
class ContextSample:
    node_id: str
    depth: int
    tokens: int


@dataclass
class RunTelemetry:
    """Accumulates measurements across one task run."""

    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_cost_usd: float = 0.0
    context_samples: list[ContextSample] = field(default_factory=list)
    repairs: dict[str, int] = field(default_factory=dict)
    compositions_attempted: int = 0
    compositions_rejected: int = 0

    # ---- recording ----------------------------------------------------
    def record_call(self, usage: dict[str, Any] | None) -> None:
        if not usage:
            return
        self.calls += int(usage.get("calls", 1) or 1)
        self.input_tokens += int(usage.get("input_tokens", 0) or 0)
        self.output_tokens += int(usage.get("output_tokens", 0) or 0)
        self.total_cost_usd += float(usage.get("cost_usd", 0.0) or 0.0)

    def record_context(self, node_id: str, depth: int, tokens: int) -> None:
        if tokens > 0:
            self.context_samples.append(ContextSample(node_id, depth, tokens))

    def record_repair(self, action: str) -> None:
        self.repairs[action] = self.repairs.get(action, 0) + 1

    def record_composition(self, ok: bool) -> None:
        self.compositions_attempted += 1
        if not ok:
            self.compositions_rejected += 1

    # ---- derived ------------------------------------------------------
    @property
    def root_context_tokens(self) -> int:
        """Peak context of the shallowest node -- the H1 quantity."""
        if not self.context_samples:
            return 0
        min_depth = min(s.depth for s in self.context_samples)
        return max(s.tokens for s in self.context_samples if s.depth == min_depth)

    @property
    def peak_context_tokens(self) -> int:
        return max((s.tokens for s in self.context_samples), default=0)

    @property
    def composition_reject_rate(self) -> float:
        if self.compositions_attempted == 0:
            return 0.0
        return self.compositions_rejected / self.compositions_attempted

    def to_dict(self) -> dict[str, Any]:
        return {
            "calls": self.calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_cost_usd": round(self.total_cost_usd, 6),
            "root_context_tokens": self.root_context_tokens,
            "peak_context_tokens": self.peak_context_tokens,
            "context_samples": [
                {"node_id": s.node_id, "depth": s.depth, "tokens": s.tokens}
                for s in self.context_samples
            ],
            "repairs": self.repairs,
            "compositions_attempted": self.compositions_attempted,
            "compositions_rejected": self.compositions_rejected,
            "composition_reject_rate": round(self.composition_reject_rate, 4),
        }


def decomposition_shape(result: Any) -> dict[str, Any]:
    """Node count, depth and branching -- the size-collapse watch.

    DecompRL reports that under RL the decomposition shrinks until the policy
    reverts to monolithic generation. Recording the shape every run is what lets
    that be seen rather than discovered late.
    """
    if result is None:
        return {}
    nodes = [result] + result.subtree()
    branching = [len(n.children) for n in nodes if n.children]
    return {
        "nodes": len(nodes),
        "leaves": sum(1 for n in nodes if not n.children),
        "max_depth": result.max_depth_reached(),
        "mean_branching": round(sum(branching) / len(branching), 2) if branching else 0.0,
        "decompose_nodes": sum(1 for n in nodes if n.children),
    }
