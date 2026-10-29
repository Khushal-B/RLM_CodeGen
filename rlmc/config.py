"""Arm configuration: the ablation axes, as data.

Every hypothesis in the thesis is one field here, so an experiment is a config
diff rather than a code branch. That is what makes "identical scaffold, one bit
changed" literally true and defensible:

    H1 artifact externalisation   ->  return_mode
    H2 adaptive depth             ->  depth_mode / max_depth
    H3 contracts as interface     ->  contract_mode

``ArmConfig.preset`` provides the named arms the plan calls for. Anything not
named by a preset can still be built by hand, but presets are what the runner
records in the run manifest so an arm is reproducible from its name alone.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

# --- H1: what a child returns to its parent -------------------------------
# "bounded"     : (ok, contract_hash, counts, <= C tokens of diagnostic)
# "full_source" : the child's complete source, as CodeTeam/ChatDev/OpenHands do
ReturnMode = Literal["bounded", "full_source"]

# --- H3: what a contract is ----------------------------------------------
# "typed" : .pyi stubs + import graph, machine-checked
# "prose" : English interface description, no mechanical check
# "none"  : children get no interface information about siblings at all
ContractMode = Literal["typed", "prose", "none"]

# --- H2: how deep to recurse ---------------------------------------------
# "adaptive" : the node chooses IMPLEMENT vs DECOMPOSE
# "fixed"    : decompose iff current_depth < max_depth
DepthMode = Literal["adaptive", "fixed"]


@dataclass
class ArmConfig:
    """One experimental arm."""

    name: str = "custom"

    # ---- hypothesis axes ----
    return_mode: ReturnMode = "bounded"
    contract_mode: ContractMode = "typed"
    depth_mode: DepthMode = "adaptive"

    # ---- the bounded-return constant C --------------------------------
    # Locked before any run, per the pre-registration in the validation plan
    # (sec. 9.3). A node returns at most this many tokens of diagnostic to its
    # parent, INDEPENDENT of subtree size. This is what makes root context
    # growth O(1) in target repository size.
    #
    # 256 tokens is chosen to fit: an ok flag, a 16-char contract hash, four
    # counters, and roughly 6-8 one-line violation briefs -- enough for the
    # repair router to act on, far too little to carry source code. Raising it
    # weakens H1; it must not drift upward during development.
    diagnostic_budget_tokens: int = 256

    # ---- recursion shape ----
    max_depth: int = 2            # 0 = single monolithic call, no decomposition
    min_children: int = 2         # a DECOMPOSE must produce at least this many
    max_children: int = 6
    max_concurrent_children: int = 4

    # ---- repair / escalation (Problem D) ----
    max_repair_rounds: int = 2    # per node, after contract check fails
    max_decompose_retries: int = 2  # per node, when composition check fails
    escalate_model: str | None = None  # stronger tier for ESCALATE; None = same

    # ---- per-node budgets ----
    node_max_iterations: int = 12
    root_max_iterations: int = 18
    max_budget_usd: float | None = None   # whole-task ceiling
    max_wallclock_s: float | None = 1800.0

    # ---- verification ----
    run_mypy: bool = False        # advisory second pass; AST walk is the record
    strict_undeclared_imports: bool = True

    # ---- model / backend ----
    backend: str = "openrouter"
    model: str = "deepseek/deepseek-chat"
    sub_model: str | None = None  # None = same as root
    base_url: str | None = None
    temperature: float = 0.2
    top_p: float | None = None
    max_output_tokens: int | None = None
    seed: int | None = None
    # Provider-specific chat-template switches (e.g. NVIDIA NIM's
    # {"thinking": bool, "reasoning_effort": str}). Passed through verbatim.
    chat_template_kwargs: dict[str, Any] = field(default_factory=dict)

    # ---- rate limiting (free endpoints) ----
    # Requests per minute allowed by the account. Enforced process-wide, so
    # concurrent build nodes cannot collectively exceed it.
    rpm_limit: int | None = None
    max_retries: int = 5
    retry_base_delay: float = 3.0
    request_timeout_s: float = 300.0

    # ---- environment ----
    rlm_environment: str = "local"
    extra: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    def describe(self) -> str:
        return (
            self.name
            + " [return=" + self.return_mode
            + " contract=" + self.contract_mode
            + " depth=" + self.depth_mode + "/" + str(self.max_depth)
            + " C=" + str(self.diagnostic_budget_tokens)
            + " model=" + self.model + "]"
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ArmConfig:
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})

    def copy_with(self, **overrides: Any) -> ArmConfig:
        d = self.to_dict()
        d.update(overrides)
        return ArmConfig.from_dict(d)

    # ------------------------------------------------------------------
    @classmethod
    def preset(cls, name: str) -> ArmConfig:
        """Named arms. ``list_presets()`` enumerates them."""
        if name not in PRESETS:
            raise ValueError(
                "unknown arm '" + name + "'. Available: " + ", ".join(sorted(PRESETS))
            )
        return PRESETS[name]()


def _p(name: str, **kw: Any) -> Any:
    def build() -> ArmConfig:
        return ArmConfig(name=name, **kw)

    return build


PRESETS: dict[str, Any] = {
    # ---- H1: the go/no-go gate -------------------------------------------
    # Identical in every respect except what a child returns upward.
    "h1-bounded": _p(
        "h1-bounded", return_mode="bounded", contract_mode="typed", depth_mode="adaptive"
    ),
    "h1-full-source": _p(
        "h1-full-source", return_mode="full_source", contract_mode="typed", depth_mode="adaptive"
    ),

    # ---- H3: contracts as the recursion interface ------------------------
    "h3-typed": _p("h3-typed", contract_mode="typed", return_mode="bounded"),
    "h3-prose": _p("h3-prose", contract_mode="prose", return_mode="bounded"),
    "h3-none": _p("h3-none", contract_mode="none", return_mode="bounded"),

    # ---- H2: adaptive vs fixed depth -------------------------------------
    "h2-adaptive": _p("h2-adaptive", depth_mode="adaptive", max_depth=3),
    "h2-depth0": _p("h2-depth0", depth_mode="fixed", max_depth=0),
    "h2-depth1": _p("h2-depth1", depth_mode="fixed", max_depth=1),
    "h2-depth2": _p("h2-depth2", depth_mode="fixed", max_depth=2),
    "h2-depth3": _p("h2-depth3", depth_mode="fixed", max_depth=3),

    # ---- external baselines ----------------------------------------------
    # Unmodified RLM: the paper's method, no contracts, no externalised output.
    "vanilla-rlm": _p("vanilla-rlm", contract_mode="none", return_mode="full_source", max_depth=1),
    # Single call, no REPL, no recursion: the lower bound.
    "base-lm": _p("base-lm", contract_mode="none", return_mode="full_source", max_depth=0,
                  depth_mode="fixed"),
}


def list_presets() -> list[str]:
    return sorted(PRESETS)
