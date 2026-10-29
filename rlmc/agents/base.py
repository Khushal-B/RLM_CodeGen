"""The common agent interface.

Every arm -- contract-recursive, vanilla RLM, plain single-call, and later
OpenHands or CodeTeam -- implements this one protocol and produces the same
artifact: a workspace directory containing an installable Python package, plus
a ``RunRecord``. The evaluator then scores workspaces without knowing or caring
which system produced them.

That separation is what makes the comparison honest. It also matches the
practical advice in the validation plan (sec. 6.5): run published baselines
through their own harness unmodified, and give your own method a thin runner
that drops the same artifact into the same place.
"""

from __future__ import annotations

import json
import platform
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

from rlmc.config import ArmConfig


@dataclass
class TaskSpec:
    """One benchmark task: a requirements document and an expected package."""

    task_id: str
    requirements: str
    package_name: str = ""
    difficulty: str = "unknown"        # easy | medium | hard
    target_loc: int = 0
    test_command: list[str] = field(default_factory=lambda: ["-m", "pytest", "-q"])
    tests_path: str | None = None      # tests live here, outside the workspace
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RunRecord:
    """Everything measured about one (task, arm, seed) run."""

    task_id: str
    arm: str
    seed: int
    ok: bool = False
    completed: bool = False            # produced an importable package at all
    workspace: str = ""
    wallclock_s: float = 0.0
    cost_usd: float = 0.0

    # contract metrics
    cvr: float = 0.0
    checked: int = 0
    violations: int = 0
    cvr_by_boundary: dict[str, float] = field(default_factory=dict)

    # H1 metrics
    root_context_tokens: int = 0
    peak_context_tokens: int = 0
    max_return_tokens: int = 0
    max_return_tokens_unbounded: int = 0
    bound_respected: bool = True

    # shape
    loc: int = 0
    files: int = 0
    nodes: int = 0
    max_depth: int = 0

    # evaluation (filled by the evaluator)
    tests_total: int = 0
    tests_passed: int = 0
    pass_rate: float = 0.0
    all_tests_pass: bool = False
    eval_error: str | None = None

    error: str | None = None
    config: dict[str, Any] = field(default_factory=dict)
    telemetry: dict[str, Any] = field(default_factory=dict)
    tree: dict[str, Any] = field(default_factory=dict)
    env: dict[str, Any] = field(default_factory=dict)
    started_at: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2, default=str), encoding="utf-8")
        return p

    @classmethod
    def load(cls, path: str | Path) -> RunRecord:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})


def environment_stamp() -> dict[str, Any]:
    """Recorded on every run so a result is reproducible from its record.

    The reporting protocol adopted at Meeting 12 requires the serving stack to
    be recorded alongside the number, precisely so a quantisation or backend
    difference can never be silently confounded with the effect being measured.
    """
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "timestamp": time.time(),
    }


class Agent(Protocol):
    """What every arm implements."""

    name: str

    def run(self, task: TaskSpec, workspace_dir: str, seed: int) -> RunRecord: ...


def make_agent(config: ArmConfig, progress: bool = False) -> Agent:
    """Build the agent an arm config asks for.

    ``max_depth == 0`` means no decomposition at all, which is the plain
    single-call lower bound rather than a degenerate contract run.
    """
    from rlmc.agents.contract_rlm import ContractRLMAgent
    from rlmc.agents.vanilla_rlm import BaseLMAgent, VanillaRLMAgent

    if config.name == "base-lm" or (config.max_depth == 0 and config.contract_mode == "none"):
        return BaseLMAgent(config)
    if config.name == "vanilla-rlm":
        return VanillaRLMAgent(config)
    return ContractRLMAgent(config, progress=progress)
