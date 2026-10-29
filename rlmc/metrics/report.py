"""Aggregation and the paired statistics the ablation needs.

Problem B of the validation plan: with n=20 and a binary outcome the standard
error swamps a real effect. The fixes implemented here are the four it asks for:

1. continuous average test pass rate rather than binary success
2. a paired design -- arms compared per task, never as two independent groups
3. bootstrap confidence intervals on the paired difference
4. seed-to-seed variance reported alongside the effect

``paired_bootstrap`` makes no distributional assumption, which matters because
pass rates are bounded, lumpy and nowhere near normal at this sample size.
"""

from __future__ import annotations

import json
import math
import random
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rlmc.agents.base import RunRecord


def load_records(runs_dir: str | Path) -> list[RunRecord]:
    """Load every run record under a directory."""
    out: list[RunRecord] = []
    for p in sorted(Path(runs_dir).rglob("record.json")):
        try:
            out.append(RunRecord.load(p))
        except Exception:
            continue
    return out


@dataclass
class ArmSummary:
    arm: str
    n: int = 0
    mean_pass_rate: float = 0.0
    pass_at_1: float = 0.0            # fraction of runs where all tests pass
    completion_rate: float = 0.0      # produced an importable package
    mean_cvr: float = 0.0
    mean_cost_usd: float = 0.0
    mean_wallclock_s: float = 0.0
    mean_loc: float = 0.0
    mean_root_context: float = 0.0
    max_root_context: int = 0
    max_return_tokens: int = 0
    bound_respected: bool = True
    seed_variance: float = 0.0
    by_difficulty: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = self.__dict__.copy()
        return d


def summarise_arm(records: list[RunRecord], arm: str, task_difficulty: dict[str, str] | None = None) -> ArmSummary:
    rows = [r for r in records if r.arm == arm]
    if not rows:
        return ArmSummary(arm=arm)

    s = ArmSummary(arm=arm, n=len(rows))
    s.mean_pass_rate = _mean([r.pass_rate for r in rows])
    s.pass_at_1 = _mean([1.0 if r.all_tests_pass else 0.0 for r in rows])
    s.completion_rate = _mean([1.0 if r.completed else 0.0 for r in rows])
    s.mean_cvr = _mean([r.cvr for r in rows])
    s.mean_cost_usd = _mean([r.cost_usd for r in rows])
    s.mean_wallclock_s = _mean([r.wallclock_s for r in rows])
    s.mean_loc = _mean([float(r.loc) for r in rows])
    s.mean_root_context = _mean([float(r.root_context_tokens) for r in rows])
    s.max_root_context = max((r.root_context_tokens for r in rows), default=0)
    s.max_return_tokens = max((r.max_return_tokens for r in rows), default=0)
    s.bound_respected = all(r.bound_respected for r in rows)

    # seed-to-seed variance: mean per seed, then variance across seeds
    by_seed: dict[int, list[float]] = defaultdict(list)
    for r in rows:
        by_seed[r.seed].append(r.pass_rate)
    seed_means = [_mean(v) for v in by_seed.values()]
    s.seed_variance = statistics.pvariance(seed_means) if len(seed_means) > 1 else 0.0

    if task_difficulty:
        by_diff: dict[str, list[float]] = defaultdict(list)
        for r in rows:
            by_diff[task_difficulty.get(r.task_id, "unknown")].append(r.pass_rate)
        s.by_difficulty = {k: round(_mean(v), 4) for k, v in sorted(by_diff.items())}
    return s


@dataclass
class PairedComparison:
    """A vs B, compared per task."""

    arm_a: str
    arm_b: str
    n_pairs: int
    mean_diff: float
    ci_low: float
    ci_high: float
    excludes_zero: bool
    wins: int
    losses: int
    ties: int
    per_task: dict[str, float] = field(default_factory=dict)

    def verdict(self) -> str:
        if not self.excludes_zero:
            return "no significant difference (95% CI includes 0)"
        direction = "better" if self.mean_diff > 0 else "worse"
        return self.arm_a + " is " + direction + " than " + self.arm_b

    def to_dict(self) -> dict[str, Any]:
        d = self.__dict__.copy()
        d["verdict"] = self.verdict()
        return d


def paired_bootstrap(
    records: list[RunRecord],
    arm_a: str,
    arm_b: str,
    metric: str = "pass_rate",
    iterations: int = 10_000,
    seed: int = 0,
    task_filter: set[str] | None = None,
) -> PairedComparison:
    """Paired bootstrap CI on the per-task difference between two arms.

    Pairing cancels out task difficulty, which varies enormously and would
    otherwise dominate the variance. Runs of the same task under different seeds
    are averaged before pairing, so each task contributes exactly one difference.
    """
    a_vals: dict[str, list[float]] = defaultdict(list)
    b_vals: dict[str, list[float]] = defaultdict(list)
    for r in records:
        if task_filter and r.task_id not in task_filter:
            continue
        value = float(getattr(r, metric, 0.0) or 0.0)
        if r.arm == arm_a:
            a_vals[r.task_id].append(value)
        elif r.arm == arm_b:
            b_vals[r.task_id].append(value)

    shared = sorted(set(a_vals) & set(b_vals))
    diffs = [_mean(a_vals[t]) - _mean(b_vals[t]) for t in shared]
    per_task = {t: round(d, 4) for t, d in zip(shared, diffs)}

    if not diffs:
        return PairedComparison(arm_a, arm_b, 0, 0.0, 0.0, 0.0, False, 0, 0, 0, {})

    rng = random.Random(seed)
    n = len(diffs)
    means = []
    for _ in range(iterations):
        sample = [diffs[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo = means[int(0.025 * len(means))]
    hi = means[min(len(means) - 1, int(0.975 * len(means)))]

    return PairedComparison(
        arm_a=arm_a,
        arm_b=arm_b,
        n_pairs=n,
        mean_diff=round(sum(diffs) / n, 4),
        ci_low=round(lo, 4),
        ci_high=round(hi, 4),
        excludes_zero=(lo > 0 or hi < 0),
        wins=sum(1 for d in diffs if d > 1e-9),
        losses=sum(1 for d in diffs if d < -1e-9),
        ties=sum(1 for d in diffs if abs(d) <= 1e-9),
        per_task=per_task,
    )


def context_growth(records: list[RunRecord], arm: str) -> list[tuple[int, int]]:
    """(target LOC, root context tokens) pairs -- the H1 plot.

    The claim is that this is flat for the externalised arm and grows for the
    control. Returned sorted by LOC so it can be plotted directly.
    """
    rows = [(r.loc, r.root_context_tokens) for r in records if r.arm == arm and r.loc > 0]
    return sorted(rows)


def correlation(pairs: list[tuple[int, int]]) -> float:
    """Pearson r between target size and root context.

    NOTE: r is the WRONG headline statistic for H1 and is kept only as a
    secondary diagnostic. H1 claims root context is *flat* in target size, and r
    measures monotonicity, not flatness: an arm whose context creeps from 900 to
    940 tokens while the target grows from 400 to 2100 LOC has r ~ 0.9 and is,
    for every practical purpose, flat. Use ``growth_slope`` for the claim.
    """
    if len(pairs) < 3:
        return 0.0
    xs = [float(a) for a, _ in pairs]
    ys = [float(b) for _, b in pairs]
    mx, my = _mean(xs), _mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    return round(num / (dx * dy), 4) if dx > 0 and dy > 0 else 0.0


def growth_slope(pairs: list[tuple[int, int]]) -> float:
    """Least-squares slope: root-context tokens gained per target LOC.

    This is the H1 statistic. The claim is that the externalised arm's slope is
    ~0 (root context constant in subtree size) while the source-returning
    control's slope is clearly positive, because everything its children write
    flows back up into the root's context.
    """
    if len(pairs) < 2:
        return 0.0
    xs = [float(a) for a, _ in pairs]
    ys = [float(b) for _, b in pairs]
    mx, my = _mean(xs), _mean(ys)
    denom = sum((x - mx) ** 2 for x in xs)
    if denom <= 0:
        return 0.0
    return round(sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / denom, 5)


def growth_ratio(pairs: list[tuple[int, int]]) -> float:
    """Context spread relative to target spread, both as fold-changes.

    ``(ctx_max/ctx_min - 1) / (loc_max/loc_min - 1)``. Scale-free, so it reads
    without knowing absolute token counts: ~0 means context barely moved while
    the target grew a lot (H1 holds); ~1 means context tracked the target
    proportionally (H1 fails).
    """
    if len(pairs) < 2:
        return 0.0
    xs = [a for a, _ in pairs if a > 0]
    ys = [b for _, b in pairs if b > 0]
    if len(xs) < 2 or len(ys) < 2:
        return 0.0
    loc_span = max(xs) / min(xs)
    ctx_span = max(ys) / min(ys)
    if loc_span <= 1.0:
        return 0.0
    return round((ctx_span - 1.0) / (loc_span - 1.0), 4)


def build_report(runs_dir: str | Path, compare: tuple[str, str] | None = None) -> dict[str, Any]:
    records = load_records(runs_dir)
    arms = sorted({r.arm for r in records})
    difficulty = {r.task_id: r.config.get("difficulty", "unknown") for r in records}

    report: dict[str, Any] = {
        "runs_dir": str(runs_dir),
        "n_records": len(records),
        "arms": {a: summarise_arm(records, a, difficulty).to_dict() for a in arms},
    }
    for a in arms:
        pairs = context_growth(records, a)
        report["arms"][a]["context_vs_loc"] = pairs
        report["arms"][a]["context_loc_correlation"] = correlation(pairs)
        # The H1 statistic: tokens of root context per LOC of target.
        report["arms"][a]["context_growth_slope"] = growth_slope(pairs)
        report["arms"][a]["context_growth_ratio"] = growth_ratio(pairs)

    if compare and compare[0] in arms and compare[1] in arms:
        report["paired"] = paired_bootstrap(records, compare[0], compare[1]).to_dict()
        report["paired_cvr"] = paired_bootstrap(
            records, compare[0], compare[1], metric="cvr"
        ).to_dict()
    return report


def render_report(report: dict[str, Any]) -> str:
    """Human-readable summary for the terminal."""
    lines: list[str] = []
    lines.append("Records: " + str(report.get("n_records", 0)))
    lines.append("")
    header = (
        f"{'arm':<18}{'n':>4}{'pass':>8}{'pass@1':>8}{'compl':>7}"
        f"{'CVR':>7}{'ctx':>8}{'tok/LOC':>10}{'growth':>9}{'cost$':>9}"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for arm, s in sorted(report.get("arms", {}).items()):
        lines.append(
            f"{arm:<18}{s['n']:>4}{s['mean_pass_rate']:>8.3f}{s['pass_at_1']:>8.3f}"
            f"{s['completion_rate']:>7.2f}{s['mean_cvr']:>7.3f}"
            f"{s['mean_root_context']:>8.0f}{s.get('context_growth_slope', 0):>10.3f}"
            f"{s.get('context_growth_ratio', 0):>9.3f}{s['mean_cost_usd']:>9.4f}"
        )
    paired = report.get("paired")
    if paired:
        lines.append("")
        lines.append(
            "Paired (" + paired["arm_a"] + " - " + paired["arm_b"] + ") on pass_rate: "
            + "mean diff " + str(paired["mean_diff"])
            + "  95% CI [" + str(paired["ci_low"]) + ", " + str(paired["ci_high"]) + "]"
            + "  n=" + str(paired["n_pairs"])
        )
        if paired["n_pairs"] < 5:
            lines.append(
                "  WARNING: only " + str(paired["n_pairs"]) + " paired task(s). The "
                "bootstrap CI is not meaningful below ~10 pairs; treat this as a "
                "smoke test, not evidence."
            )
        lines.append("  W/T/L: " + str(paired["wins"]) + "/" + str(paired["ties"]) + "/"
                     + str(paired["losses"]) + "   -> " + paired["verdict"])
    return "\n".join(lines)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def save_report(report: dict[str, Any], path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return p
