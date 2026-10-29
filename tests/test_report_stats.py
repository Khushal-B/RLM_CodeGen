"""The H1 statistic must measure flatness, not monotonicity."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rlmc.metrics.report import correlation, growth_ratio, growth_slope, paired_bootstrap
from rlmc.agents.base import RunRecord


def test_slope_is_near_zero_for_a_flat_arm():
    # context creeps 900 -> 940 while the target grows 400 -> 2100 LOC
    pairs = [(400, 900), (900, 915), (1500, 930), (2100, 940)]
    assert abs(growth_slope(pairs)) < 0.05


def test_correlation_is_misleading_where_slope_is_not():
    """The reason slope replaced r as the headline statistic."""
    pairs = [(400, 900), (900, 915), (1500, 930), (2100, 940)]
    assert correlation(pairs) > 0.9        # looks like strong growth
    assert growth_slope(pairs) < 0.05      # but it is flat


def test_slope_is_large_for_a_growing_arm():
    pairs = [(400, 1600), (900, 3600), (1500, 6000), (2100, 8400)]
    assert growth_slope(pairs) > 3.0


def test_growth_ratio_is_scale_free():
    flat = [(400, 900), (2100, 940)]
    grows = [(400, 1600), (2100, 8400)]
    assert growth_ratio(flat) < 0.1
    assert growth_ratio(grows) > 0.8


def test_growth_stats_degrade_safely():
    assert growth_slope([]) == 0.0
    assert growth_slope([(1, 1)]) == 0.0
    assert growth_ratio([(5, 5), (5, 9)]) == 0.0     # no spread in x
    assert correlation([(1, 1), (2, 2)]) == 0.0      # too few points


def test_paired_bootstrap_pairs_by_task():
    recs = [
        RunRecord(task_id="a", arm="X", seed=0, pass_rate=0.6),
        RunRecord(task_id="b", arm="X", seed=0, pass_rate=0.4),
        RunRecord(task_id="a", arm="Y", seed=0, pass_rate=0.3),
        RunRecord(task_id="b", arm="Y", seed=0, pass_rate=0.2),
        RunRecord(task_id="c", arm="Y", seed=0, pass_rate=0.9),   # unpaired
    ]
    cmp = paired_bootstrap(recs, "X", "Y", iterations=500)
    assert cmp.n_pairs == 2          # 'c' is dropped, it has no X counterpart
    assert cmp.mean_diff == pytest.approx(0.25)
    assert cmp.wins == 2


def test_paired_bootstrap_averages_seeds_before_pairing():
    recs = [
        RunRecord(task_id="a", arm="X", seed=0, pass_rate=0.8),
        RunRecord(task_id="a", arm="X", seed=1, pass_rate=0.4),
        RunRecord(task_id="a", arm="Y", seed=0, pass_rate=0.2),
        RunRecord(task_id="a", arm="Y", seed=1, pass_rate=0.2),
    ]
    cmp = paired_bootstrap(recs, "X", "Y", iterations=500)
    assert cmp.n_pairs == 1
    assert cmp.mean_diff == pytest.approx(0.4)   # (0.6) - (0.2)


def test_no_shared_tasks_is_not_a_crash():
    recs = [RunRecord(task_id="a", arm="X", seed=0, pass_rate=0.5),
            RunRecord(task_id="b", arm="Y", seed=0, pass_rate=0.5)]
    cmp = paired_bootstrap(recs, "X", "Y", iterations=100)
    assert cmp.n_pairs == 0
    assert not cmp.excludes_zero


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
