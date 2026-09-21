"""λ is fractional and a tile is not: the rounding, and the gap it costs.

#13 shipped the simplest rule that could work — every tile of a class takes the
plan with the largest λ — and wrote the quota variant down as the fallback "for
when the gap measurement says the simple rule is not enough", with the
measurement itself TODO'd because it needs the LP bound and only the master has
one. The master has one now, and the measurement is decisive:

    LP objective (certified)   53,510.7
    argmax                          0.0     gap 100 %
    quota                      52,420.0     gap   2.04 %     (#13's target: 3 %)

Zero, because on a day-0 board the largest single weight is the IDLE column —
19.8 tiles of 25 — so argmax idles the farm and throws away a mix of 21 plans.

R007: every guard here was broken and seen red before it was trusted.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.planner.columns import (DAYS, ClassMix, Plan, assign_by_quota,
                                   assign_tiles, rounded_value)
from agent.obs import LOCKED_KEY

KEY = 10240


def _plan(value: float, hours: float = 1.0) -> Plan:
    return Plan(chains=(int(value),), value=value,
                rows={"labour": (hours,) * DAYS, "cash_out": (0.0,) * DAYS,
                      "wheat_net": (0.0,) * DAYS, "fert_net": (0.0,) * DAYS,
                      "stored": (0.0,) * DAYS})


def _mix(lam, values=None, count=None) -> dict[int, ClassMix]:
    values = values if values is not None else [float(i + 1) for i in range(len(lam))]
    return {KEY: ClassMix(class_key=KEY, count=count or int(round(sum(lam))),
                          plans=tuple(_plan(v) for v in values),
                          lam=tuple(lam))}


def test_quota_keeps_the_mix_where_argmax_collapses_it():
    """One heavy worthless plan and several light valuable ones.

    This is the day-0 board in miniature: the idle column carries the most
    weight, so argmax gives every tile a plan worth nothing.
    """
    mixes = _mix([6.0, 2.0, 1.0, 1.0], values=[0.0, 30.0, 20.0, 10.0])
    keys = [KEY] * 10

    by_max = assign_tiles(keys, mixes)
    by_quota = assign_by_quota(keys, mixes)

    assert rounded_value(by_max, mixes) == 0.0
    assert rounded_value(by_quota, mixes) == pytest.approx(2 * 30 + 20 + 10)
    assert len({c.plan_index for c in by_quota if c}) == 4


def test_integral_weights_are_reproduced_exactly():
    """When λ is already whole, the rounding must be the identity."""
    mixes = _mix([3.0, 2.0, 1.0])
    choices = assign_by_quota([KEY] * 6, mixes)
    taken = sorted(c.plan_index for c in choices if c)
    assert taken == [0, 0, 0, 1, 1, 2]


def test_the_remainder_goes_to_the_largest_fraction():
    mixes = _mix([1.2, 1.7, 1.1], count=4)
    choices = assign_by_quota([KEY] * 4, mixes)
    taken = sorted(c.plan_index for c in choices if c)
    # floors are 1,1,1 and the spare seat goes to plan 1 (fraction .7)
    assert taken == [0, 1, 1, 2]


def test_ties_fall_to_the_lowest_plan_index():
    """`assign_tiles` promises determinism; the quota rule keeps the promise."""
    mixes = _mix([0.5, 0.5, 0.5, 0.5], count=2)
    taken = sorted(c.plan_index for c in assign_by_quota([KEY] * 2, mixes) if c)
    assert taken == [0, 1]


def test_a_class_offered_more_weight_than_it_has_tiles_keeps_the_heaviest():
    """The LP is fractional and the board is not; the trim is from the bottom."""
    mixes = _mix([4.0, 3.0, 2.0], count=5)
    taken = sorted(c.plan_index for c in assign_by_quota([KEY] * 5, mixes) if c)
    assert taken.count(0) >= taken.count(2), taken
    assert len(taken) == 5


def test_a_locked_tile_never_takes_a_plan():
    """Working a LOCKED tile spends hours as a silent no-op (F042)."""
    mixes = _mix([2.0])
    choices = assign_by_quota([KEY, LOCKED_KEY, KEY, LOCKED_KEY], mixes)
    assert choices[1] is None and choices[3] is None
    assert choices[0] is not None and choices[2] is not None


def test_the_gap_on_a_real_board_is_inside_the_issues_target():
    """#13's own acceptance, run for the first time against a real LP bound."""
    from agent.obs import decode_world
    from agent.planner import master as M
    from agent.planner.inputs import GRAPH_PATH, load_contractor
    from agent.tile_dp.graph import TileGraph
    from offline_lab.kaggle_env import new_environment

    env = new_environment()
    env.reset(2)
    obs = env.state[0].observation
    graph = TileGraph.load(GRAPH_PATH)
    result = M.equilibrate(object(), obs, load_contractor(days=20),
                           M.supply_from_obs(obs), iter_cap=150)
    assert result.certified, result.stopped

    view = decode_world(obs, at_day_start=True,
                        graph_keys=frozenset(graph.key_index))
    inverse = {state: key for key, state in graph.key_index.items()}
    reps, _counts, _of_tile = result.classes
    mixes = M.to_mixes(result, 20, {i: inverse[s] for i, s in enumerate(reps)})
    keys = [int(k) for k in np.asarray(view.me.keys).reshape(-1)]

    quota = rounded_value(assign_by_quota(keys, mixes), mixes)
    gap = (result.objective - quota) / result.objective
    assert gap <= 0.03, f"integrality gap {gap:.2%} exceeds #13's 3 % target"
    assert rounded_value(assign_tiles(keys, mixes), mixes) < quota, (
        "argmax did not lose to the quota rule — the measurement that justifies "
        "the quota rule no longer holds")
