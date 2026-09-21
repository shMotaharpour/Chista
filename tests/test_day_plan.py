"""The loop closed: master, assignment, and the day layer's verdict.

`master` commits a day on a labour row that charges each worked tile its ops
plus the walk to reach it — a LOWER bound, deliberately, so the LP stays a
relaxation. `wsr` is what says whether the day can actually be walked, and
this is where the two meet.

R007: every guard here was broken and seen red before it was trusted.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.planner import columns as C
from agent.planner import day as D
from agent.planner import master as M
from agent.planner.colgen import classes_of
from agent.planner.inputs import GRAPH_PATH, load_contractor


@pytest.fixture(scope="module")
def board():
    from agent.obs import decode_world
    from agent.tile_dp.graph import TileGraph
    from offline_lab.kaggle_env import new_environment

    env = new_environment()
    env.reset(2)
    obs = env.state[0].observation
    graph = TileGraph.load(GRAPH_PATH)
    view = decode_world(obs, at_day_start=True,
                        graph_keys=frozenset(graph.key_index))
    keys = [int(k) for k in np.asarray(view.me.keys).reshape(-1)]
    owned = M._owned_states(object(), obs)
    _reps, _counts, of_tile = classes_of(
        owned, M._owned_distances(obs, C.shed_distance()))
    walker = iter(of_tile)
    class_of_tile = [next(walker, None) if k >= 0 else None for k in keys]
    return obs, load_contractor(days=20), M.supply_from_obs(obs), class_of_tile


def test_the_day_the_master_commits_can_actually_be_walked(board):
    """End to end: a certified LP mix, rounded, and carried by real workers."""
    obs, contractor, supply, class_of_tile = board
    result = D.plan(obs, contractor, supply, class_of_tile=class_of_tile,
                    iter_cap=200, hands=4, budget_s=2.0)

    assert result.master.certified, result.master.stopped
    assert result.day.chains, "the master committed nothing at all"
    assert result.day.complete, (
        f"wsr could not walk the committed day: {result.day.placed} of "
        f"{result.day.tasks} tasks, reason {result.day.reason!r}")
    assert result.day.tasks >= len(result.day.chains)


def test_the_committed_tiles_are_the_ones_beside_the_shed(board):
    """The travel term's visible consequence, asserted on the board itself."""
    obs, contractor, supply, class_of_tile = board
    result = D.plan(obs, contractor, supply, class_of_tile=class_of_tile,
                    iter_cap=200, hands=4, budget_s=2.0)
    steps = C.shed_distance()
    from agent.world.rules import BOARD_SIZE

    worked = [steps[y * BOARD_SIZE + x] for (x, y), _ops, _e in result.day.chains]
    assert worked, "nothing committed"
    assert max(worked) <= 3, (
        f"a tile {max(worked)} steps out was committed while nearer ones "
        f"idled: {worked}")


def test_a_declined_tile_is_not_a_dropped_one(board):
    """Every chain that reaches wsr is a chain the DP chose at these prices.

    The day carries only tiles whose day-0 chain is non-empty. A tile missing
    from it was re-planned and chose to wait, which is a different thing from
    being deleted to make the day fit — and the difference is the reason the
    master re-prices instead of truncating.
    """
    obs, contractor, supply, class_of_tile = board
    result = D.plan(obs, contractor, supply, class_of_tile=class_of_tile,
                    iter_cap=200, hands=4, budget_s=2.0)
    assigned = sum(1 for c in result.choices if c is not None)
    assert len(result.day.chains) <= assigned
    for _cell, ops, _entity in result.day.chains:
        assert ops, "an empty chain reached the day layer"
        assert "PASS" not in ops, (
            f"a do-nothing op reached the day layer as work: {ops} — an empty "
            f"chain must be left out, not padded into one")


def test_an_empty_day_fits_and_says_so():
    """No chains is a legal day, and it may not be reported as a failure."""
    fitted = D.fit((), hands=3, hours_committed=0.0)
    assert fitted.complete and fitted.tasks == 0 and fitted.reason == ""
    assert fitted.overhead == 1.0, (
        "no work is not an overhead of infinity, nor a division by zero")


def test_the_hours_only_ever_come_down(board):
    """wsr is a feasibility oracle here, not a calibration source.

    Dividing the supply by the ratio the route came in over does not converge —
    the ratio is a property of the plan, not of the board:

        hours 15.60   5 tiles   route 16 turns   1.14x
        hours 13.68   4 tiles   route 18 turns   1.29x

    Fewer hours bought fewer tiles and a LONGER route. So a day that FITS must
    leave the supply alone: it is evidence the row was not too generous, never
    evidence of how much slack is left.
    """
    obs, contractor, supply, class_of_tile = board
    result = D.plan(obs, contractor, supply, class_of_tile=class_of_tile,
                    iter_cap=200, hands=4, budget_s=2.0)
    assert result.day.complete
    assert result.overhead == 1.0, (
        f"the supply was corrected by {result.overhead:.3f}x on a day that "
        f"fits — a fitting day is not a measurement of the slack")
    assert result.solves == 1, (
        f"the master was solved {result.solves} times on a day that fit the "
        f"first time — the loop is calibrating against a fitting day")


def test_an_empty_day_zero_chain_is_left_out_not_padded():
    """A tile the DP declined is absent from the day, not present doing nothing.

    Asserted on a mix built here rather than on the board: no tile on a day-0
    board happens to have an empty day-0 chain, so the real fixture cannot
    exercise the branch and a guard that rides on it cannot fail.
    """

    from agent.planner.columns import DAYS, Choice, ClassMix, Plan

    def plan_with(chain_id):
        return Plan(chains=(chain_id,), value=1.0,
                    rows={n: (0.0,) * DAYS for n in
                          ("labour", "cash_out", "wheat_net", "fert_net",
                           "stored")})

    # chain 0 is the empty chain in the shipped registry; a real one is not.
    from agent.tile_dp.chains import chain_ops
    real = next(i for i in range(1, 64) if chain_ops(i))
    assert not chain_ops(0), "chain 0 is no longer the empty chain"

    mixes = {0: ClassMix(class_key=0, count=2,
                         plans=(plan_with(0), plan_with(real)),
                         lam=(1.0, 1.0))}
    chains = D.day_chains([Choice(0, 0), Choice(0, 1)], mixes, pool=())
    assert len(chains) == 1, (
        f"the empty chain was carried into the day: {chains}")
    assert chains[0][1] == tuple(chain_ops(real))
