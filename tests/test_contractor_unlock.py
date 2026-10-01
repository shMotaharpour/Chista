"""The lock channel: a tile that is LOCKED now and unlocks on day `k`.

The lock belongs to the tile, not to the tile graph. The graph only knows the
quadrant that starts open (F042: a locked tile is not in `graph.key_index` at
all), so only the unlock day can say when a tile becomes priceable. The rule —
the owner's design — is: while the tile is locked it counts no-op, and the day it
reaches `k` it is treated as an empty tile and its column is built from the graph
and the duals.

What is pinned here:
- a locked tile works nothing and is free on days `d < k`, its plan's locked
  prefix is the NO_ACTION chain, and its first worked day is `k`;
- the value it carries is `V[k, s]` — what an empty tile is worth from the day it
  exists — and not `V[0, s]`, the value of a tile that was there all along;
- from `k` on the plan is the DP's own argmax over the same graph edges;
- the default (no unlock days) is bit-identical to the pre-lock contractor, which
  is what makes the rule safe to add.

Seen RED by neutering the day mask in `TileContractor._recover_many`
(`active = ~locked[d]` forced all-true — the pre-lock walk):
`test_a_locked_tile_is_free_before_its_unlock_day` then reports the locked days
as worked. The failure text is in the commit message (R007).
"""

from __future__ import annotations

import numpy as np
import pytest

from agent.artifact import artifact_path
from agent.tile_dp.chains import NO_ACTION, chain_ops
from agent.tile_dp.contractor import HORIZON_DAYS, TileContractor
from agent.tile_dp.graph import TileGraph
from agent.world.model import N_RESOURCE, RESOURCE_ID

DAYS = HORIZON_DAYS
LABOR = RESOURCE_ID["LABOR"]
WHEAT = RESOURCE_ID["WHEAT"]
CARROT = RESOURCE_ID["CARROT"]

#: The graph's root: the bare tile a locked tile becomes on its unlock day.
BARE = 0
#: The unlock day the fixtures use for a locked tile.
K = 3
#: A nonzero walk distance, so `price_many` has two groups to keep apart.
DIST = 4


@pytest.fixture(scope="module")
def contractor() -> TileContractor:
    return TileContractor(TileGraph.load(artifact_path("tile_graph", ".npz")))


@pytest.fixture(scope="module")
def graph() -> TileGraph:
    return TileGraph.load(artifact_path("tile_graph", ".npz"))


def duals() -> tuple[np.ndarray, np.ndarray]:
    """Small non-negative duals: wheat, carrots and the labour hour priced."""
    p = np.zeros((DAYS, N_RESOURCE))
    w = np.zeros((DAYS, N_RESOURCE))
    p[:, WHEAT] = 25.0
    p[:, CARROT] = 35.0
    w[:, LABOR] = 3.0
    return p, w


def worked_days(board, tile: int = 0) -> list[int]:
    """Days the plan spends a worker or takes a unit off the tile."""
    return [d for d in range(board.days)
            if board.per_day_cost[tile, d, LABOR]
            or board.per_day_produce[tile, d].any()]


# ------------------------------------------------------------------ the rule

def test_a_locked_tile_is_free_before_its_unlock_day(contractor) -> None:
    """Days `d < k`: NO_ACTION, no worker, no input, no output, no entity."""
    p, w = duals()
    board = contractor.price(p, w, [BARE], unlock_days=[K])
    for d in range(K):
        _day, _state, chain = board.plans[0][d]
        assert chain_ops(chain) == NO_ACTION, (
            f"day {d} is locked but its chain is {chain_ops(chain)}")
        assert not board.per_day_cost[0, d].any(), (
            f"day {d} is locked but the plan spends {board.per_day_cost[0, d]}")
        assert not board.per_day_produce[0, d].any(), (
            f"day {d} is locked but the plan produces {board.per_day_produce[0, d]}")
        assert int(board.per_day_entity[0, d]) == 0, (
            f"day {d} is locked but the plan constructs entity "
            f"{int(board.per_day_entity[0, d])}")


def test_the_first_worked_day_is_the_unlock_day(contractor) -> None:
    """The column starts working the tile on `k`, not before and not after."""
    p, w = duals()
    board = contractor.price(p, w, [BARE], unlock_days=[K])
    worked = worked_days(board)
    assert worked, "a locked tile never worked, and a bare tile is worth working"
    assert worked[0] == K, (
        f"the tile unlocks on day {K} but the plan's first worked day is "
        f"{worked[0]} (all worked days: {worked})")

    # The same tile unlocked today may start on day 0: the shift is the lock's
    # doing and not the fixture's.
    now = contractor.price(p, w, [BARE])
    assert worked_days(now)[0] == 0, (
        f"the unlocked bare tile's plan starts on {worked_days(now)[0]}, not day 0")


def test_the_locked_value_is_the_value_of_an_empty_tile_from_the_unlock_day(
        contractor) -> None:
    """`V[k, s]`, not `V[0, s]`: the days before the unlock are not its days."""
    p, w = duals()
    board = contractor.price(p, w, [BARE], unlock_days=[K])
    assert board.tile_values[0] == board.values[K, BARE], (
        f"tile value {board.tile_values[0]} is not V[{K}, {BARE}] "
        f"({board.values[K, BARE]})")
    now = contractor.price(p, w, [BARE])
    assert board.tile_values[0] < now.tile_values[0], (
        f"a tile that exists for fewer days is worth {board.tile_values[0]} "
        f"against {now.tile_values[0]} for one that exists all season")


def test_the_plan_from_the_unlock_day_follows_the_argmax_it_priced(
        contractor, graph) -> None:
    """From `k` on, the recovered chain is the DP's own choice on the graph."""
    p, w = duals()
    board = contractor.price(p, w, [BARE], unlock_days=[K])
    checked = 0
    for d, state, chain_id in board.plans[0]:
        if d < K:
            continue
        lo, hi = graph.edges_of(state)
        cand = board.rewards[d][lo:hi] + board.values[d + 1][
            np.asarray(graph.edge_next[lo:hi], dtype=np.intp)]
        row = lo + int(np.argmax(cand))
        assert int(graph.edge_chain[row]) == chain_id, (d, state, chain_id)
        checked += 1
    assert checked == DAYS - K, "the plan was not checked from the unlock day on"


def test_a_tile_that_unlocks_after_the_season_contributes_nothing(
        contractor) -> None:
    """`k == days`: the tile never exists, so its column is empty and worth 0."""
    p, w = duals()
    board = contractor.price(p, w, [BARE], unlock_days=[DAYS])
    assert board.reduced_cost == 0.0 and board.tile_values[0] == 0.0
    assert not board.per_day_cost.any() and not board.per_day_produce.any()
    assert all(chain_ops(chain) == NO_ACTION for _d, _s, chain in board.plans[0])


# -------------------------------------------------- the default is untouched

def test_the_default_is_bit_identical_to_the_pre_lock_contractor(
        contractor) -> None:
    """No unlock days (and every zero) is the old pricing, field for field."""
    p, w = duals()
    owned = [BARE, 3, 4]
    base = contractor.price(p, w, owned)
    zero = contractor.price(p, w, owned, unlock_days=[0] * len(owned))

    assert base.reduced_cost == zero.reduced_cost
    assert base.plans == zero.plans
    for name in ("values", "columns", "produce", "tile_values",
                 "per_day_cost", "per_day_produce", "per_day_entity"):
        assert np.array_equal(getattr(base, name), getattr(zero, name)), (
            f"{name} differs between the pre-lock path and an all-zero unlock")

    groups = {0: [BARE, 3], DIST: [4, BARE]}
    many = contractor.price_many(p, w, groups)
    many_zero = contractor.price_many(
        p, w, groups,
        unlock_by_distance={h: [0] * len(v) for h, v in groups.items()})
    for h in groups:
        assert many[h].plans == many_zero[h].plans
        assert many[h].reduced_cost == many_zero[h].reduced_cost
        assert np.array_equal(many[h].per_day_cost, many_zero[h].per_day_cost)


# ------------------------------------------------------- the walk stays aligned

def test_two_tiles_of_one_group_keep_their_own_unlock_day(contractor) -> None:
    """One batched walk, per-tile locks: a global/group index slip is silent."""
    p, w = duals()
    pair = contractor.price(p, w, [BARE, BARE], unlock_days=[0, K])
    now = contractor.price(p, w, [BARE])
    locked = contractor.price(p, w, [BARE], unlock_days=[K])

    assert pair.plans[0] == now.plans[0], "the unlocked tile picked the lock's plan"
    assert pair.plans[1] == locked.plans[0], "the locked tile picked day 0's plan"
    assert pair.tile_values[0] == now.tile_values[0]
    assert pair.tile_values[1] == locked.tile_values[0]

    groups = {0: [BARE, BARE], DIST: [BARE]}
    locks = {0: [0, K], DIST: [K]}
    mixed = contractor.price_many(p, w, groups, unlock_by_distance=locks)
    for h in groups:
        ref = contractor.price(p, w, groups[h], travel_hours=h,
                               unlock_days=locks[h])
        board = mixed[h]
        assert board.plans == ref.plans, f"h={h}: plans"
        assert board.reduced_cost == ref.reduced_cost, f"h={h}: signal"
        assert np.array_equal(board.tile_values, ref.tile_values), f"h={h}: values"
        assert np.array_equal(board.per_day_cost, ref.per_day_cost), f"h={h}: cost"
        assert np.array_equal(board.per_day_produce, ref.per_day_produce), \
            f"h={h}: produce"


# ------------------------------------------------------------------- guards

def test_a_malformed_lock_map_is_refused(contractor) -> None:
    """A lock list that does not line up is a caller bug, never a silent default."""
    p, w = duals()
    with pytest.raises(ValueError, match="aligned one-to-one"):
        contractor.price(p, w, [BARE], unlock_days=[0, K])
    with pytest.raises(ValueError, match="no entry for distance"):
        contractor.price_many(p, w, {0: [BARE], DIST: [BARE]},
                              unlock_by_distance={0: [0]})
    with pytest.raises(ValueError, match="outside 0..30"):
        contractor.price(p, w, [BARE], unlock_days=[DAYS + 1])
    with pytest.raises(ValueError, match="outside 0..30"):
        contractor.price(p, w, [BARE], unlock_days=[-1])
