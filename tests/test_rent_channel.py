"""The rent channel: every day a tile is OURS pays, and only LOCKED is free.

Charging an occupied slot and leaving a bare one free made the model
indifferent between working land it already holds and letting it stand -- the
opposite of getting productivity out of land. The rule is the owner's: holding
a tile costs, whether or not anyone works it that day, and the horizon a plan
runs over carries when its quadrant was bought. LOCKED tiles never enter the
graph, so they are the only ones with no rent.

Seen RED by deleting the subtraction: the shift assertions fail while the shape
checks still pass.
"""

from __future__ import annotations

import numpy as np

from agent.manager.core import _load_graph
from agent.tile_dp.contractor import TileContractor
from agent.world.model import N_RESOURCE

DAYS = 4


def _board():
    graph = _load_graph()
    contractor = TileContractor(graph, days=DAYS)
    return (contractor, np.full((DAYS, N_RESOURCE), 10.0),
            np.zeros((DAYS, N_RESOURCE)))


def test_every_day_of_a_held_tile_pays_its_rent() -> None:
    contractor, p, w = _board()
    base = contractor._base_rewards(p, w)
    rent = np.arange(1.0, DAYS + 1.0)
    shift = contractor._base_rewards(p, w, rent=rent) - base
    assert shift.shape == base.shape, "the rent changed the reward shape"
    for d in range(DAYS):
        assert np.allclose(shift[d], -rent[d]), (
            f"day {d}: a held tile did not pay its rent")


def test_the_channel_is_inert_when_it_is_not_handed_a_rent() -> None:
    contractor, p, w = _board()
    base = contractor._base_rewards(p, w)
    same = contractor._base_rewards(p, w, rent=None)
    assert np.array_equal(base, same), "rent=None must be bit-identical to no channel"
