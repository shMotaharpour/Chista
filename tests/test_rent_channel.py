"""The rent channel: an occupied slot pays per day, a bare one does not.

The tile DP is handed two price vectors today and neither of them costs a tile
for OCCUPYING it, so a plan can hold land for free and the land's own price
never reaches the decision that uses it. This guard pins the channel that
carries it, in the one place every day's edge reward is formed.

Seen RED by removing the subtraction line: every assertion about the shift
fails while the shape checks still pass, which is the signature of a channel
that is wired but ignored.
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
    p = np.full((DAYS, N_RESOURCE), 10.0)
    w = np.zeros((DAYS, N_RESOURCE))
    occupied = np.zeros(int(graph.n_states), dtype=np.float64)
    occupied[:] = 1.0                      # every slot occupied: the strongest case
    return graph, contractor, p, w, occupied


def test_a_charged_slot_lowers_that_day_and_a_bare_one_does_not() -> None:
    graph, contractor, p, w, occupied = _board()
    base = contractor._base_rewards(p, w)
    rent = np.arange(1.0, DAYS + 1.0)
    charged = contractor._base_rewards(p, w, rent=rent, occupied=occupied)
    shift = charged - base
    assert shift.shape == base.shape, "the rent changed the reward shape"
    for d in range(DAYS):
        assert np.allclose(shift[d], -rent[d]), (
            f"day {d}: an occupied slot did not pay its rent")
    bare = np.zeros_like(occupied)
    uncharged = contractor._base_rewards(p, w, rent=rent, occupied=bare)
    assert np.allclose(uncharged, base), "a bare slot was charged rent"


def test_the_channel_is_inert_when_it_is_not_handed_a_rent() -> None:
    _, contractor, p, w, occupied = _board()
    base = contractor._base_rewards(p, w)
    same = contractor._base_rewards(p, w, rent=None, occupied=occupied)
    assert np.array_equal(base, same), "rent=None must be bit-identical to no channel"
