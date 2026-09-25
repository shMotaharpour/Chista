"""The land plan: how many of the next quadrant's tiles are open, as a function of the
soft decision to buy it.

The owner's design (this session): the manager should see the land purchase inside its own
LP, as `tiles per day` — a soft decision per day, smoothed so gradient-style methods can move
it, and rounded at the end to "buy on day d, use the land from the hour after the purchase".
The engine itself (kaggriculture.py:712-726) settles BUY_LAND atomically at the buy turn and
turns those cells from "LOCKED" to None on the spot, so the model must make the tiles open on
the very day of the purchase.

The contract is therefore:

    tiles_open(t; s_d for each candidate day d) = 25 * clip(sum of s_d for d <= t, 0, 1)

* `s_d` is the soft buy intensity for day d, in [0, 1]. One quadrant per plan: the forced
  order (F042) is an OUTER loop over quadrants, not a variable here.
* The tiles open on day t are what has been bought by the morning of day t. Buying on day d
  makes the tiles usable ON day d (the engine settles the purchase during day d's turns and
  the tiles are then ordinary empty tiles; the DP plans at day starts, and a tile bought at
  day d's hour 0 is a plain tile from that day's first planning point on).
* The hard model is `s` in {0,1}; the sigmoid/relu smoothing only exists to make the search
  continuous. Rounding owns the truth: at the end, `round_plan` turns s into one day.

Everything here is a pure function, testable without the engine.
"""
from __future__ import annotations

import numpy as np


def tiles_open(s: np.ndarray) -> np.ndarray:
    """Tiles of the next quadrant open on each day, from the soft buy curve `s`.

    `s` is (days,) in [0, 1]: the buy intensity per day. The running sum is the fraction
    bought by that day's planning point, and the quadrant has 25 tiles. Values above 1 (or
    below 0) are clipped, so a sloppy curve cannot invent tiles.

    A single-spike `s` (1 at day d, 0 elsewhere) is the hard plan "buy on day d": the tiles
    are open from day d itself, matching the engine's atomic settlement.
    """
    s = np.clip(np.asarray(s, dtype=np.float64), 0.0, 1.0)
    bought = np.clip(np.cumsum(s), 0.0, 1.0)
    return 25.0 * bought


def cash_cost(s: np.ndarray, price: int) -> np.ndarray:
    """What the purse pays on each day for this quadrant, row-ready for `cash_out`.

    The payment lands on the day the buy settles — the day `s` steps up. With the hard plan
    (a single spike at d) the whole price sits on day d.
    """
    s = np.clip(np.asarray(s, dtype=np.float64), 0.0, 1.0)
    return float(price) * s


def smooth(s_raw: np.ndarray) -> np.ndarray:
    """The soft version of a raw buy curve, for the search.

    `softplus`-shaped squash into [0, 1]: monotone, so a larger raw value never means a
    smaller buy, and near-zero when the raw value is very negative (the gradient can pull a
    day's buy down without it vanishing).
    """
    z = np.asarray(s_raw, dtype=np.float64)
    # numerically stable softplus, squashed to [0, 1]: sigma(z) = 1 / (1 + exp(-z)) is simpler
    # and is exactly the "sigmoid" the owner named; softplus is its un-squashed shape.
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def round_plan(s: np.ndarray) -> int | None:
    """The day to buy, from the soft curve — or None when the model says do not buy.

    The honest reading of the owner's design: the soft curve is a search device; the plan it
    names is the single day with the most buy in it, and only if that peak is meaningful
    (>= 0.5). Day 0 means "buy at the season's first planning point".
    """
    s = np.clip(np.asarray(s, dtype=np.float64), 0.0, 1.0)
    if s.size == 0 or float(s.max()) < 0.5:
        return None
    return int(np.argmax(s))
