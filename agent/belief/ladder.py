"""The market's arithmetic as arrays: the price ladder, and the exact day split.

`agent/world/prices.py` owns the price formula; this module owns its INVERSE
uses — given a lot and a horizon, what does selling cost or fetch, and where
should the units sit. Everything here is closed-form over one cumulative-sum
table per good, built once at import (36,864 prices x 9 goods, ~1 ms):

    P[g, v]    the engine quote at inventory `v`  (bit-identical, parity-tested)
    S[g, v]    its cumsum: the coins of the FIRST v units sold from 0
    FLOOR[g]   the first inventory whose quote is the $1 floor (None if never)

Three identities do all the work, each proved against the engine's own
unit-by-unit walk (tests/test_market_ladder.py):

  * sell ladder   n units from inventory i fetch S[i+n] - S[i], and only
    i+n - i* units of supply land, where i* is the floor point — the engine
    stalls at $1 (`_commit_unit`: a sale at the floor adds no supply);
  * buy ladder    n units at inventory i cost S[i] - S[i-n] (a buy is quoted
    at `price(I - 1)`, so the window sits one below the walk);
  * day split     the exact optimum of "sell `lot` units across D days, the
    drains in between" — a shortest-path DP over `units sold so far`, whose
    transitions are ladder windows. The naive per-step greedy is measured in
    the tests and loses (the drain couples the days: a unit sold today is
    quoted on tomorrow's start too).

The DP is O(D * lot^2) fully vectorized — one (lot+1, lot+1) matrix per day —
and its answer equals brute-force enumeration on randomized cases in the
tests. This is what prices a schedule before it is played, and what the sell
allocator hands the order book.
"""

from __future__ import annotations

import numpy as np

from agent.world.model import PRODUCTS
from agent.world.prices import (HINGE_GAIN, MARKET_I0, MARKET_PARAMS,
                                PRICE_FLOOR)

#: The quote table: (9,) goods x (36'864,) inventories, [G_LO, G_HI).
#: The bounds cover every reachable inventory: the market starts at I0 = 10'000,
#: its drains are tens of units a day, and the largest scripted basket is the
#: shed's own capacity. G_LO is the margin below 0 where the engine's own
#: formula is still well-defined (its shapes clamp negatives to 0 except the
#: two the parity test exercises).
G_LO: int = -4096
G_HI: int = 32768

_N = len(PRODUCTS)
_shape_fn = {
    "linear": lambda x: x,
    "sq": lambda x: x * x,
    "sqrt": lambda x: np.sqrt(np.maximum(0.0, x)),
    "log": lambda x: np.log1p(np.maximum(x, 0.0)),
    "log10": lambda x: np.log10(1.0 + np.maximum(x, 0.0)),
    "hinge": lambda x, t: (lambda u: u + HINGE_GAIN * np.maximum(0.0, u - 1.0) ** 2)(
        x / t),
}
_base = np.array([MARKET_PARAMS[g]["base"] for g in PRODUCTS], dtype=np.float64)
_T = np.array([MARKET_PARAMS[g]["T"] for g in PRODUCTS], dtype=np.float64)


def _quotes(gi: int) -> np.ndarray:
    """Row `gi` of P: the engine formula, vectorized over the grid.

    Below I0 the shape term is ADDED (scarcity pushes the price up); above it
    the term is SUBTRACTED (glut pushes it down) — kaggriculture.py:192-206.
    """
    p = MARKET_PARAMS[PRODUCTS[gi]]
    inv = np.arange(G_LO, G_HI, dtype=np.float64)
    below = inv < MARKET_I0
    x = np.where(below, MARKET_I0 - inv, inv - MARKET_I0)
    vals = np.empty(len(inv), dtype=np.float64)
    t = _T[gi]
    for sel, func, tgt, sign in ((below, p["below_func"], p["below_target"], +1.0),
                                 (~below, p["above_func"], p["above_target"], -1.0)):
        if not sel.any():
            continue
        shape = _shape_fn[func]
        f_t = float(shape(np.array([t]), t)[0]) if func == "hinge" \
            else float(shape(np.array([t]))[0])
        amp = tgt * _base[gi] / f_t
        fx = shape(x[sel], t) if func == "hinge" else shape(x[sel])
        vals[sel] = _base[gi] + sign * amp * fx
    return np.maximum(PRICE_FLOOR, np.round(vals)).astype(np.int64)


P: np.ndarray = np.stack([_quotes(i) for i in range(_N)])
S: np.ndarray = np.zeros((_N, P.shape[1] + 1), dtype=np.int64)
S[:, 1:] = np.cumsum(P, axis=1)

#: First fully-clamped inventory per good, or None (the floor is never hit).
FLOOR: tuple[int | None, ...] = tuple(
    None if (hit := np.nonzero(P[i] <= PRICE_FLOOR)[0]).size == 0
    else int(hit[0]) + G_LO for i in range(_N))
_FLOOR_IX = np.array([G_HI - G_LO if f is None else f - G_LO for f in FLOOR])

_IX = {g: i for i, g in enumerate(PRODUCTS)}


def good_index(good: str) -> int:
    return _IX[good]


def supply_after_sell(good: str, inventory: int, units: int) -> int:
    """Units of supply a sale of `units` from `inventory` lands: the engine
    stalls at the floor (`_commit_unit` :659-660)."""
    i, lo = _IX[good], inventory - G_LO
    stall = int(_FLOOR_IX[i]) - lo
    return int(min(units, max(0, stall)))


def sell_coins(good: str, inventory: int, units: int) -> int:
    """Coins `units` fetch selling from `inventory`, floor stall included."""
    i, lo = _IX[good], inventory - G_LO
    stall = int(_FLOOR_IX[i]) - lo
    if units <= max(0, stall):
        return int(S[i, lo + units] - S[i, lo])
    head = int(S[i, max(0, int(_FLOOR_IX[i]))] - S[i, lo]) if stall > 0 else 0
    return head + (units - max(0, stall))          # $1 each past the floor


def buy_coins(good: str, inventory: int, units: int) -> int:
    """Coins `units` cost buying down from `inventory` (quoted at I - 1)."""
    i, hi = _IX[good], inventory - G_LO
    return int(S[i, hi] - S[i, hi - units])


def split_days(good: str, inventory: int, lot: int, drains: np.ndarray
               ) -> tuple[np.ndarray, int]:
    """The exact best day split of a `lot`, drains between the days.

    State = units sold before the day; the day's inventory is then
    `inventory + m - W[d]` (W = the drains of the days before d). The
    transition matrix is the ladder window per state, so one matmul-shaped
    max per day — O(D * lot^2) with no Python loop over units.
    Returns (per-day units, total coins).
    """
    i = _IX[good]
    drains = np.asarray(drains, dtype=np.int64)
    D = len(drains)
    W = np.concatenate([[0], np.cumsum(drains)[:-1]])
    lot = int(lot)
    dp = np.full(lot + 1, -np.inf)
    dp[0] = 0.0
    choice = np.zeros((D, lot + 1), dtype=np.int64)
    n = np.arange(lot + 1)
    for d in range(D):
        start = int(inventory) + n - int(W[d])         # start inventory per state
        A = np.stack([S[i, s - G_LO: s - G_LO + lot + 1] - S[i, s - G_LO]
                      for s in start])                 # A[m, j]: m-th state's cumsum
        B = np.full((lot + 1, lot + 1), -np.inf)
        for m in range(lot + 1):
            B[m, m:] = A[m, m:] - A[m, m]              # sell k-m from state m
        cand = dp[:, None] + B
        choice[d] = np.argmax(cand, axis=0)
        dp = cand.max(axis=0)
    xs = np.zeros(D, dtype=np.int64)
    m = lot
    for d in range(D - 1, -1, -1):
        pm = int(choice[d, m])
        xs[d] = m - pm
        m = pm
    return xs, int(dp[lot])
