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

    Engine order inside a day is units, market, town (F037): day d's sale
    quotes the inventory BEFORE that day's drain. Day d starts at
    `inventory - n - W[d]`, where `n` = units sold on the days before d
    and `W[d]` = the drains of the days before d — both subtract. Selling
    `k = j - n` units that day fetches `c(inv - n + k) - c(inv - n)` in
    cumsum terms (the ladder moves as the shared inventory falls), which
    is the transition the DP maxes per day — O(D * lot^2), no Python loop
    over units.
    Returns (per-day units, total coins).

    History: the pre-fix state added `n` instead of subtracting it AND
    reused the day-0 cumsum window for every state (A[m, j] - A[m, m]),
    which priced day d's sale as if the whole day-d basket moved down one
    shared ladder — the two errors cancelled only when the plan sold
    nothing early. Found by enumerating every split of a 12-lot and
    pricing each through `sell_coins`; the DP deferred everything to the
    last day and paid 2,646 where 2,696 was available (MILK, 9,950, 5
    days, drain 1/day).
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
        start = int(inventory) - n - int(W[d])         # sold + drained subtract
        # c(x) at cumsum index; selling k units from `start` fetches
        # c(start + k) - c(start). A[j, k] = c(start_j + k) - c(start_j).
        A = np.stack([S[i, s - G_LO: s - G_LO + lot + 1] - S[i, s - G_LO]
                      for s in start])                 # A[j, k]: j sold so far
        # transition: from state n=m to n=j (> m), sell k = j - m TODAY:
        # coins = c(start_m + k) - c(start_m) = A[m, k] with k = j - m.
        B = np.full((lot + 1, lot + 1), -np.inf)
        for m in range(lot + 1):
            k = np.arange(0, lot + 1 - m)              # sell k today
            B[m, m:] = A[m, k]                         # A indexed by TODAY's units
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


def plan_coins(good: str, inventory: int, plan: "np.ndarray | list[int]",
               drains: np.ndarray) -> int:
    """Coins a SPECIFIC multi-day sell plan fetches — the manager's what-if.

    `plan[d]` is the units to sell on day d (engine order per day: the
    sale quotes the inventory before that day's drain, F037). This is the
    evaluator side of `split_days`: the DP returns the optimal plan, this
    prices ANY plan — 3/day vs 1/day differ here because the ladder moves
    with the shared inventory and the town drains between the days.
    Guards: `plan_coins` on `split_days`' own output equals its reported
    total, and a deferring shape (all on the last day) prices BELOW the
    optimum on a drained board.
    """
    i = _IX[good]
    drains = np.asarray(drains, dtype=np.int64)
    plan = [int(k) for k in plan]
    if len(plan) != len(drains):
        raise ValueError(
            f"plan has {len(plan)} days against {len(drains)} drains")
    coins = 0
    inv = int(inventory)
    for d, k in enumerate(plan):
        k = max(0, min(k, inv))
        coins += sell_coins(good, inv, k)
        inv = inv - k - max(0, int(drains[d]))
    return coins
