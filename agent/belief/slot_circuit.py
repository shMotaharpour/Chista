"""The slot circuit: the rival's trained distribution -> one sell schedule.

`agent/belief/solvers.py` owns the game matrix; `opponent.py` owns the
rival's action distribution; this module owns the WIRE between them — the
piece that was missing. One entry point:

    plan_day_slots(good, lot, obs, model) -> (schedule (24,), expected revenue)

  the units of `good` to offer at each hour of today, chosen to maximise
  the expected revenue against the rival's TRAINED hourly behaviour.

How it works (the closed circuit):

1.  CANDIDATE SCHEDULES — our side. A fixed shape family per lot size
    (dump-now, even spread, front third, back third, mid-day band,
    every-other-hour drip) — 6 candidates over 24 hours.
2.  RIVAL SCENARIOS — their side, from the trained model. The model gives
    P(sell bin | state) per good; each sell bin maps to its mean volume;
    each volume becomes 3 hourly placements (front / spread / back), and
    the hold bin is the zero schedule. These weighted scenarios REPLACE
    the hand-picked ones the solver used to need a caller for.
3.  EXPECTED PAYOFF — for every (candidate, scenario) pair, the revenue
    over the hourly walk with BOTH schedules' supply and the town's drain
    (engine sequencing: units, market, town). Expected revenue = the
    probability-weighted sum over scenarios. Against a rival whose
    schedule we cannot identify inside the turn (`infer_rival_slot` is
    honest about that), the rival's units are placed FIRST in each turn —
    the pessimistic half of the interleaving.
4.  BEST RESPONSE — the candidate with the highest expected revenue.

The output feeds `shed.market_queue` as the hour plan, replacing the
uniform spread.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from agent.belief.ladder import G_LO, S as CUM, good_index
from agent.belief.opponent import OpponentModel
from agent.belief.schemas import SHOP_BASKET, field_of
from agent.world.model import PRODUCTS
from agent.world.rules import (CENTER_SELL_INTERVAL_TURNS,
                               SHOP_SELL_INTERVAL_TURNS, TURNS_PER_DAY)

HOURS = TURNS_PER_DAY
_GI = {g: i for i, g in enumerate(PRODUCTS)}


def _drain_per_hour(obs) -> np.ndarray:
    """The town's per-hour drain of every good, from the open shops.

    Per good: the open shops' per-event units / 4 (a tick every 4 turns),
    plus the centre's 1 unit / 24 for every non-fertilizer product.
    """
    shops = list(field_of(field_of(obs, "town", {}) or {},
                          "unlocked_shops", []) or [])
    out = np.zeros(len(PRODUCTS))
    for name in shops:
        items, mult = SHOP_BASKET[name]
        for it in items:
            out[_GI[it]] += mult / SHOP_SELL_INTERVAL_TURNS
    for it in _center_items():
        out[_GI[it]] += 1.0 / CENTER_SELL_INTERVAL_TURNS
    return out


def _center_items() -> tuple:
    from agent.world.rules import TOWN_CENTER_PRODUCTS
    return TOWN_CENTER_PRODUCTS


def _candidate_schedules(lot: int) -> np.ndarray:
    """Our shape family over 24 hours: 6 candidates per lot.

    The engine's orders carry WHOLE units, so every candidate is built with
    an integer split (largest remainders) — a fractional hourly amount
    would silently sell nothing through the integer ladder.
    """
    lot = int(lot)

    def _spread(units: int, hours: list[int]) -> np.ndarray:
        s = np.zeros(HOURS)
        base, extra = divmod(units, len(hours))
        for k, h in enumerate(hours):
            s[h] = base + (1 if k < extra else 0)
        return s

    hours = list(range(HOURS))
    front = hours[:8]
    back = hours[-8:]
    mid = hours[4:16]
    drip = hours[::2]
    S = np.stack([
        _spread(lot, hours[:1]),          # dump now
        _spread(lot, hours),              # even spread
        _spread(lot, front),              # front third
        _spread(lot, back),               # back third
        _spread(lot, mid),                # mid-day band
        _spread(lot, drip),               # every-other-hour drip
    ])
    return S


def _rival_scenarios(model: OpponentModel, good: str, step: int, price: int
                     ) -> tuple[np.ndarray, np.ndarray]:
    """The rival's hourly placements from the trained distribution.

    Returns (schedules, weights): each row is a 24-hour rival schedule; the
    weights sum to 1 and ARE the model's probabilities.
    """
    p = model.policy(good, step, price)
    key = model._key(good, step, price)
    qs = np.asarray(model.qty_sum.get(key, np.zeros(model.n_bins)), dtype=float)
    counts = np.asarray(model.counts.get(key, np.zeros(model.n_bins)),
                        dtype=float)
    bin_means = np.where(counts > 0, qs / np.maximum(counts, 1.0), 0.0)
    # the rival's day volume in the model's expected-sell scale
    day_volume = float(model.expected_sell(good, step, price)) * TURNS_PER_DAY

    scenarios: list[np.ndarray] = []
    weights: list[float] = []
    # bin 0 = hold: the rival does nothing
    scenarios.append(np.zeros(HOURS)); weights.append(float(p[0]))
    # sell bins 1..n-2: the bin's share of the day volume, 3 placements
    for b in range(1, model.n_bins - 1):
        share = float(p[b])
        if share <= 0:
            continue
        vol = min(day_volume, 120.0)
        s_front = np.zeros(HOURS); s_front[:8] = vol / 8
        s_spread = np.full(HOURS, vol / HOURS)
        s_back = np.zeros(HOURS); s_back[-8:] = vol / 8
        w = share / 3.0
        scenarios += [s_front, s_spread, s_back]
        weights += [w, w, w]
    # net-buy bin (dual goods): no sell volume to place
    scenarios.append(np.zeros(HOURS))
    weights.append(float(p[-1]) if model.n_bins > 3 else 0.0)
    S = np.array(scenarios)
    W = np.array(weights)
    return S, W / W.sum()


def _revenue(good: str, start_inv: float, ours: np.ndarray,
             theirs: np.ndarray, drain: np.ndarray) -> float:
    """Coins of playing `ours` against `theirs` over the day.

    Per turn: the rival's units land FIRST (the pessimistic half of the
    interleaving — the trained model cannot identify their slot), then our
    units quote at the post-rival ladder, then the town drains.
    """
    i = _GI[good]
    inv = float(start_inv)
    revenue = 0.0
    drain_g = float(drain[_GI[good]])
    for t in range(HOURS):
        th = float(theirs[t])
        o = float(ours[t])
        if th > 0:
            inv += th
        if o > 0:
            lo = int(round(inv)) - G_LO
            lo = min(max(lo, 0), CUM.shape[1] - 1 - int(o))
            revenue += float(CUM[i, lo + int(o)] - CUM[i, lo])
            inv += o
        inv = max(0.0, inv - drain_g)
    return revenue


def plan_day_slots(good: str, lot: int, obs, model: OpponentModel
                   ) -> tuple[np.ndarray, float]:
    """The best hourly sell schedule for `lot` of `good` today.

    Returns (schedule (24,), expected revenue). The schedule is what
    `shed.orders_by_hour` turns into the per-hour queue.
    """
    step = int(field_of(obs, "step", 0))
    market = field_of(obs, "market", {}) or {}
    inv = float(dict(field_of(market, "inventory", {}) or {}).get(good, 0))
    price = int(dict(field_of(market, "prices", {}) or {}).get(good, 0))
    hour_now = step % TURNS_PER_DAY

    ours = _candidate_schedules(int(lot))
    theirs, weights = _rival_scenarios(model, good, step, price)
    drain = _drain_per_hour(obs)
    gi = _GI[good]

    A = np.zeros((len(ours), len(theirs)))
    for i in range(len(ours)):
        sched = np.zeros(HOURS)
        sched[hour_now:] = ours[i][: HOURS - hour_now]
        for j in range(len(theirs)):
            A[i, j] = _revenue(good, inv, sched, theirs[j], drain)
    expected = A @ weights
    best = int(np.argmax(expected))
    sched = np.zeros(HOURS)
    sched[hour_now:] = ours[best][: HOURS - hour_now]
    return sched, float(expected[best])
