"""Phase 2 — the rival's action model, the demand forecast, and the hidden order.

Three separate things, kept apart on purpose:

* **The rival's action model.** `N[state, action] += 1` over their observed
  actions, Laplace-smoothed — estimation, not hidden-state learning, because the
  tracker already gives us their volumes. The state key is (good, day, price
  bucket) and the action is a sell-size bin. For the seven one-way goods a buy
  cannot exist, so the action space is one-dimensional and the counts converge
  fast; WHEAT and FERTILIZER carry one extra bin for a net buy.

* **The demand forecast.** Already-open shops are *facts* (a shop never closes),
  so their remaining consumption is deterministic. What is random is which shops
  open later: one every `UNLOCK_INTERVAL` days, type drawn with replacement from
  the 8 types. The *count* of those unlocks in a window is deterministic and only
  the basket is a draw, so both moments are exact in closed form — no sampling,
  and therefore nothing here needs a vectorising backend (F058 measured what JAX
  costs on the platform: ~1.7 s import + ~0.2 s JIT, and no contention penalty
  where SciPy pays +41 %; the choice is a measurement, not a default).

* **The hidden intra-turn order.** The observation gives aggregate volumes and an
  end-of-turn inventory, never the interleaving of the ten slots. What it leaves
  behind is the rival's realised *average* price, and that is public through their
  money balance. `infer_rival_slot` matches that number against the candidate
  slot positions. Two engine facts shape what it can say: same-index quotes are
  equal (so a slot index only matters relative to the other queue, and within one
  side of our volume every index prices identically), and if we do not trade in
  the turn at all, every candidate ties — the interleaving *is* the measurement.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from agent.world.model import PRODUCTS
from agent.world.prices import MARKET_PARAMS, price_of, price_vec
from agent.belief.schemas import (CENTER_INTERVAL, CENTER_PRODUCTS, DUAL, G_IX,
                                  MAX_ORDERS, SHOP_BASKET, SHOP_INTERVAL,
                                  SHOP_TYPES, UNLOCK_INTERVAL, field_of)
from agent.belief.tracker import FlowRecord, MarketTracker

#: The (9,) goods order, from the world's own name for it.
GOODS: tuple[str, ...] = PRODUCTS


class OpponentModel:
    """Empirical action counts over the rival, Laplace-smoothed."""

    BINS = (1.0, 3.0, 6.0)      # sell-size bin edges

    def __init__(self, alpha: float = 0.5, n_bins: int = 4) -> None:
        self.alpha = alpha
        self.n_bins = n_bins
        self.counts: dict[tuple[str, int, int], np.ndarray] = {}
        self.qty_sum: dict[tuple[str, int, int], np.ndarray] = {}

    @staticmethod
    def _bucket(price: int, base: int) -> int:
        ratio = (price / base) if base else 1.0
        return int(np.clip(round((ratio - 1.0) * 3), -3, 3))

    def _key(self, good: str, step: int, price: int) -> tuple[str, int, int]:
        return (good, int(step // 24), self._bucket(price, MARKET_PARAMS[good]["base"]))

    def _bin(self, qty: float) -> int:
        b = 0
        for edge in self.BINS:
            if qty >= edge:
                b += 1
        return b

    def observe(self, rec: FlowRecord, tracker: MarketTracker) -> None:
        for i, g in enumerate(GOODS):
            key = self._key(g, rec.step, int(tracker.prices[i]))
            arr = self.counts.setdefault(key, np.zeros(self.n_bins))
            qs = self.qty_sum.setdefault(key, np.zeros(self.n_bins))
            q = float(rec.rival_sales[i])
            b = self._bin(q)
            arr[b] += 1.0
            qs[b] += q
            if g in DUAL and rec.rival_buys[i] > 0:
                arr[-1] += 1.0            # the residual went negative: a net buy

    def policy(self, good: str, step: int, price: int) -> np.ndarray:
        """Action distribution for a state, smoothed; unseen states hold."""
        arr = self.counts.get(self._key(good, step, price))
        if arr is None:
            return np.array([1.0] + [0.0] * (self.n_bins - 1))
        p = arr + self.alpha
        return p / p.sum()

    def expected_sell(self, good: str, step: int, price: int) -> float:
        """Expected units the rival sells next turn in this state."""
        key = self._key(good, step, price)
        p = self.policy(good, step, price)
        qs = self.qty_sum.get(key, np.zeros(self.n_bins))
        counts = self.counts.get(key, np.zeros(self.n_bins))
        mean_qty = np.where(counts > 0, qs / np.maximum(counts, 1.0), 0.0)
        return float(p @ mean_qty)


def basket_matrix() -> np.ndarray:
    """(n_shop_types, 9) units consumed per shop event, per shop type."""
    M = np.zeros((len(SHOP_TYPES), len(GOODS)))
    for i, s in enumerate(SHOP_TYPES):
        items, mult = SHOP_BASKET[s]
        for item in items:
            M[i, G_IX[item]] += mult
    return M


def drain_forecast(obs: Any, steps_ahead: int,
                   type_prior: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Exact mean and sd of the town drain over the next `steps_ahead` turns."""
    step = int(field_of(obs, "step", 0))
    shops = list(field_of(field_of(obs, "town", {}) or {}, "unlocked_shops", []) or [])
    p = (np.full(len(SHOP_TYPES), 1.0 / len(SHOP_TYPES)) if type_prior is None
         else np.asarray(type_prior, dtype=float))
    p = p / p.sum()
    M = basket_matrix()

    steps = max(0, int(steps_ahead))
    shop_events = steps // SHOP_INTERVAL + 1
    center_events = steps // CENTER_INTERVAL + 1

    mean = np.zeros(len(GOODS))
    for s in shops:                                   # facts, not draws
        items, mult = SHOP_BASKET[s]
        for item in items:
            mean[G_IX[item]] += mult * shop_events
    var = np.zeros(len(GOODS))
    for k in range(step // 24 + 1, (step + steps) // 24 + 1):
        if k <= 0 or k % UNLOCK_INTERVAL != 0:        # the unlock schedule is known
            continue
        events = max(0, (step + steps - k * 24) // SHOP_INTERVAL + 1)
        if events == 0:
            continue
        contrib = M * events
        m = p @ contrib
        mean += m
        var += np.maximum(0.0, (p @ (contrib ** 2)) - m * m)
    for item in CENTER_PRODUCTS:                      # the centre eats every day
        mean[G_IX[item]] += center_events
    return mean, np.sqrt(var)


def quantile_price_floor(item: str, inventory: float, mean: float, sd: float,
                         z: float = 2.0) -> int:
    """The price if the drain falls `z` sd short — risk priced as a price."""
    return price_of(item, float(inventory) + z * float(sd) - float(mean))


def expected_price_curve(good: str, inventory: float, quantities: np.ndarray,
                         turns_ahead: int, drain_per_turn: float,
                         rival_per_turn: float) -> np.ndarray:
    """Price for each candidate quantity we sell, `turns_ahead` turns out."""
    eff = inventory + quantities - (drain_per_turn + rival_per_turn) * turns_ahead
    return price_vec(good, np.maximum(0.0, eff))


def infer_rival_slot(our_queue: dict[int, float], rival_volume: float, good: str,
                     inventory_start: float, realised_avg_price: float,
                     n_slots: int = 10) -> dict:
    """Infer which slot the rival's volume sat in, from the price it realised.

    `our_queue` maps slot index -> units we sold in that turn (known exactly).
    The candidate with the modelled average price closest to the observed one
    wins. `identifiable` is False when we sold nothing: then every candidate
    prices identically and the answer is meaningless, not merely noisy.
    """
    scored = []
    for k in range(n_slots):
        inv = float(inventory_start)
        theirs = float(rival_volume)
        revenue = 0.0
        for slot in range(n_slots):
            our_left = float(our_queue.get(slot, 0.0))
            their_now = theirs if slot == k else 0.0
            if slot == k:
                theirs = 0.0
            while our_left > 0 or their_now > 0:
                quote = price_of(good, max(0.0, inv))     # the same quote for both
                if our_left > 0:
                    inv += 1.0
                    our_left -= 1.0
                if their_now > 0:
                    revenue += quote
                    inv += 1.0
                    their_now -= 1.0
        modelled = revenue / rival_volume if rival_volume > 0 else 0.0
        scored.append((abs(modelled - realised_avg_price), k, modelled))
    scored.sort()
    return dict(estimate=scored[0][1],
                modelled_avg=float(scored[0][2]),
                error=float(scored[0][0]),
                ranking=[(k, round(m, 2)) for _e, k, m in scored[:5]],
                identifiable=bool(any(v > 0 for v in our_queue.values())))
