"""Phase 2 — the rival's action model, the demand forecast, and the hidden order.

Three separate things, kept apart on purpose:

* **The rival's action model.** `N[state, action] += 1` over their observed
  actions, Laplace-smoothed — estimation, not hidden-state learning, because the
  tracker already gives us their volumes. The SHIPPED artifact's state key is
  PER-GOOD: (good, demand bucket, price bucket) for the six goods with a
  variable shop demand (CARROT, TOMATO, STRAWBERRY, MILK, EGG, WOOL — their
  behaviour tracks which shops are open), and (good, day, price bucket) for
  WHEAT, MELON, FERTILIZER. The reason is what each dimension carries:
  MELON has no shop buyer at all (its only buyer is the centre, 1/day flat),
  FERTILIZER no shop buyer and no centre either, and WHEAT's demand is so
  wide a day dimension tracks its calendar; the six demand-keyed goods lose
  nothing by dropping the day (their shops' unlock schedule is near-uniform)
  and gain the demand signal — measured, the per-good scheme beat the
  global key 1.2714 -> 1.2104 log loss with FEWER states (337 vs 943).
  The action is a sell-size bin. For the seven one-way goods a buy cannot
  exist, so the action space is one-dimensional and the counts converge
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
from agent.world.rules import TURNS_PER_DAY
from agent.world.prices import MARKET_PARAMS, price_of, price_vec
from agent.belief.schemas import (CENTER_INTERVAL, CENTER_PRODUCTS, DUAL, G_IX,
                                  MAX_ORDERS, SHOP_BASKET, SHOP_INTERVAL,
                                  SHOP_TYPES, UNLOCK_INTERVAL, field_of)
from agent.belief.tracker import FlowRecord, MarketTracker

#: The (9,) goods order, from the world's own name for it.
GOODS: tuple[str, ...] = PRODUCTS

#: The trained artifact, loaded once. The builder
#: (`offline_lab.build.opponent_model`) aggregates the replay store into the
#: SAME key/bin structure, so a state seen in the corpus primes the policy
#: before the first online observation; the agent keeps counting on top of it.


def _load_trained() -> dict[tuple, np.ndarray] | None:
    """The corpus counts as {key: (n_bins,) float array}, or None.

    Keys are (good, day, price_bucket, activity); an artifact written
    without the activity column reads as activity = -1 (the plain
    3-tuple state, the pre-#65 shape).
    """
    from agent.artifact import ARTIFACT_DIR

    npz = ARTIFACT_DIR / "opponent_counts.npz"
    if not npz.exists():
        return None
    with np.load(npz, allow_pickle=False) as data:
        act = data["activity"] if "activity" in data.files else None
        return {(str(g), int(d), int(b), int(a) if act is not None else -1):
                np.asarray(c, dtype=float)
                for g, d, b, a, c in zip(
                    data["goods"], data["days"], data["buckets"],
                    act if act is not None else [-1] * len(data["goods"]),
                    data["counts"], strict=True)}


def _load_trained_qty() -> dict[tuple, np.ndarray] | None:
    from agent.artifact import ARTIFACT_DIR

    npz = ARTIFACT_DIR / "opponent_counts.npz"
    if not npz.exists():
        return None
    with np.load(npz, allow_pickle=False) as data:
        act = data["activity"] if "activity" in data.files else None
        return {(str(g), int(d), int(b), int(a) if act is not None else -1):
                np.asarray(q, dtype=float)
                for g, d, b, a, q in zip(
                    data["goods"], data["days"], data["buckets"],
                    act if act is not None else [-1] * len(data["goods"]),
                    data["qty_sum"], strict=True)}


class OpponentModel:
    """Empirical action counts over the rival, Laplace-smoothed.

    Smoothing is HIERARCHICAL: a state's distribution is shrunk toward its
    good's marginal distribution with weight `w(state) = shrink * topup`,
    where `topup` caps how many pseudo-counts the prior can contribute
    (states with few observations lean on the good's aggregate; states with
    thousands barely move). The corpus-primed table makes the marginals
    real; the shrinkage is what keeps the 100-count states from reading a
    day-specific fluke as a law.
    """

    BINS = (1.0, 3.0, 6.0)      # sell-size bin edges
    SHRINK_TOPUP: float = 50.0  # pseudo-counts the good's prior can add

    def __init__(self, alpha: float = 0.5, n_bins: int = 4,
                 pretrained: bool = True) -> None:
        self.alpha = alpha
        self.n_bins = n_bins
        self.counts: dict[tuple, np.ndarray] = {}
        self.qty_sum: dict[tuple, np.ndarray] = {}
        self._marginals: dict[str, np.ndarray] = {}
        self._marginal_dirty = True
        if pretrained:
            # the corpus primes the table; online observe() keeps counting on
            # top of it (the same "+= 1", so nothing about the policy changes)
            trained = _load_trained()
            if trained:
                self.counts.update(trained)
                self.qty_sum.update(_load_trained_qty() or {})
            self._marginal_dirty = True

    def good_marginal(self, good: str) -> np.ndarray:
        """The good's aggregate action distribution (its own prior)."""
        if self._marginal_dirty:
            agg: dict[str, np.ndarray] = {}
            for key, arr in self.counts.items():
                a = agg.setdefault(key[0], np.zeros(self.n_bins))
                a += np.asarray(arr, dtype=float)
            self._marginals = agg
            self._marginal_dirty = False
        m = self._marginals.get(good)
        if m is None:
            return np.array([1.0] + [0.0] * (self.n_bins - 1))
        return m / m.sum()

    @staticmethod
    def _bucket(price: int, base: int) -> int:
        ratio = (price / base) if base else 1.0
        return int(np.clip(round((ratio - 1.0) * 3), -3, 3))

    def _key(self, good: str, step: int, price: int) -> tuple[str, int, int]:
        return (good, int(step // 24), self._bucket(price, MARKET_PARAMS[good]["base"]))

    def _key_activity(self, good: str, step: int, price: int,
                      activity: int) -> tuple[str, int, int, int]:
        """The activity key (#65 follow-up): the 3-tuple with the rival's
        own activity bucket APPENDED — (good, day, price_bucket, activity).
        The activity regime is the rival's own recent behaviour, so it
        rides on top of every per-good key shape without changing them.
        """
        base = self._key(good, step, price)
        return base + (int(activity),)

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
        self._marginal_dirty = True

    def policy(self, good: str, step: int, price: int,
               activity: int | None = None) -> np.ndarray:
        """Action distribution for a state, hierarchically smoothed.

        `(counts + alpha + w * marginal) / (n + k*alpha + w)` with
        `w = SHRINK_TOPUP * n / (n + SHRINK_TOPUP)`: a 10-observation state
        gets ~8 pseudo-counts from its good's aggregate, a 5000-observation
        state gets 49 — the empirical state dominates where it has data,
        the good's own prior carries it where it does not. Unseen states
        read the good's marginal directly. With `activity`, the
        activity-keyed table is used (see `expected_sell`).
        """
        key = (self._key_activity(good, step, price, activity)
               if activity is not None
               else self._key(good, step, price))
        arr = self.counts.get(key)
        marg = self.good_marginal(good)
        if arr is None:
            return marg
        n = float(np.sum(arr))
        w = self.SHRINK_TOPUP * n / (n + self.SHRINK_TOPUP)
        p = (np.asarray(arr, dtype=float) + self.alpha
             + w * marg)
        return p / p.sum()

    def _activity_marginal(self, good: str, day: int, bucket: int
                           ) -> tuple[np.ndarray, np.ndarray]:
        """counts/qty_sum summed over the activity axis for one plain state.

        The old (pre-activity) artifact *was* this marginal, so a caller
        without a bucket reads exactly what main's table answered.
        """
        acc = np.zeros(self.n_bins)
        acc_q = np.zeros(self.n_bins)
        base = (good, day, bucket)
        for key, arr in self.counts.items():
            if key[:3] == base:
                acc += np.asarray(arr, dtype=float)
                acc_q += np.asarray(self.qty_sum.get(key,
                                                     np.zeros(self.n_bins)),
                                    dtype=float)
        return acc, acc_q

    def expected_sell(self, good: str, step: int, price: int,
                      activity: int | None = None) -> float:
        """Expected units the rival sells next turn in this state.

        `activity` (the rival's own sell bucket over the last 24 turns,
        from `tracker.MarketTracker.activity_bucket`) selects the
        activity-keyed table; a silent rival's row answers near-pure
        hold, which is what makes a PASS rival predictable (measured:
        1,230 phantom units over 10 days without it, 0 with it). Without
        a bucket the plain state is the ACTIVITY MARGINAL — all bucket
        rows summed — the same answer the pre-activity artifact gave.
        """
        if activity is not None:
            key = self._key_activity(good, step, price, activity)
            p = self.policy(good, step, price, activity=activity)
            qs = self.qty_sum.get(key, np.zeros(self.n_bins))
            counts = self.counts.get(key, np.zeros(self.n_bins))
            mean_qty = np.where(counts > 0, qs / np.maximum(counts, 1.0), 0.0)
            return float(p @ mean_qty)
        acc, acc_q = self._activity_marginal(
            good, int(step // TURNS_PER_DAY),
            self._bucket(price, MARKET_PARAMS[good]["base"]))
        key = self._key(good, step, price)
        marg = self.good_marginal(good)
        if float(acc.sum()) <= 0.0 and key not in self.counts:
            return float(marg[0] * 0.0)      # an unseen plain state: hold
        n = float(acc.sum())
        w = self.SHRINK_TOPUP * n / (n + self.SHRINK_TOPUP)
        p = (acc + self.alpha + w * marg)
        p = p / p.sum()
        mean_qty = np.where(acc > 0, acc_q / np.maximum(acc, 1.0), 0.0)
        return float(p @ mean_qty)

    def expected_sell_day(self, obs: Any,
                          activity: int | None = None) -> dict[str, float]:
        """The rival's expected sell volume PER DAY for every good, from the
        observation alone — the `residual` forecast() consumes.

        Per good: the model's expected per-turn volume at the observed
        price, weighted by the state's sell probability, summed over the
        day's 24 turns (the same turn shape the forecast's walk applies it
        in). This is step 1 of #65: the wire between the trained model and
        the price path.

        `activity` selects the activity-keyed table (the rival's own
        24-turn sell bucket, from `activity_bucket(tracker)`); without it
        the plain states answer, which the activity artifact does not
        carry — callers wire the tracker's bucket or get zeros.
        """
        market = field_of(obs, "market", {}) or {}
        raw_inv = dict(field_of(market, "inventory", {}) or {})
        out: dict[str, float] = {}
        day_start = int(field_of(obs, "step", 0))
        for g in GOODS:
            inv = float(raw_inv.get(g, 0))
            price = int(field_of(market, "prices", {}) and
                        dict(field_of(market, "prices", {})).get(g, 0)
                        or price_of(g, inv))
            per_turn = self.expected_sell(g, day_start, price,
                                          activity=activity)
            out[g] = float(per_turn * TURNS_PER_DAY)
        return out


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
