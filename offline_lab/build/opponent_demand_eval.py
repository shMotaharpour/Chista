"""#65 step 2: the shop-demand state dimension, built and scored.

Builds a SECOND trained table whose state key adds the good's shop-demand
level — the units/day the CURRENT open-shop set demands of that good, from
`SHOP_BASKET` in closed form, bucketed (0 / 1-2 / 3-6 / 7+). Trains on the
same 40 dates, scores BOTH tables on the same 7 holdout dates with the same
harness, and prints the per-good verdict. Nothing in `agent/belief` changes
until the numbers say the dimension earns its state space.
"""
from __future__ import annotations

import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, "/chista/Dev1/Chista")
from offline_lab.build.opponent_model import STORE, _load_dates
from agent.belief.opponent import OpponentModel
from agent.belief.schemas import SHOP_BASKET, field_of
from agent.world.model import PRODUCTS

EPS = 1e-12
HOLDOUT_DATES = 7
_DEMAND_BINS = (0.0, 1.5, 6.5)          # 0 / 1-2 / 3-6 / 7+ units per day


def demand_bucket(obs, good: str) -> int:
    """Which demand level the CURRENT shop set puts on `good`.

    From the open shops' own baskets in closed form (each instance consumes
    every `SHOP_INTERVAL` turns, single-product shops x2) scaled to a day.
    """
    shops = list(field_of(field_of(obs, "town", {}) or {},
                          "unlocked_shops", []) or [])
    per_event = 0.0
    for name in shops:
        items, mult = SHOP_BASKET[name]
        if good in items:
            per_event += mult
    units_per_day = per_event * (24 / 4)        # a tick every 4 turns
    b = 0
    for edge in _DEMAND_BINS:
        if units_per_day > edge:
            b += 1
    return b


def demand_from_row(shops: str, good: str) -> int:
    """The same bucket from a city_steps row's town_shops string."""
    names = [s.strip() for s in str(shops).split(",") if s.strip()]
    per_event = 0.0
    for name in names:
        basket = SHOP_BASKET.get(name)
        if basket and good in basket[0]:
            per_event += basket[1]
    units_per_day = per_event * 6.0
    b = 0
    for edge in _DEMAND_BINS:
        if units_per_day > edge:
            b += 1
    return b


class DemandModel(OpponentModel):
    """OpponentModel with the demand bucket as a 4th key dimension."""

    def _key(self, good: str, step: int, price: int) -> tuple:
        raise NotImplementedError("use demand_key with the observation")

    def demand_key(self, good: str, step: int, price: int, obs) -> tuple:
        base_key = OpponentModel._key(self, good, step, price)
        return base_key + (demand_bucket(obs, good),)

    def observe_demand(self, obs, step: int, op: str, item: str, qty: float
                       ) -> None:
        price = int(dict(field_of(field_of(obs, "market", {}) or {},
                                  "prices", {}) or {}
                         ).get(item, 0))
        key = self.demand_key(item, step, price, obs)
        arr = self.counts.setdefault(key, np.zeros(self.n_bins))
        qs = self.qty_sum.setdefault(key, np.zeros(self.n_bins))
        if op == "SELL":
            b = self._bin(float(qty))
            arr[b] += 1.0
            qs[b] += float(qty)
        else:
            arr[-1] += 1.0
            qs[-1] += float(qty)

    def policy_demand(self, good: str, step: int, price: int, obs
                      ) -> np.ndarray:
        key = self.demand_key(good, step, price, obs)
        arr = self.counts.get(key)
        marg = self.good_marginal(good)
        if arr is None:
            return marg
        n = float(np.sum(arr))
        w = self.SHRINK_TOPUP * n / (n + self.SHRINK_TOPUP)
        p = np.asarray(arr, dtype=float) + self.alpha + w * marg
        return p / p.sum()


def main() -> int:
    t0 = time.time()
    dates = _load_dates()
    holdout = dates[-HOLDOUT_DATES:]
    train = dates[:-HOLDOUT_DATES]
    print(f"train: {len(train)} dates, holdout: {holdout}")

    # ---- train the demand model on the SAME train dates ------------------- #
    import duckdb
    con = duckdb.connect(config={"threads": 2, "memory_limit": "1GB"})
    price_cols = ", ".join(f"price_{g}" for g in PRODUCTS)
    items_sql = ", ".join(repr(g) for g in PRODUCTS)
    dm = DemandModel(pretrained=False)
    n_obs = 0
    for date in train:
        mo = str(STORE / date / "market_orders.parquet")
        city = str(STORE / date / "city_steps.parquet")
        state: dict[tuple[int, int], tuple] = {}
        for e, s, *rest in con.execute(
            f"select episode_id, step, town_shops, {price_cols} "
            f"from read_parquet('{city}')"
        ).fetchall():
            state[(e, s)] = (rest[0], rest[1:])
        agg = con.execute(
            f"select episode_id, step, op, item, sum(qty)::DOUBLE qty "
            f"from read_parquet('{mo}') "
            f"where op in ('SELL', 'BUY_PRODUCT') and item in ({items_sql}) "
            f"group by 1, 2, 3, 4"
        ).fetchall()
        for e, s, op, item, qty in agg:
            st = state.get((e, s))
            if st is None:
                continue
            shops, prices = st
            obs_stub = {"town": {"unlocked_shops": [
                t.strip() for t in str(shops).split(",") if t.strip()]}}
            dm.observe_demand(obs_stub, s, op, item, float(qty))
            n_obs += 1
        if time.time() - t0 > 20 and (n_obs // 5_000_000):
            pass
    n_states = len(dm.counts)
    print(f"demand model: {n_states:,} states, {n_obs:,} obs "
          f"({time.time()-t0:.0f}s)")

    # ---- score BOTH models on the same holdout rows ----------------------- #
    base = OpponentModel(pretrained=True)      # the shipped 3-key model
    base_keys = sorted(base.counts)
    base_ix = {k: i for i, k in enumerate(base_keys)}
    dm_keys = sorted(dm.counts)
    dm_ix = {k: i for i, k in enumerate(dm_keys)}

    # policy matrices, vectorized over the rows as they stream
    ll_base = ll_dm = 0.0
    n = 0
    per_good: dict[str, list] = defaultdict(lambda: [0.0, 0.0, 0])
    for date in holdout:
        mo = str(STORE / date / "market_orders.parquet")
        city = str(STORE / date / "city_steps.parquet")
        state: dict[tuple[int, int], tuple] = {}
        for e, s, *rest in con.execute(
            f"select episode_id, step, town_shops, {price_cols} "
            f"from read_parquet('{city}')"
        ).fetchall():
            state[(e, s)] = (rest[0], rest[1:])
        agg = con.execute(
            f"select episode_id, step, op, item, sum(qty)::DOUBLE qty "
            f"from read_parquet('{mo}') "
            f"where op in ('SELL', 'BUY_PRODUCT') and item in ({items_sql}) "
            f"group by 1, 2, 3, 4"
        ).fetchall()
        for e, s, op, item, qty in agg:
            st = state.get((e, s))
            if st is None:
                continue
            shops, prices = st
            gi = PRODUCTS.index(item)
            price = int(prices[gi])
            act = dm._bin(float(qty)) if op == "SELL" else dm.n_bins - 1
            obs_stub = {"town": {"unlocked_shops": [
                t.strip() for t in str(shops).split(",") if t.strip()]}}
            # baseline: the shipped model
            pb = base.policy(item, s, price)
            # the demand model
            pd = dm.policy_demand(item, s, price, obs_stub)
            lb = -np.log(max(pb[act], EPS))
            ld = -np.log(max(pd[act], EPS))
            ll_base += lb
            ll_dm += ld
            acc = per_good[item]
            acc[0] += lb
            acc[1] += ld
            acc[2] += 1
            n += 1

    print(f"\nholdout rows: {n:,} ({time.time()-t0:.0f}s)")
    print(f"{'model':<22} {'log loss':>9}")
    print(f"{'3-key (shipped)':<22} {ll_base/n:9.4f}")
    print(f"{'4-key (+shop demand)':<22} {ll_dm/n:9.4f}")
    print("\nper good (log loss, shipped vs demand):")
    worse = better = 0
    for g in PRODUCTS:
        lb, ld, cnt = per_good[g]
        if not cnt:
            continue
        tag = "BETTER" if ld < lb else "worse"
        better += ld < lb
        worse += ld >= lb
        print(f"  {g:<11} n={cnt:>9,}  {lb/cnt:.4f} vs {ld/cnt:.4f}  {tag}")
    print(f"\nverdict: demand dimension better on {better}/"
          f"{better+worse} goods; aggregate {'BETTER' if ll_dm < ll_base else 'WORSE'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
