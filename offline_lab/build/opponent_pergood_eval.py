"""#65 step 2c — the WINNING scheme: per-good key shape.

Two experiments, one conclusion:

* global 4-key (demand ADDED to day):  1.1824 aggregate, 3 goods regressed
* per-good combined (demand ADDED for 6 goods): 1.2835, WORSE than shipped

The lesson: the demand bucket's win came from it REPLACING the day
dimension for the demand-sensitive goods — those goods' behaviour depends
on WHICH SHOPS ARE OPEN, not on WHICH DAY IT IS, and merging both into one
key just split the data again. The mixed scheme keys

    demand-sensitive goods (CARROT, TOMATO, STRAWBERRY, MILK, EGG, WOOL):
        (good, demand_bucket, price_bucket)      — day DROPPED
    the rest (WHEAT, MELON, FERTILIZER):
        (good, day, price_bucket)                — the shipped key

and scores 1.2104 aggregate over the shipped 1.2714, beating it on 6/9
goods (the demand-sensitive ones lose 0.07-0.23 log loss each) while the
three plain-key goods stay at their shipped scores up to the tiny
shrinkage interaction. State count: 337 — SPARSER and better.

This module builds the mixed model on the full train dates, scores it on
the holdout, and (when --write is passed) serializes it as the replacement
artifact with its own info file.
"""
from __future__ import annotations

import argparse
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
_DEMAND_BINS = (0.0, 1.5, 6.5)
DEMAND_GOODS = frozenset({
    "CARROT", "TOMATO", "STRAWBERRY", "MILK", "EGG", "WOOL"})


def demand_from_row(shops: str, good: str) -> int:
    """The demand bucket (0..3) a town_shops string puts on a good."""
    names = [s.strip() for s in str(shops).split(",") if s.strip()]
    per_event = 0.0
    for name in names:
        basket = SHOP_BASKET.get(name)
        if basket and good in basket[0]:
            per_event += basket[1]
    units_per_day = per_event * 6.0            # a tick every 4 turns
    b = 0
    for edge in _DEMAND_BINS:
        if units_per_day > edge:
            b += 1
    return b


class MixedModel(OpponentModel):
    """Per-good key shape: demand-replaces-day for the sensitive goods,
    the shipped (good, day, price) for the rest."""

    def mixed_key(self, good: str, step: int, price: int, shops: str
                  ) -> tuple:
        base = OpponentModel._key(self, good, step, price)
        if good in DEMAND_GOODS:
            return (good, demand_from_row(shops, good), base[2])
        return base

    def observe_mixed(self, shops: str, step: int, op: str, item: str,
                      qty: float, price: int) -> None:
        key = self.mixed_key(item, step, price, shops)
        arr = self.counts.setdefault(key, np.zeros(self.n_bins))
        qs = self.qty_sum.setdefault(key, np.zeros(self.n_bins))
        if op == "SELL":
            b = self._bin(float(qty))
            arr[b] += 1.0
            qs[b] += float(qty)
        else:
            arr[-1] += 1.0
            qs[-1] += float(qty)

    def policy_mixed(self, good: str, step: int, price: int, shops: str
                     ) -> np.ndarray:
        key = self.mixed_key(good, step, price, shops)
        arr = self.counts.get(key)
        marg = self.good_marginal(good)
        if arr is None:
            return marg
        n = float(np.sum(arr))
        w = self.SHRINK_TOPUP * n / (n + self.SHRINK_TOPUP)
        p = np.asarray(arr, dtype=float) + self.alpha + w * marg
        return p / p.sum()


def build(train_dates: list[str], con) -> MixedModel:
    price_cols = ", ".join(f"price_{g}" for g in PRODUCTS)
    items_sql = ", ".join(repr(g) for g in PRODUCTS)
    mm = MixedModel(pretrained=False)
    for date in train_dates:
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
            mm.observe_mixed(str(shops), s, op, item, float(qty),
                             int(prices[PRODUCTS.index(item)]))
    return mm


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true",
                    help="serialize the mixed model as the shipped artifact")
    args = ap.parse_args()

    t0 = time.time()
    dates = _load_dates()
    holdout = dates[-HOLDOUT_DATES:]
    train = dates[:-HOLDOUT_DATES]
    print(f"train: {len(train)} dates, holdout: {holdout}")

    import duckdb
    con = duckdb.connect(config={"threads": 2, "memory_limit": "1GB"})
    price_cols = ", ".join(f"price_{g}" for g in PRODUCTS)
    items_sql = ", ".join(repr(g) for g in PRODUCTS)

    mm = build(train, con)
    print(f"mixed model: {len(mm.counts):,} states ({time.time()-t0:.0f}s)")

    ll_b = ll_m = 0.0
    n = 0
    per_good: dict[str, list] = defaultdict(lambda: [0.0, 0.0, 0])
    base = OpponentModel(pretrained=True)
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
            price = int(prices[PRODUCTS.index(item)])
            act = mm._bin(float(qty)) if op == "SELL" else mm.n_bins - 1
            pb = base.policy(item, s, price)
            pm = mm.policy_mixed(item, s, price, str(shops))
            ll_b += -np.log(max(pb[act], EPS))
            ll_m += -np.log(max(pm[act], EPS))
            acc = per_good[item]
            acc[0] += -np.log(max(pb[act], EPS))
            acc[1] += -np.log(max(pm[act], EPS))
            acc[2] += 1
            n += 1

    print(f"\nholdout rows: {n:,}")
    print(f"{'model':<34} {'log loss':>9}")
    print(f"{'shipped 3-key':<34} {ll_b/n:9.4f}")
    print(f"{'mixed per-good keys (this run)':<34} {ll_m/n:9.4f}")
    print("\nper good (shipped vs mixed):")
    for g in PRODUCTS:
        lb, lm, cnt = per_good[g]
        if cnt:
            print(f"  {g:<11} {lb/cnt:.4f} vs {lm/cnt:.4f} "
                  f"{'BETTER' if lm < lb - 1e-9 else 'worse'}")
    verdict = "BETTER" if ll_m < ll_b else "WORSE"
    print(f"\nverdict: {verdict} ({ll_b/n:.4f} -> {ll_m/n:.4f}); "
          f"states {len(base.counts)} -> {len(mm.counts)}")

    if args.write:
        from agent.artifact import artifact_path, write_info
        from agent.tile_dp.contract import engine_fingerprint
        keys = sorted(mm.counts)
        goods = np.array([k[0] for k in keys])
        d1 = np.array([k[1] for k in keys], dtype=np.int32)
        d2 = np.array([k[2] for k in keys], dtype=np.int8)
        counts = np.stack([np.asarray(mm.counts[k]) for k in keys]).astype(np.float32)
        qty_sum = np.stack([np.asarray(mm.qty_sum[k]) for k in keys]).astype(np.float32)
        npz_path = artifact_path("opponent_counts", ".npz")
        np.savez_compressed(npz_path, goods=goods, days=d1, buckets=d2,
                            counts=counts, qty_sum=qty_sum)
        write_info("opponent_counts", kind="opponent_counts",
                   file=npz_path.name,
                   contract="mixed per-good keys: (good, demand_bucket, "
                            "price_bucket) for CARROT/TOMATO/STRAWBERRY/"
                            "MILK/EGG/WOOL — (good, day, price_bucket) for "
                            "WHEAT/MELON/FERTILIZER; days col = demand "
                            "bucket 0..3 or season day",
                   engine=engine_fingerprint(), registry=None,
                   stats={"states": len(mm.counts),
                          "observations": int(sum(float(np.sum(a)) for a in
                                                  mm.counts.values())),
                          "train_dates": len(train),
                          "holdout_dates": HOLDOUT_DATES,
                          "holdout": holdout, "bins": mm.n_bins},
                   source="offline_lab.build.opponent_pergood_eval:build")
        print(f"wrote {npz_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
