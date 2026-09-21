"""Build the opponent action-count artifact — offline, from the replay store.

Run from the repo root:

    .venv/bin/python -m offline_lab.build.opponent_model [--progress]

Reads the parquet store (`/chista/Chista/kaggriculture-episodes-analyses/data/
replays_parquet/<date>/`), aggregates every SELL/BUY_PRODUCT of every player
into `agent/belief`'s own state key — (good, day, price bucket), action =
sell-size bin, plus a net-buy bin for the two dual goods — and writes

    agent/artifact/opponent_counts.npz   the counts
    agent/artifact/opponent_counts.json  the info (contract, stats, source)

The state key and the bins are `OpponentModel`'s own (`agent/belief/opponent.py`);
this builder never redefines them — it imports them. The agent loads the
artifact at runtime and keeps counting online from there.

Split: the store's dump DATES are the train/holdout split (last 7 dates are
held out and named in the info, so the evaluation builder can find them).

Performance (measured 2026-09-19, 8-core box): the per-date SQL itself is
fast (~1.5 s); the wall was the state lookup done as 92 OR-batched queries
(155 s/date). Per date the full city_steps table is ~600 k rows and fits
easily (~4 s to dict), so the shape is: per date — one state dict, one
GROUP BY over the orders, count, drop. 40 dates ≈ 4 min. Keeping ALL
dates' state rows is what OOM-killed the first run (3.9 GB RSS).
"""
from __future__ import annotations

import argparse
import glob
import os
import time
from pathlib import Path

import numpy as np

from agent.artifact import artifact_path, write_info
from agent.belief.opponent import OpponentModel
from agent.world.model import PRODUCTS

STORE = Path("/chista/Chista/kaggriculture-episodes-analyses/data/replays_parquet")
HOLDOUT_DATES = 7
_DAY = 24
_ITEMS_SQL = ", ".join(repr(g) for g in PRODUCTS)


def _load_dates() -> list[str]:
    dates = sorted(os.path.basename(p) for p in glob.glob(str(STORE / "*"))
                   if (Path(p) / "market_orders.parquet").exists())
    if not dates:
        raise FileNotFoundError(f"no replay dates under {STORE}")
    return dates


def build(train_dates: list[str], progress: bool = False) -> OpponentModel:
    """Build the activity-keyed model: the state key is
    (good, activity_bucket, day/demand, price_bucket), where activity_bucket
    = the rival's own sell volume over the 24 turns before (start / silent /
    low / mid / high) — the regime a PASS rival lives in is 'silent', whose
    corpus answer is near-pure hold.
    """
    import duckdb

    con = duckdb.connect(config={"threads": 2, "memory_limit": "1GB"})
    price_cols = ", ".join(f"price_{g}" for g in PRODUCTS)
    inv_cols = ", ".join(f"inv_{g}" for g in PRODUCTS)
    model = OpponentModel(pretrained=False)
    n_obs = 0
    t0 = time.time()

    for date in train_dates:
        mo = str(STORE / date / "market_orders.parquet")
        city = str(STORE / date / "city_steps.parquet")

        # per-(episode, step, player) sell volume, for the activity window
        sold: dict[tuple[int, int, int], float] = {}
        for e, s, p, sold_qty in con.execute(
            f"select episode_id, step, player, "
            f"sum(case when op = 'SELL' then qty else 0 end)::DOUBLE "
            f"from read_parquet('{mo}') group by 1, 2, 3"
        ).fetchall():
            sold[(e, s, p)] = float(sold_qty)

        state: dict[tuple[int, int], tuple[tuple, tuple]] = {}
        for e, s, *rest in con.execute(
            f"select episode_id, step, {price_cols}, {inv_cols} "
            f"from read_parquet('{city}')"
        ).fetchall():
            state[(e, s)] = (tuple(rest[:len(PRODUCTS)]),
                             tuple(rest[len(PRODUCTS):]))

        agg = con.execute(
            f"select episode_id, step, player, op, item, sum(qty)::DOUBLE qty "
            f"from read_parquet('{mo}') "
            f"where op in ('SELL', 'BUY_PRODUCT') and item in ({_ITEMS_SQL}) "
            f"group by 1, 2, 3, 4, 5"
        ).fetchall()

        counts, qty_sum, n_bins = model.counts, model.qty_sum, model.n_bins
        for e, s, player, op, item, qty in agg:
            st = state.get((e, s))
            if st is None:
                continue
            prices = st[0]
            # the ACTING player's own 24-turn sell window (this episode,
            # this player, the 24 steps before this one) — the activity
            # regime the state lives in
            window = 0.0
            for back in range(1, 25):
                window += sold.get((e, s - back, player), 0.0)
            if s < 24:
                activity = 0                        # start: no history yet
            else:
                activity = 1                        # silent by default
                if window > 60:
                    activity = 4
                elif window > 10:
                    activity = 3
                elif window > 0:
                    activity = 2
            key = model._key_activity(item, s, prices[PRODUCTS.index(item)],
                                      activity)
            arr = counts.setdefault(key, [0.0] * n_bins)
            qs = qty_sum.setdefault(key, [0.0] * n_bins)
            if op == "SELL":
                b = model._bin(float(qty))
                arr[b] += 1.0
                qs[b] += float(qty)
            else:                       # BUY_PRODUCT: the dual goods only
                arr[-1] += 1.0
                qs[-1] += float(qty)
            n_obs += 1
        if progress:
            print(f"  {date}: {n_obs:,} aggregated ({time.time()-t0:.0f}s)",
                  flush=True)
    return model


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-days", type=int, default=None,
                    help="use only the newest N dates minus the holdout")
    ap.add_argument("--progress", action="store_true")
    args = ap.parse_args()

    dates = _load_dates()
    holdout = dates[-HOLDOUT_DATES:]
    train = dates[:-HOLDOUT_DATES]
    if args.train_days:
        train = train[-args.train_days:]
    print(f"train dates: {len(train)}  holdout: {holdout}")

    t0 = time.time()
    model = build(train, progress=args.progress)
    n_keys = len(model.counts)
    n_obs = sum(float(a[i]) for a in model.counts.values() for i in range(a.__len__()))
    print(f"built: {n_keys:,} states, {n_obs:,.0f} observations "
          f"({time.time()-t0:.0f}s)")

    # --- serialize: keys are (good, day, price_bucket); arrays are (n_bins,) --
    keys = sorted(model.counts)
    goods = np.array([k[0] for k in keys])
    days = np.array([k[1] for k in keys], dtype=np.int32)
    buckets = np.array([k[2] for k in keys], dtype=np.int8)
    counts = np.stack([np.asarray(model.counts[k]) for k in keys]).astype(np.float32)
    qty_sum = np.stack([np.asarray(model.qty_sum[k]) for k in keys]).astype(np.float32)
    npz_path = artifact_path("opponent_counts", ".npz")
    np.savez_compressed(npz_path, goods=goods, days=days, buckets=buckets,
                        counts=counts, qty_sum=qty_sum)

    from agent.tile_dp.contract import engine_fingerprint
    write_info("opponent_counts", kind="opponent_counts",
               file=npz_path.name,
               contract="N[good, day, price_bucket, action] sell-size counts "
                        "over both seats of the store's SELL/BUY_PRODUCT "
                        "orders; bins = OpponentModel.BINS + net-buy tail",
               engine=engine_fingerprint(),
               registry=None,
               stats={"states": n_keys, "observations": int(n_obs),
                      "train_dates": len(train), "holdout_dates": HOLDOUT_DATES,
                      "holdout": holdout, "bins": model.n_bins},
               source="offline_lab.build.opponent_model:build")
    print(f"wrote {npz_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
