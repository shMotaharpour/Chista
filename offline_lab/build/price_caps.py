"""The sell-price caps: what a good's price really did, given the demand its shops made.

`belief` forecasts a price path from the town's drain alone (`.price_paths`), and a
drain-only path has no shop demand in it: it keeps rising past the day the season's
shops have decided, so the plan is priced against a price no engine paid. This builder
answers the same question from the store of real episodes instead of from the drain --
the mean price of a good on a day, conditioned on how many of that good's consumer
shops the town held -- so a reader can bound a forecast by what actually happened.

The condition is the repository's own demand bucket, not a new axis: `demand_from_row`
(`offline_lab.build.opponent_pergood_eval`) already answers "how much demand does the
town hold for this good", and `opponent_counts` is keyed by it. Reusing it keeps one
definition of demand rather than two. The consumers come from `agent.world.rules.SHOPS`
(engine-derived), where a single-product shop consumes 2 units -- so a good's demand is
the SHOP rule's depth, never a taste.

The mean of means is exact here: each row of the scan is one (day, shop set) group with
its own count, and a count-weighted mean over disjoint groups IS the group's mean.
`KAGGLE_REPLAYS_PARQUET` names the store; rebuild only when the store grows.
"""

from __future__ import annotations

import os
from pathlib import Path

import duckdb
import numpy as np

from agent.artifact import artifact_path, write_info
from agent.world.rules import PRODUCTS
from offline_lab.build.opponent_pergood_eval import demand_from_row

#: The artifact's name, equal to the file stems it writes.
NAME = "sell_price_caps"

#: The store of real episodes, the builder's only input.
STORE = Path(os.environ.get(
    "KAGGLE_REPLAYS_PARQUET",
    "/home/amirelite_ai/Chista/kaggriculture-episodes-analyses/data/replays_parquet"))

#: The season's days: the store's own horizon, one row per day per episode.
DAYS = 30

#: The demand axis is sized from the rule, not from the store: the bucket is a count of
#: consumer shops, and the town holds at most `MAX_SHOP_INSTANCES` of them.
BUCKETS = 32


def scan(con: duckdb.DuckDBPyConnection) -> list[tuple]:
    """One pass over the store: every (day, shop set) group with its own mean prices.

    The shop set is a short categorical, so grouping by it collapses the season's steps
    into a few thousand rows and keeps the pass to a single scan of the store.
    """
    prices = ", ".join(f"avg(price_{g}) AS p_{g}" for g in PRODUCTS)
    return con.sql(f"""
        SELECT day, town_shops, count(*) AS n, {prices}
        FROM read_parquet('{STORE}/*/city_steps.parquet')
        GROUP BY day, town_shops
    """).fetchall()


def build() -> Path:
    """Write the caps artifact: the mean price per (good, day, demand bucket)."""
    con = duckdb.connect()
    con.sql("SET threads=8")
    rows = scan(con)
    n_goods = len(PRODUCTS)
    weight = np.zeros((n_goods, DAYS, BUCKETS), dtype=np.float64)
    total = np.zeros((n_goods, DAYS, BUCKETS), dtype=np.float64)
    for record in rows:
        day, shops, n = int(record[0]), record[1], int(record[2])
        means = record[3:]
        if not 0 <= day < DAYS:
            continue
        for gi, good in enumerate(PRODUCTS):
            bucket = int(demand_from_row(shops or "", good))
            if not 0 <= bucket < BUCKETS:
                continue
            mean = float(means[gi])
            if np.isnan(mean):
                continue
            weight[gi, day, bucket] += n
            total[gi, day, bucket] += n * mean
    with np.errstate(invalid="ignore", divide="ignore"):
        caps = np.where(weight > 0.0, total / np.maximum(weight, 1.0), np.nan)

    npz_path = artifact_path(NAME, ".npz")
    np.savez_compressed(npz_path,
                        goods=np.array(PRODUCTS, dtype=object),
                        caps=caps.astype(np.float32),
                        weight=weight.astype(np.float64))
    write_info(NAME, kind="sell_price_caps", file=npz_path.name,
               contract="mean price the engine paid, per (good, day, demand bucket) of "
                        "the town's own shop set; the bucket is "
                        "offline_lab.build.opponent_pergood_eval:demand_from_row, the "
                        "same key opponent_counts uses; weight = the episodes behind "
                        "each cell, so a reader can refuse a cell it does not trust",
               engine=_engine(), registry=None,
               stats={"goods": list(PRODUCTS), "days": DAYS, "buckets": BUCKETS,
                      "groups": len(rows),
                      "cells": int(np.sum(weight > 0.0)),
                      "steps_behind_day0": int(weight[0, 0, :].sum())},
               source="offline_lab.build.price_caps:build")
    return npz_path


def _engine() -> str:
    """The engine fingerprint, from the one place that defines it."""
    from agent.tile_dp.contract import engine_fingerprint
    return engine_fingerprint()


if __name__ == "__main__":
    print("wrote", build())
