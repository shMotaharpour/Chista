"""Evaluate the trained opponent model on the holdout dates.

Run from the repo root:

    .venv/bin/python -m offline_lab.build.opponent_eval

The claim under test: the corpus-primed `OpponentModel` predicts the rival's
next-turn action better than the two baselines it must beat —

  1. always-hold  (predict bin 3 everywhere; the corpus's global mode),
  2. the global frequency model (one distribution for every state).

Score: log loss and Brier score over the holdout dates' (state, action)
pairs, per good and overall — computed exactly, no sampling. The held-out
DATES (the last 7 of the store, named in the artifact info) were never seen
in training.
"""
from __future__ import annotations

import glob
import json
import os
import time
from pathlib import Path

import numpy as np

from agent.belief.opponent import OpponentModel
from agent.world.model import PRODUCTS

STORE = Path("/chista/Chista/kaggriculture-episodes-analyses/data/replays_parquet")
_DAY = 24
EPS = 1e-12


def _holdout_dates() -> list[str]:
    info = json.loads((Path("agent/artifact/opponent_counts.json")).read_text())
    return list(info["stats"]["holdout"])


def _eval_rows(dates: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray,
                                          np.ndarray]:
    """(state_idx, action_idx, good_idx, activity_idx) from the holdout.

    The state key/bins come from OpponentModel itself; the rows are the
    ACTUAL actions the seats committed on the held-out days, and each row
    carries the acting seat's OWN activity bucket (its 24-turn sell
    window, the same computation the builder applies) so the
    activity-keyed table is scored on the states it actually models.
    """
    import duckdb

    con = duckdb.connect(config={"threads": 2, "memory_limit": "1GB"})
    model = OpponentModel(pretrained=True)
    price_cols = ", ".join(f"price_{g}" for g in PRODUCTS)
    items_sql = ", ".join(repr(g) for g in PRODUCTS)

    keys = sorted(model.counts)
    key_ix = {k: i for i, k in enumerate(keys)}

    s_idx, a_idx, g_idx, act_idx = [], [], [], []
    for date in dates:
        mo = str(STORE / date / "market_orders.parquet")
        city = str(STORE / date / "city_steps.parquet")

        sold: dict[tuple[int, int, int], float] = {}
        for e, s, p, sold_qty in con.execute(
            f"select episode_id, step, player, "
            f"sum(case when op = 'SELL' then qty else 0 end)::DOUBLE "
            f"from read_parquet('{mo}') group by 1, 2, 3"
        ).fetchall():
            sold[(e, s, p)] = float(sold_qty)

        state: dict[tuple[int, int], tuple] = {}
        for e, s, *rest in con.execute(
            f"select episode_id, step, {price_cols} "
            f"from read_parquet('{city}')"
        ).fetchall():
            state[(e, s)] = tuple(rest)
        agg = con.execute(
            f"select episode_id, step, player, op, item, sum(qty)::DOUBLE qty "
            f"from read_parquet('{mo}') "
            f"where op in ('SELL', 'BUY_PRODUCT') and item in ({items_sql}) "
            f"group by 1, 2, 3, 4, 5"
        ).fetchall()
        for e, s, player, op, item, qty in agg:
            prices = state.get((e, s))
            if prices is None:
                continue
            window = sum(sold.get((e, s - back, player), 0.0)
                         for back in range(1, 25))
            if s < 24:
                activity = 0
            else:
                activity = 1
                if window > 60:
                    activity = 4
                elif window > 10:
                    activity = 3
                elif window > 0:
                    activity = 2
            key = model._key_activity(item, s, prices[PRODUCTS.index(item)],
                                      activity)
            ix = key_ix.get(key)
            if ix is None:
                continue                      # a state training never saw
            b = (model._bin(float(qty)) if op == "SELL"
                 else model.BUY_BIN)
            s_idx.append(ix)
            a_idx.append(b)
            g_idx.append(PRODUCTS.index(item))
            act_idx.append(activity)
    return (np.array(s_idx), np.array(a_idx), np.array(g_idx),
            np.array(act_idx))


def _policy_matrix(model: OpponentModel, keys) -> np.ndarray:
    """The model's OWN policy per state (hierarchical smoothing applied).

    The key's price bucket is mapped back to a representative price
    (`base * (1 + bucket/3)`), the same mapping `_bucket` quantizes. The
    activity dimension is queried with `activity=`, bypassing the
    smoothing's marginal fallback for unseen activity states.
    """
    from agent.world.prices import MARKET_PARAMS
    out = np.zeros((len(keys), model.n_bins))
    for i, key in enumerate(keys):
        good, day, bucket = key[0], key[1], key[2]
        activity = key[3] if len(key) > 3 else None
        price = int(round((1.0 + bucket / 3.0) * MARKET_PARAMS[good]["base"]))
        out[i] = model.policy(good, day * 24, price, activity=activity)
    return out


def main() -> int:
    t0 = time.time()
    holdout = _holdout_dates()
    print(f"holdout dates: {holdout}")

    trained = OpponentModel(pretrained=True)
    keys = sorted(trained.counts)
    P_trained = _policy_matrix(trained, keys)

    # baseline 2: the GLOBAL frequency distribution (ignores the state)
    global_counts = np.zeros(trained.n_bins)
    for arr in trained.counts.values():
        global_counts += np.asarray(arr)
    P_global = np.tile((global_counts + trained.alpha) /
                       (global_counts.sum() + trained.alpha * trained.n_bins),
                       (len(keys), 1))

    # baseline 1: always-hold = bin 3 with certainty
    P_hold = np.zeros((len(keys), trained.n_bins))
    P_hold[:, trained.n_bins - 1] = 1.0

    s_idx, a_idx, g_idx, _act = _eval_rows(holdout)
    n = len(s_idx)
    print(f"holdout rows: {n:,} over {len(set(s_idx.tolist())):,} seen states "
          f"({time.time()-t0:.0f}s)")

    onehot = np.zeros((n, trained.n_bins))
    onehot[np.arange(n), a_idx] = 1.0

    print(f"\n{'model':<14} {'log loss':>9} {'brier':>7}")
    rows = {}
    for name, P in (("trained", P_trained), ("global-freq", P_global),
                    ("always-hold", P_hold)):
        q = np.clip(P[s_idx], EPS, 1.0)
        ll = float(-np.mean(np.log(q[np.arange(n), a_idx])))
        br = float(np.mean(np.sum((P[s_idx] - onehot) ** 2, axis=1)))
        rows[name] = (ll, br)
        print(f"{name:<14} {ll:9.4f} {br:7.4f}")

    print("\nper-good log loss (trained vs global vs hold):")
    for gi, g in enumerate(PRODUCTS):
        m = g_idx == gi
        if not m.any():
            continue
        line = f"  {g:<11} n={int(m.sum()):>8,} "
        for name, P in (("trained", P_trained), ("global-freq", P_global),
                        ("always-hold", P_hold)):
            q = np.clip(P[s_idx[m]], EPS, 1.0)
            line += f" {name}={-np.mean(np.log(q[np.arange(m.sum()), a_idx[m]])):.4f}"
        print(line)

    better = rows["trained"][0] < rows["global-freq"][0] and \
        rows["trained"][0] < rows["always-hold"][0]
    print(f"\nverdict: trained beats both baselines on log loss: {better}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
