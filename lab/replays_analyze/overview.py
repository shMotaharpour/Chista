"""Replay analysis: overview stats of the competition replays (Parquet store).

Phase 1 — what do we have and what do top players do?
  1. Store overview: episodes, date range, table row counts
  2. Score distribution; who the top players are
  3. Crop mix of winners vs losers
  4. Sell timing of top players (the V2 calibration data)

Usage: python -m lab.replays_analyze.overview
"""
from __future__ import annotations

import duckdb
import os
import sys

REPLAYS = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                       "competition_replays", "parquet")
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                   "docs", "replays Analysis")
GLOB = os.path.join(REPLAYS, "*", "episodes.parquet")


def con():
    c = duckdb.connect()
    for t in ("episodes", "steps", "farm_steps", "private_steps", "actions",
              "market_orders", "tiles_delta"):
        c.execute(f"CREATE VIEW {t} AS SELECT * FROM '{REPLAYS}/*/{t}.parquet'")
    return c


def main():
    os.makedirs(OUT, exist_ok=True)
    c = con()
    lines = ["# Replay Analysis — Overview", ""]

    # 1. store overview
    n_ep = c.execute("SELECT count(*), sum(n_steps) FROM episodes").fetchone()
    lines.append(f"## Store: {n_ep[0]} episodes, {n_ep[1]:,} total steps")
    for t in ("steps", "farm_steps", "private_steps", "actions", "market_orders", "tiles_delta"):
        n = c.execute(f"SELECT count(*) FROM {t}").fetchone()[0]
        lines.append(f"- `{t}`: {n:,} rows")

    # 2. score distribution
    lines.append("\n## Score distribution (all episodes, both players)")
    rows = c.execute("""
        SELECT unnest([reward0, reward1]) r FROM episodes
    """).df()
    r = rows["r"]
    lines.append(f"- games: {len(r):,} player-scores; min={r.min():.0f} p25={r.quantile(.25):.0f} "
                 f"median={r.median():.0f} p75={r.quantile(.75):.0f} p90={r.quantile(.9):.0f} "
                 f"p99={r.quantile(.99):.0f} max={r.max():.0f}")

    # top agents by avg reward
    lines.append("\n### Top agents (min 3 games)")
    top = c.execute("""
        SELECT agent, count(*) games, round(avg(m)) avg_money, max(m) best
        FROM (SELECT unnest([agent0, agent1]) agent,
                     unnest([reward0, reward1]) m
              FROM episodes)
        GROUP BY 1 HAVING count(*) >= 3 ORDER BY avg_money DESC LIMIT 15
    """).df()
    lines.append(top.to_markdown(index=False))

    # 3. crop mix winners vs losers
    lines.append("\n## Crop mix: winners vs losers (PLANT actions)")
    mix = c.execute("""
        SELECT a.arg1 crop,
               CASE WHEN (a.player=0 AND e.reward0>e.reward1) OR (a.player=1 AND e.reward1>e.reward0)
                    THEN 'winner' ELSE 'loser' END side,
               count(*) plants
        FROM actions a JOIN episodes e USING (episode_id)
        WHERE a.op='PLANT'
        GROUP BY 1, 2 ORDER BY 1, 2
    """).df()
    lines.append(mix.to_markdown(index=False))

    # 4. sell timing of top players
    lines.append("\n## Sell timing by product (winners, day distribution p25/p50/p75)")
    sells = c.execute("""
        SELECT m.item, m.step / e.turnsPerDay AS day,
               count(*) n
        FROM market_orders m JOIN episodes e USING (episode_id)
        WHERE m.op='SELL'
          AND ((m.player=0 AND e.reward0>e.reward1) OR (m.player=1 AND e.reward1>e.reward0))
        GROUP BY 1, 2
    """).df()
    import pandas as pd
    tt = sells.groupby("item")["day"].quantile([.25, .5, .75]).unstack()
    tt.columns = ["p25", "p50", "p75"]
    tt["total_sells"] = sells.groupby("item")["n"].sum()
    lines.append(tt.sort_values("total_sells", ascending=False).to_markdown())

    text = "\n".join(lines)
    out_md = os.path.join(OUT, "overview.md")
    with open(out_md, "w") as f:
        f.write(text)
    print(text)


if __name__ == "__main__":
    main()
