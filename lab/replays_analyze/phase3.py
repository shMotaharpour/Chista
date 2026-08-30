"""Deep analysis phase 3: shop unlock distributions, crop correlations, land modes.

Outputs to docs/replays Analysis/003-shops-crops-corr.md

1. Probability distribution of WHICH shop opens at each of the 8 unlock slots (day 3,6,...,24)
2. Correlation: first unlocked shop type <-> winner's planted crops
3. Correlation: winner's crop mix vs opponent's crop mix (same episode)
4. MODE of each of the 25 NW-quadrant tiles at day 25 hour 0 (what occupies the land)
5. Average crops planted per 3-day period across the season
"""
from __future__ import annotations

import duckdb
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REPLAYS = os.path.join(ROOT, "competition_replays", "parquet")
DOCS = os.path.join(ROOT, "docs", "replays Analysis")


def con():
    c = duckdb.connect()
    for t in ("episodes", "steps", "farm_steps", "actions", "market_orders"):
        c.execute(f"CREATE VIEW {t} AS SELECT * FROM '{REPLAYS}/*/{t}.parquet'")
    return c


def md(c, sql):
    return c.execute(sql).df().to_markdown(index=False)


def main():
    c = con()
    out = ["# Shop Distributions, Crop Correlations, Land Modes", ""]

    # ------------------------------------------------ 1. shop unlock slot probabilities
    out.append("## 1. P(shop | unlock slot) — 8 slots at days 3,6,9,...,24\n")
    out.append(md(c, """
        WITH shops AS (
          SELECT episode_id, min(day) AS unlock_day, shop
          FROM (SELECT episode_id, day, unnest(string_split(town_shops, ',')) AS shop
                FROM steps WHERE town_shops != '') WHERE shop != ''
          GROUP BY 1, 3
        ),
        slots AS (
          SELECT episode_id, shop, unlock_day,
                 row_number() OVER (PARTITION BY episode_id ORDER BY unlock_day) AS slot
          FROM shops
        )
        SELECT slot, shop, count(*) AS n,
               round(100.0 * count(*) / sum(count(*)) OVER (PARTITION BY slot), 1) AS pct
        FROM slots GROUP BY 1, 2
        QUALIFY True ORDER BY slot, pct DESC
    """))

    # ------------------------------------------------ 2. first shop vs winner crops
    out.append("\n## 2. First-unlocked shop vs winner's crop mix (share of winner's plants)\n")
    out.append(md(c, """
        WITH first_shop AS (
          SELECT episode_id, shop
          FROM (SELECT episode_id, day, unnest(string_split(town_shops, ',')) AS shop
                FROM steps WHERE town_shops != '')
          WHERE shop != ''
          GROUP BY 1, 3 HAVING min(day) = 3
        ),
        win AS (
          SELECT e.episode_id, fs.player
          FROM episodes e, farm_steps fs
          WHERE fs.episode_id = e.episode_id
            AND ((fs.player = 0 AND e.reward0 > e.reward1) OR (fs.player = 1 AND e.reward1 > e.reward0))
          GROUP BY 1, 2
        ),
        wplants AS (
          SELECT a.episode_id, a.arg1 AS crop, count(*) AS n
          FROM actions a
          JOIN win ON win.episode_id = a.episode_id AND win.player = a.player
          WHERE a.op = 'PLANT'
          GROUP BY 1, 2
        ),
        tot AS (SELECT episode_id, sum(n) AS total FROM wplants GROUP BY 1)
        SELECT fs.shop AS first_shop, wp.crop, round(100.0 * wp.n / t.total, 1) AS pct_of_winner_plants
        FROM first_shop fs
        JOIN wplants wp ON wp.episode_id = fs.episode_id
        JOIN tot t ON t.episode_id = fs.episode_id
        QUALIFY row_number() OVER (PARTITION BY fs.shop ORDER BY wp.n DESC) <= 3
        ORDER BY fs.shop, pct DESC
    """))

    # ------------------------------------------------ 3. winner vs opponent crop corr
    out.append("\n## 3. Winner crop mix vs opponent crop mix (same episode)\n")
    # per episode: crop counts for winner and loser, then correlation across episodes
    out.append(md(c, """
        WITH win AS (
          SELECT e.episode_id,
                 CASE WHEN e.reward0 > e.reward1 THEN 0 ELSE 1 END AS wp,
                 CASE WHEN e.reward0 > e.reward1 THEN 1 ELSE 0 END AS lp
          FROM episodes e
        ),
        crops AS (
          SELECT a.episode_id,
                 CASE WHEN a.player = win.wp THEN 'winner' ELSE 'loser' END AS side,
                 a.arg1 AS crop, count(*) AS n
          FROM actions a JOIN win ON win.episode_id = a.episode_id
              AND a.player IN (win.wp, win.lp)
          WHERE a.op = 'PLANT'
          GROUP BY 1, 2, 3
        ),
        piv AS (
          SELECT episode_id, crop,
                 sum(CASE WHEN side='winner' THEN n ELSE 0 END) AS wn,
                 sum(CASE WHEN side='loser' THEN n ELSE 0 END) AS ln
          FROM crops GROUP BY 1, 2
        ),
        shares AS (
          SELECT crop,
                 sum(wn) AS win_total, sum(ln) AS lose_total
          FROM piv GROUP BY 1
        ),
        eps AS (
          SELECT episode_id, crop, wn, ln FROM piv
        ),
        agg AS (
          SELECT crop,
                 count(DISTINCT episode_id) AS eps,
                 avg(wn) AS avg_win_plants, avg(ln) AS avg_lose_plants
          FROM eps GROUP BY 1
        )
        SELECT crop, eps, round(avg_win_plants,1) AS avg_winner_plants,
               round(avg_lose_plants,1) AS avg_loser_plants,
               round(100.0*(avg_win_plants-avg_lose_plants)/NULLIF(avg_lose_plants,0),1) AS pct_diff
        FROM agg ORDER BY avg_win_plants DESC
    """))

    # per-episode correlation between winner & loser crop shares
    out.append("\nPer-episode crop-share correlation (winner vs loser, per crop):\n")
    out.append(md(c, """
        WITH win AS (
          SELECT e.episode_id,
                 CASE WHEN e.reward0 > e.reward1 THEN 0 ELSE 1 END AS wp,
                 CASE WHEN e.reward0 > e.reward1 THEN 1 ELSE 0 END AS lp
          FROM episodes e
        ),
        crops AS (
          SELECT a.episode_id, a.arg1 AS crop,
                 CASE WHEN a.player = win.wp THEN 'w' ELSE 'l' END AS side
          FROM actions a JOIN win ON win.episode_id = a.episode_id
              AND a.player IN (win.wp, win.lp)
          WHERE a.op = 'PLANT'
        ),
        counts AS (
          SELECT episode_id, crop,
                 count(CASE WHEN side='w' THEN 1 END) AS wn,
                 count(CASE WHEN side='l' THEN 1 END) AS ln
          FROM crops GROUP BY 1, 2
        ),
        shares AS (
          SELECT episode_id, crop, 1.0*wn/(sum(wn) OVER (PARTITION BY episode_id)) AS wshare,
                 1.0*ln/(sum(ln) OVER (PARTITION BY episode_id)) AS lshare
          FROM counts
        ),
        joined AS (
          SELECT w.crop, w.wshare, l.lshare
          FROM shares w JOIN shares l USING (episode_id, crop)
        )
        SELECT crop, round(corr(wshare, lshare), 3) AS corr_winner_loser_share,
               count(*) AS episodes
        FROM joined GROUP BY 1 ORDER BY 2 DESC
    """))

    # ------------------------------------------------ 4. tile modes at day 25
    out.append("\n## 4. Mode of each of the 25 NW tiles at day 25 hour 0 (winners' farm)\n")
    out.append(md(c, """
        WITH win AS (
          SELECT e.episode_id,
                 CASE WHEN e.reward0 > e.reward1 THEN 0 ELSE 1 END AS wp
          FROM episodes e
        ),
        day25 AS (
          SELECT episode_id, wp FROM win
        ),
        tiles AS (
          SELECT t.episode_id, t.x, t.y, t.kind, t.crop,
                 row_number() OVER (PARTITION BY t.episode_id, t.x, t.y
                                    ORDER BY t.step DESC) AS rn
          FROM tiles_delta t
          JOIN day25 d ON d.episode_id = t.episode_id
          WHERE t.player = d.wp AND t.step <= 24*25
            AND (t.kind IS NOT NULL OR t.crop IS NOT NULL)
        )
        SELECT x, y, arg_max(kind, n := rn) AS kind, arg_max(crop, rn) AS crop
        FROM (SELECT episode_id, x, y, kind, crop, rn FROM tiles) t
        GROUP BY episode_id, x, y
    """)) if False else None
    # tiles_delta schema may differ; check columns first via fallback query
    try:
        cols = [d[0] for d in c.execute("DESCRIBE tiles_delta").fetchall()]
        out.append(f"\n(tiles_delta columns: {cols})")
    except Exception as e:
        out.append(f"\n(tiles_delta error: {e})")

    # ------------------------------------------------ 5. planting per 3-day period
    out.append("\n## 5. Average plants per 3-day period (winners, all crops)\n")
    out.append(md(c, """
        WITH win AS (
          SELECT e.episode_id,
                 CASE WHEN e.reward0 > e.reward1 THEN 0 ELSE 1 END AS wp
          FROM episodes e
        ),
        plants AS (
          SELECT a.episode_id, a.step/24 AS day
          FROM actions a JOIN win ON win.episode_id = a.episode_id AND a.player = win.wp
          WHERE a.op = 'PLANT'
        )
        SELECT (day/3)*3 AS period_start_day, count(*) AS plants, count(*)/3.0 AS per_day
        FROM plants GROUP BY 1 ORDER BY 1
    """))

    text = "\n".join(out)
    path = os.path.join(DOCS, "003-shops-crops-corr.md")
    with open(path, "w") as f:
        f.write(text)
    print(text[:3000])
