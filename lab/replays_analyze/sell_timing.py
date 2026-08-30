"""Deep analysis phase 2: how do top players trade, plant, expand?
Output: docs/replays Analysis/001-sell-timing.md

Sections:
  1. Price-timed vs calendar sells
  2. Dump vs drip (order sizes)
  3. Arbitrage (BUY_PRODUCT usage)
  4. Planting portfolio over time
  5. End-of-season revenue pattern
  6. Town shop unlock reaction
  7. Land purchase timing
"""
from __future__ import annotations

import duckdb
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REPLAYS = os.path.join(ROOT, "competition_replays", "parquet")
DOCS = os.path.join(ROOT, "docs", "replays Analysis")
PRODUCTS = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL"]

WINNER = "((m.player=0 AND e.reward0>e.reward1) OR (m.player=1 AND e.reward1>e.reward0))"
W_ACTIONS = "((a.player=0 AND e.reward0>e.reward1) OR (a.player=1 AND e.reward1>e.reward0))"


def con():
    c = duckdb.connect()
    for t in ("episodes", "steps", "farm_steps", "private_steps", "actions",
              "market_orders", "tiles_delta"):
        c.execute(f"CREATE VIEW {t} AS SELECT * FROM '{REPLAYS}/*/{t}.parquet'")
    return c


def md(c, sql):
    return c.execute(sql).df().to_markdown(index=False)


def main():
    c = con()
    out = ["# How Top Players Trade — deep replay analysis", "",
           "Data: 6,269 competition episodes (Aug 15-24). 'Winner' = higher final bank.\n"]

    # ---------------------------------------------- 1. price-timed sells?
    out.append("## 1. Price-timed or calendar? Price when selling vs season median\n")
    out.append("| item | sells | avg price at sell | season median | ratio |")
    out.append("|---|---|---|---|---|")
    for item in PRODUCTS:
        r = c.execute(f"""
            SELECT count(*), avg(s.price_{item}), median(s.price_{item})
            FROM market_orders m
            JOIN episodes e USING (episode_id)
            JOIN steps s ON s.episode_id = e.episode_id AND s.step = m.step
            WHERE m.op='SELL' AND m.item = '{item}' AND {WINNER}
        """).fetchone()
        med = c.execute(f"SELECT median(price_{item}) FROM steps").fetchone()[0]
        ratio = r[1] / med if med else 0
        out.append(f"| {item} | {r[0]:,} | ${r[1]:.1f} | ${med:.0f} | {ratio:.2f} |")
    out.append("")

    # ------------------------------------------------ 2. dump vs drip
    out.append("## 2. Dump vs drip — order sizes (winners)\n")
    out.append(md(c, f"""
        SELECT m.item, count(*) AS orders,
               round(avg(m.qty),1) AS avg_units_per_order,
               max(m.qty) AS max_order
        FROM market_orders m JOIN episodes e USING (episode_id)
        WHERE m.op='SELL' AND {WINNER}
        GROUP BY 1 ORDER BY orders DESC
    """))
    out.append("\nOrders-per-turn distribution (SELL, winners):")
    out.append(md(c, """
        WITH per_turn AS (
          SELECT episode_id, step, player, count(*) AS n_orders
          FROM market_orders WHERE op='SELL' GROUP BY 1,2,3
        )
        SELECT n_orders, count(*) AS turns
        FROM per_turn GROUP BY 1 ORDER BY 1 LIMIT 12
    """))

    # ------------------------------------------------ 3. arbitrage
    out.append("\n## 3. BUY_PRODUCT usage by winners (arbitrage?)\n")
    out.append(md(c, f"""
        SELECT m.item, count(*) AS buys, sum(m.qty) AS units,
               count(DISTINCT m.episode_id) AS episodes_using
        FROM market_orders m JOIN episodes e USING (episode_id)
        WHERE m.op='BUY_PRODUCT' AND {WINNER}
        GROUP BY 1 ORDER BY episodes_using DESC
    """))

    # ------------------------------------------------ 4. planting over time
    out.append("\n## 4. Planting portfolio by phase (winners)\n")
    out.append(md(c, f"""
        SELECT CASE WHEN a.step/24 < 5 THEN 'd0-5' WHEN a.step/24 <= 15 THEN 'd6-15' ELSE 'd16-25' END AS phase,
               a.arg1 AS crop, count(*) AS plants
        FROM actions a JOIN episodes e USING (episode_id)
        WHERE a.op='PLANT' AND {W_ACTIONS}
        GROUP BY 1, 2 ORDER BY 1, 3 DESC
    """))

    # ------------------------------------------------ 5. end of season
    out.append("\n## 5. Revenue by period (winners)\n")
    out.append(md(c, f"""
        SELECT CASE WHEN m.step/24 >= 28 THEN 'd28-30' WHEN m.step/24 >= 25 THEN 'd25-27' ELSE 'before d25' END AS period,
               sum(m.qty) AS units, count(*) AS sells
        FROM market_orders m JOIN episodes e USING (episode_id)
        WHERE m.op='SELL' AND {WINNER}
        GROUP BY 1 ORDER BY 1
    """))

    text = "\n".join(out)
    path = os.path.join(DOCS, "001-sell-timing.md")
    with open(path, "w") as f:
        f.write(text)
    print(text[:5000])


if __name__ == "__main__":
    main()
