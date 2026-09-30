"""What each seat REALISED, from the city's own inventory -- computed inside DuckDB.

`city_steps.inv_{9}` is the market's inventory: certain, not inferred. Over one step,
    inv_g(t) - inv_g(t-1) = (our sells + their sells)_g - drain_g(t)
so with our own sells known and the drain derived from the open shops, the rival's
realised flow is the residual. That number is what the scenario axis needs; the logged
`market_orders` rows are orders, and an order capped by an empty shed is not a sale.

Everything (lag, the shop baskets, the join) runs in SQL. Only the summary comes back.
Rules are imported and interpolated into the query -- never retyped (R002).
"""
from __future__ import annotations

import os
import sys

import duckdb

sys.path.insert(0, "/home/amirelite_ai/Chista/ChistaWRS")
import agent.world.rules as R  # noqa: E402

GOODS = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER")
INTERVAL = R.SHOP_SELL_INTERVAL_TURNS
# the shop -> goods map, with the measured double-consumption of single-product shops
baskets = [(s, g, (2 if len(gs) == 1 else 1))
           for s, gs in R.SHOPS.items() for g in gs if g in GOODS]
CENTRE = ", ".join(f"('{g}')" for g in R.TOWN_CENTER_PRODUCTS if g in GOODS)
VALUES = ", ".join(f"('{s}', '{g}', {w})" for s, g, w in baskets)

QUERY = f"""
WITH inv AS (
  SELECT episode_id, step, town_shops, {", ".join(f"inv_{g} AS {g}" for g in GOODS)},
         {', '.join(f'inv_{g} - lag(inv_{g}) OVER w AS d_{g}' for g in GOODS)}
  FROM read_parquet('{{part}}/city_steps.parquet')
  WINDOW w AS (PARTITION BY episode_id ORDER BY step)
),
basket AS (SELECT * FROM (VALUES {VALUES}) AS t(shop, item, units)),
centre AS (SELECT item FROM (VALUES {CENTRE}) AS t(item)),
drain AS (
  SELECT i.episode_id, i.step, b.item, sum(b.units) / {INTERVAL}.0 AS drain
  FROM inv i
  CROSS JOIN UNNEST(string_split(coalesce(i.town_shops, ''), ',')) AS u(shop)
  JOIN basket b ON b.shop = u.shop
  GROUP BY 1, 2, 3
),
centre_drain AS (
  SELECT i.episode_id, i.step, c.item, 1.0 / {INTERVAL}.0 AS drain
  FROM inv i CROSS JOIN centre c
),
ours AS (
  SELECT episode_id, step, item, sum(qty) AS ours
  FROM read_parquet('{{part}}/market_orders.parquet')
  WHERE op = 'SELL' AND player = TRUE GROUP BY 1, 2, 3
),
theirs AS (
  SELECT episode_id, step, item, sum(qty) AS logged
  FROM read_parquet('{{part}}/market_orders.parquet')
  WHERE op = 'SELL' AND player = FALSE GROUP BY 1, 2, 3
),
long AS (
  {' UNION ALL '.join(
      f"SELECT episode_id, step, (step / 24)::INT AS day, '{g}' AS item, d_{g} AS d_inv FROM inv"
      for g in GOODS)}
)
SELECT l.episode_id, l.day, l.item,
       sum(l.d_inv)                                       AS d_inv,
       sum(coalesce(dr.drain, 0))                          AS drain,
       sum(coalesce(o.ours, 0))                            AS ours,
       sum(l.d_inv) + sum(coalesce(dr.drain, 0)) - sum(coalesce(o.ours, 0)) AS theirs_realised,
       sum(coalesce(th.logged, 0))                         AS theirs_logged
FROM long l
LEFT JOIN (SELECT episode_id, step, item, sum(drain) AS drain FROM (
             SELECT * FROM drain UNION ALL SELECT * FROM centre_drain)
           GROUP BY 1, 2, 3) dr
        ON dr.episode_id = l.episode_id AND dr.step = l.step AND dr.item = l.item
LEFT JOIN ours   o  ON o.episode_id  = l.episode_id  AND o.step  = l.step  AND o.item  = l.item
LEFT JOIN theirs th ON th.episode_id = l.episode_id  AND th.step = l.step AND th.item = l.item
GROUP BY 1, 2, 3
"""


def main() -> None:
    part = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser(
        "~/Chista/kaggriculture-episodes-analyses/data/replays_parquet/2026-09-19")
    con = duckdb.connect()
    for st in ("SET memory_limit='2GB'", "SET threads=4", "SET temp_directory='/tmp/duckdb_swap'",
               "SET preserve_insertion_order=false"):
        con.execute(st)
    con.execute(f"CREATE TABLE COLS AS {QUERY.format(part=part, CENTRE=CENTRE)}")
    print("rows:", con.execute("SELECT count(*) FROM COLS").fetchone()[0])
    print(con.execute("""
        SELECT item,
               round(sum(d_inv), 0)        AS inv_change,
               round(sum(drain), 0)        AS town_drain,
               round(sum(ours), 0)         AS ours,
               round(sum(theirs_realised), 0) AS theirs_realised,
               round(sum(theirs_logged), 0)   AS theirs_logged,
               round(100.0 * sum(theirs_realised) / nullif(sum(theirs_logged), 0), 1) AS pct_realised
        FROM COLS GROUP BY 1 ORDER BY abs(sum(theirs_logged)) DESC""").fetchdf().to_string(index=False))
    con.execute("COPY COLS TO '/tmp/scenario_axes/realised_supply.parquet' (FORMAT PARQUET)")
    print("written /tmp/scenario_axes/realised_supply.parquet")


if __name__ == "__main__":
    main()
