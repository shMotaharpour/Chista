"""What each seat REALISED, from the city's own inventory, per SEAT.

`city_steps.inv_{9}` is certain, so the flow identity
    inv_g(t) - inv_g(t-1) = (our flow + their flow)_g - drain_g(t)
gives one seat's realised flow as the residual. It is solved PER SEAT, not per seat-pair, because
the planning axis has to be keyed by AGENT IDENTITY: the owner's agent plays both seats in the
competition, so a seat index is not a definition of the opponent.

WHEAT and FERTILIZER are two-way goods -- a buy is quoted at the pre-trade inventory and a sale
earns the post-trade one, so both enter with the same sign and only their NET is observable. That
is why a wheat residual may legitimately be negative. The other seven can only be sold, so their
flow is one-directional and non-negative, and the identity is held to that.
"""
from __future__ import annotations

import os
import sys

import duckdb

sys.path.insert(0, os.path.expanduser("~/Chista/ChistaWRS"))
import agent.world.rules as R  # noqa: E402

GOODS = ('WHEAT', 'CARROT', 'TOMATO', 'STRAWBERRY', 'MELON', 'EGG', 'MILK', 'WOOL', 'FERTILIZER')
TWO_WAY = ('WHEAT', 'FERTILIZER')
ONE_WAY = ('CARROT', 'TOMATO', 'STRAWBERRY', 'MELON', 'EGG', 'MILK', 'WOOL')
INTERVAL = R.SHOP_SELL_INTERVAL_TURNS
baskets = [(s, g, (2 if len(gs) == 1 else 1)) for s, gs in R.SHOPS.items() for g in gs if g in GOODS]
CENTRE = tuple(g for g in R.TOWN_CENTER_PRODUCTS if g in GOODS)

INV = ", ".join(f"inv_{g} AS {g}" for g in GOODS)
DINV = ", ".join(f"inv_{g} - lag(inv_{g}) OVER w AS d_{g}" for g in GOODS)
VALUES = ", ".join(f"({s!r}, {g!r}, {w})" for s, g, w in baskets)
CENTRE_V = ", ".join(f"({g!r})" for g in CENTRE)
LONG = " UNION ALL ".join(
    f"SELECT episode_id, step, (step / 24)::INT AS day, {g!r} AS item, d_{g} AS d_inv FROM inv"
    for g in GOODS)


def seat_query(part: str, seat: str) -> str:
    return f"""
    WITH inv AS (SELECT episode_id, step, town_shops, {INV}, {DINV}
                 FROM read_parquet('{part}/city_steps.parquet')
                 WINDOW w AS (PARTITION BY episode_id ORDER BY step)),
    basket AS (SELECT * FROM (VALUES {VALUES}) AS t(shop, item, units)),
    centre AS (SELECT item FROM (VALUES {CENTRE_V}) AS t(item)),
    drain AS (SELECT i.episode_id, i.step, b.item, sum(b.units) / {INTERVAL}.0 AS d
              FROM inv i CROSS JOIN UNNEST(string_split(coalesce(i.town_shops, ''), ',')) AS u(shop)
              JOIN basket b ON b.shop = u.shop GROUP BY 1, 2, 3),
    centre_drain AS (SELECT i.episode_id, i.step, c.item, 1.0 / {INTERVAL}.0 AS d
                     FROM inv i CROSS JOIN centre c),
    ours AS (SELECT episode_id, step, item,
                    sum(qty) FILTER (WHERE op = 'SELL')  AS sells,
                    sum(qty) FILTER (WHERE op <> 'SELL') AS buys
             FROM read_parquet('{part}/market_orders.parquet') WHERE player = {seat} GROUP BY 1, 2, 3),
    long AS ({LONG})
    SELECT l.episode_id, l.day, l.item,
           sum(l.d_inv) + sum(coalesce(dr.d, 0)) -
           CASE WHEN l.item IN {TWO_WAY!r}
                THEN sum(coalesce(o.sells, 0) - coalesce(o.buys, 0))
                ELSE sum(coalesce(o.sells, 0)) END        AS flow,
           {seat} AS seat
    FROM long l
    LEFT JOIN (SELECT episode_id, step, item, sum(d) AS d FROM (
                 SELECT * FROM drain UNION ALL SELECT * FROM centre_drain) GROUP BY 1, 2, 3) dr
           ON dr.episode_id = l.episode_id AND dr.step = l.step AND dr.item = l.item
    LEFT JOIN ours o ON o.episode_id = l.episode_id AND o.step = l.step AND o.item = l.item
    GROUP BY 1, 2, 3"""


def main() -> None:
    part = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser(
        "~/Chista/kaggriculture-episodes-analyses/data/replays_parquet/2026-09-19")
    con = duckdb.connect()
    for st in ("SET memory_limit='2GB'", "SET threads=4", "SET temp_directory='/tmp/duckdb_swap'",
               "SET preserve_insertion_order=false"):
        con.execute(st)
    # each seat query opens with WITH, so it is wrapped: a UNION ALL cannot follow CREATE ... AS WITH
    parts = [f"SELECT * FROM ({seat_query(part, seat=bool_)})" for bool_ in ("TRUE", "FALSE")]
    con.execute("CREATE TABLE FLOW AS " + " UNION ALL ".join(parts))
    print("rows:", con.execute("SELECT count(*) FROM FLOW").fetchone()[0])
    bad = con.execute(f"SELECT item, round(sum(flow), 0) AS f FROM FLOW "
                      f"WHERE item IN {ONE_WAY!r} GROUP BY 1 HAVING sum(flow) < 0").fetchall()
    if bad:
        raise SystemExit(f"identity broken: a sell-only good came out negative: {bad}")
    print("guard: the seven sell-only goods are non-negative; the two-way ones carry a sign")
    print(con.execute("""
        SELECT item, round(sum(flow) FILTER (WHERE seat) , 0) AS seat0_flow,
               round(sum(flow) FILTER (WHERE NOT seat), 0) AS seat1_flow,
               round(sum(abs(flow)), 0) AS abs_total
        FROM FLOW GROUP BY 1 ORDER BY abs_total DESC""").fetchdf().to_string(index=False))
    con.execute("COPY FLOW TO '/tmp/scenario_axes/realised_by_seat.parquet' (FORMAT PARQUET)")
    print("written /tmp/scenario_axes/realised_by_seat.parquet")


if __name__ == "__main__":
    main()
