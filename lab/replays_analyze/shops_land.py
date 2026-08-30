"""Sections 6 & 7: town shop reaction + land purchases. Standalone output file 002."""
import duckdb
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REPLAYS = os.path.join(ROOT, "competition_replays", "parquet")
DOCS = os.path.join(ROOT, "docs", "replays Analysis")


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
    out = ["# Shops & Land — top player behavior", ""]

    out.append("## 6. Town shop unlocks and player reaction\n")
    out.append("### Shop unlock days\n")
    out.append(md(c, """
        WITH shops AS (
          SELECT episode_id, min(day) AS unlock_day, shop
          FROM (SELECT episode_id, day, unnest(string_split(town_shops, ',')) AS shop
                FROM steps WHERE town_shops != '') WHERE shop != ''
          GROUP BY 1, 3
        )
        SELECT shop, count(*) AS unlocks, round(avg(unlock_day),1) AS avg_day,
               min(unlock_day) AS earliest, max(unlock_day) AS latest
        FROM shops GROUP BY 1 ORDER BY unlocks DESC
    """))

    out.append("\n### Planting rate before vs after the crop's consuming shop unlocks\n")
    plants_sql = """
        WITH shop_demands AS (
          SELECT 'BAKERY' shop, 'EGG' crop UNION ALL
          SELECT 'BAKERY', 'WHEAT' UNION ALL
          SELECT 'PIZZA_SHOP', 'MILK' UNION ALL
          SELECT 'PIZZA_SHOP', 'TOMATO' UNION ALL
          SELECT 'BRUNCH_SPOT', 'EGG' UNION ALL
          SELECT 'BRUNCH_SPOT', 'STRAWBERRY' UNION ALL
          SELECT 'YARN_STORE', 'WOOL' UNION ALL
          SELECT 'ICE_CREAM_SHOP', 'STRAWBERRY' UNION ALL
          SELECT 'PET_CAFE', 'CARROT' UNION ALL
          SELECT 'SMOOTHIE_SHOP', 'STRAWBERRY' UNION ALL
          SELECT 'SMOOTHIE_SHOP', 'MILK' UNION ALL
          SELECT 'FARMERS_MARKET', 'WHEAT' UNION ALL
          SELECT 'FARMERS_MARKET', 'CARROT' UNION ALL
          SELECT 'FARMERS_MARKET', 'TOMATO' UNION ALL
          SELECT 'FARMERS_MARKET', 'STRAWBERRY'
        ),
        shops AS (
          SELECT episode_id, min(day) AS unlock_day, shop
          FROM (SELECT episode_id, day, unnest(string_split(town_shops, ',')) AS shop
                FROM steps WHERE town_shops != '') WHERE shop != ''
          GROUP BY 1, 3
        ),
        plants AS (
          SELECT a.episode_id, a.arg1 AS crop, a.step/24 AS day
          FROM actions a JOIN episodes e USING (episode_id)
          WHERE a.op='PLANT' AND ((a.player=0 AND e.reward0>e.reward1) OR (a.player=1 AND e.reward1>e.reward0))
        )
        SELECT sd.crop,
               round(avg(CASE WHEN p.day < s.unlock_day THEN 1.0 ELSE 0 END) * 24, 2) AS plants_per_day_before,
               round(avg(CASE WHEN p.day >= s.unlock_day THEN 1.0 ELSE 0 END) * 24, 2) AS plants_per_day_after,
               count(*) AS plant_events
        FROM shop_demands sd
        JOIN shops s ON s.shop = sd.shop
        JOIN plants p ON p.episode_id = s.episode_id AND p.crop = sd.crop
        GROUP BY 1 HAVING count(*) > 500 ORDER BY plant_events DESC
    """
    out.append(md(c, plants_sql))

    out.append("\n### Selling rate of the demanded crop before vs after shop unlock (winners)\n")
    sells_sql = """
        WITH shop_demands AS (
          SELECT 'BAKERY' shop, 'EGG' crop UNION ALL SELECT 'BAKERY','WHEAT' UNION ALL
          SELECT 'YARN_STORE','WOOL' UNION ALL SELECT 'PET_CAFE','CARROT' UNION ALL
          SELECT 'FARMERS_MARKET','WHEAT' UNION ALL SELECT 'FARMERS_MARKET','STRAWBERRY'
        ),
        shops AS (
          SELECT episode_id, min(day) AS unlock_day, shop
          FROM (SELECT episode_id, day, unnest(string_split(town_shops, ',')) AS shop
                FROM steps WHERE town_shops != '') WHERE shop != ''
          GROUP BY 1, 3
        ),
        sells AS (
          SELECT m.episode_id, m.item AS crop, m.step/24 AS day
          FROM market_orders m JOIN episodes e USING (episode_id)
          WHERE m.op='SELL' AND ((m.player=0 AND e.reward0>e.reward1) OR (m.player=1 AND e.reward1>e.reward0))
        )
        SELECT sd.crop,
               round(avg(CASE WHEN sl.day < s.unlock_day THEN 1.0 ELSE 0 END) * 24, 2) AS sells_per_day_before,
               round(avg(CASE WHEN sl.day >= s.unlock_day THEN 1.0 ELSE 0 END) * 24, 2) AS sells_per_day_after,
               count(*) AS sell_events
        FROM shop_demands sd
        JOIN shops s ON s.shop = sd.shop
        JOIN sells sl ON sl.episode_id = s.episode_id AND sl.crop = sd.crop
        GROUP BY 1 HAVING count(*) > 300 ORDER BY sell_events DESC
    """
    out.append(md(c, sells_sql))

    out.append("\n## 7. Land purchases (winners)\n")
    out.append("### Quadrants bought per winning episode\n")
    out.append(md(c, """
        WITH per_ep AS (
          SELECT episode_id, player, count(*) AS buys
          FROM actions WHERE op='BUY_LAND' GROUP BY 1,2
        )
        SELECT buys AS quadrants_bought, count(*) AS episodes
        FROM per_ep GROUP BY 1 ORDER BY 1
    """))
    out.append("\n### Timing of each successive land purchase (winner episodes)\n")
    out.append(md(c, """
        WITH nth_land AS (
          SELECT episode_id, player, step/24 AS day,
                 row_number() OVER (PARTITION BY episode_id, player ORDER BY step) AS nth
          FROM actions WHERE op='BUY_LAND'
        )
        SELECT nth AS quadrant_nth, count(*) AS episodes,
               round(avg(day),1) AS avg_day, min(day) AS earliest, max(day) AS latest
        FROM nth_land GROUP BY 1 ORDER BY 1
    """))

    text = "\n".join(out)
    path = os.path.join(DOCS, "002-shops-land.md")
    with open(path, "w") as f:
        f.write(text)
    print(text[:3500])


if __name__ == "__main__":
    main()
