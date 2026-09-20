"""Rebuild the real-day fixture from the archive. Not a test - a tool, run by hand.

The archive is two gigabytes and lives in another repo, so the tests never read it: this writes the
land work of a spread of days as JSON, and `test_real_days.py` checks that in. Run it with a
interpreter that has duckdb - the analyses repo's own venv has one - and point `KAGGLE_REPLAYS_PARQUET`
somewhere else if the dumps move:

    KAGGLE_REPLAYS_PARQUET=/path/to/replays_parquet \\
      python tests/day_layer/fixtures/extract_real_days.py

A day is the ops that landed on every tile, the hands the player paid for them, and the shed it began
with. Movement ops are not tile work and are dropped; a hand that did nothing has a null op, which
arrives as NaN and not as None.
"""
import json
import os
import pathlib

import duckdb

ROOT = os.environ.get(
    "KAGGLE_REPLAYS_PARQUET",
    "/chista/Chista/kaggriculture-episodes-analyses/data/replays_parquet",
)
MOVES = {"NORTH", "SOUTH", "EAST", "WEST", "PASS", "PICKUP", "DROP"}
DUMP = "2026-09-16"
DAYS = (1, 3, 6, 10)
EPISODES = 3
OUT = pathlib.Path(__file__).parent / "real_days.json"


def main() -> None:
    con = duckdb.connect()
    con.execute("SET memory_limit='2GB'")
    episodes = [
        row[0] for row in con.sql(f"""
            SELECT DISTINCT episode_id FROM '{ROOT}/{DUMP}/farm_steps.parquet' WHERE player = false
            ORDER BY episode_id LIMIT {EPISODES}
        """).fetchall()
    ]

    days = []
    for episode in episodes:
        for day in DAYS:
            rows = con.sql(f"""
                SELECT step % 24 AS hour, x, y, op
                FROM '{ROOT}/{DUMP}/hands_steps.parquet'
                WHERE episode_id = {episode} AND player = false AND step // 24 = {day}
                UNION ALL
                SELECT step % 24 AS hour, farmer_x AS x, farmer_y AS y, op
                FROM '{ROOT}/{DUMP}/farm_steps.parquet'
                WHERE episode_id = {episode} AND player = false AND step // 24 = {day}
                ORDER BY hour
            """).df()
            if rows.empty:
                continue
            hands = con.sql(f"""
                SELECT max(n_hands) FROM '{ROOT}/{DUMP}/farm_steps.parquet'
                WHERE episode_id = {episode} AND player = false AND step // 24 = {day}
            """).fetchone()[0]
            shed = con.sql(f"""
                SELECT * FROM '{ROOT}/{DUMP}/private_steps.parquet'
                WHERE episode_id = {episode} AND player = false
                  AND step // 24 = {day} AND step % 24 = 0
            """).df()

            available = {}
            if not shed.empty:
                row = shed.iloc[0]
                for column in row.index:
                    if column.startswith("shed_") and int(row[column]) > 0:
                        available[column[5:]] = 1

            chains: dict[tuple[int, int], list[str]] = {}
            for unit in rows.itertuples():
                if not isinstance(unit.op, str) or unit.op in MOVES:
                    continue
                chains.setdefault((int(unit.x), int(unit.y)), []).append(unit.op)

            days.append({
                "episode": int(episode),
                "day": int(day),
                "hands": int(hands or 0),
                "available": available,
                "chains": [[list(cell), ops] for cell, ops in sorted(chains.items())],
            })

    OUT.write_text(json.dumps(days, separators=(",", ":")))
    print(f"{len(days)} days, {OUT.stat().st_size} bytes -> {OUT}")


if __name__ == "__main__":
    main()
