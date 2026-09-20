"""Rebuild the real-day fixture from the archive. Not a test - a tool, run by hand.

The archive is two gigabytes and lives in another repo, so the tests never read it: this writes the
land work of a spread of days as JSON, and `test_real_days.py` checks that in. Run it with an
interpreter that has duckdb - the analyses repo's own venv has one - and point
`KAGGLE_REPLAYS_PARQUET` somewhere else if the dumps move:

    KAGGLE_REPLAYS_PARQUET=/path/to/replays_parquet \\
      python tests/day_layer/fixtures/extract_real_days.py

A day is the ops that landed on every tile, the hands the player paid for them, and the shed it began
with. Movement ops are not tile work and are dropped; a hand that did nothing has a null op, which
arrives as NaN and not as None.

Three sets, so one day of one dump cannot stand for a season:

  A  a day from each of many dumps        - the agents change from dump to dump
  B  many days of one dump                - the season changes within a dump
  C  spread over dumps and days and games - the two together, chosen by a fixed seed
"""
import json
import os
import pathlib
import random

import duckdb

ROOT = os.environ.get(
    "KAGGLE_REPLAYS_PARQUET",
    "/chista/Chista/kaggriculture-episodes-analyses/data/replays_parquet",
)
MOVES = {"NORTH", "SOUTH", "EAST", "WEST", "PASS", "PICKUP", "DROP"}
OUT = pathlib.Path(__file__).parent / "real_days.json"
PER_SET = 30
SEED = 20260920
MID_DAY = 5
#: The twelve the fixture started with: three games of one dump, four days each. Kept in front so
#: the fixture is a superset of what it was, and a test that once passed cannot quietly stop being
#: asked.
FIRST_DUMP, FIRST_DAYS, FIRST_EPISODES = "2026-09-16", (1, 3, 6, 10), 3


def dumps() -> list[str]:
    return sorted(p.name for p in pathlib.Path(ROOT).iterdir() if p.is_dir())


def extract(con, dump: str, episode: int, day: int):
    """One day's land work, or None when the dump has no such day."""
    rows = con.sql(f"""
        SELECT step % 24 AS hour, x, y, op
        FROM '{ROOT}/{dump}/hands_steps.parquet'
        WHERE episode_id = {episode} AND player = false AND step // 24 = {day}
        UNION ALL
        SELECT step % 24 AS hour, farmer_x AS x, farmer_y AS y, op
        FROM '{ROOT}/{dump}/farm_steps.parquet'
        WHERE episode_id = {episode} AND player = false AND step // 24 = {day}
        ORDER BY hour
    """).df()
    if rows.empty:
        return None

    hands = con.sql(f"""
        SELECT max(n_hands) FROM '{ROOT}/{dump}/farm_steps.parquet'
        WHERE episode_id = {episode} AND player = false AND step // 24 = {day}
    """).fetchone()[0]
    shed = con.sql(f"""
        SELECT * FROM '{ROOT}/{dump}/private_steps.parquet'
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

    return {
        "dump": dump,
        "episode": int(episode),
        "day": int(day),
        "hands": int(hands or 0),
        "available": available,
        "chains": [[list(cell), ops] for cell, ops in sorted(chains.items())],
    }


def main() -> None:
    con = duckdb.connect()
    con.execute("SET memory_limit='2GB'")
    rng = random.Random(SEED)
    every = dumps()
    days = []

    def episodes_of(dump: str, limit: int = 40) -> list[int]:
        return [row[0] for row in con.sql(f"""
            SELECT DISTINCT episode_id FROM '{ROOT}/{dump}/farm_steps.parquet'
            WHERE player = false ORDER BY episode_id LIMIT {limit}
        """).fetchall()]

    # 0: the original twelve.
    for episode in episodes_of(FIRST_DUMP, FIRST_EPISODES):
        for day in FIRST_DAYS:
            found = extract(con, FIRST_DUMP, episode, day)
            if found:
                days.append(found)

    # A: a mid-season day from each of many dumps.
    step = max(1, len(every) // PER_SET)
    for dump in every[::step][:PER_SET]:
        for episode in episodes_of(dump, 2):
            found = extract(con, dump, episode, MID_DAY)
            if found:
                days.append(found)
                break

    # B: many days of one dump, so the season changes inside a single sample.
    home = every[len(every) // 2]
    for episode in episodes_of(home, 2):
        for day in range(1, 29):
            found = extract(con, home, episode, day)
            if found:
                days.append(found)
            if len([d for d in days if d["dump"] == home]) >= PER_SET:
                break
        break

    # C: dumps and days and games together, chosen by a fixed seed.
    while len(days) < FIRST_EPISODES * len(FIRST_DAYS) + 3 * PER_SET:
        dump = rng.choice(every)
        pool = episodes_of(dump, 40)
        if not pool:
            continue
        found = extract(con, dump, rng.choice(pool), rng.randint(1, 28))
        if found:
            days.append(found)

    OUT.write_text(json.dumps(days, separators=(",", ":")))
    print(f"{len(days)} days from {len({d['dump'] for d in days})} dumps, "
          f"{OUT.stat().st_size} bytes -> {OUT}")


if __name__ == "__main__":
    main()
