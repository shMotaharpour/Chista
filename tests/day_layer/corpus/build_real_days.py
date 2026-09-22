"""Rebuild the real-day corpus from the archive. Not a test - a tool, run by hand.

The archive is two gigabytes and lives in another repo, so the tests never read it: this writes the
land work of a spread of days as JSON, and `test_real_days.py` checks that in. Run it with an
interpreter that has duckdb - the analyses repo's own venv has one - and point
`KAGGLE_REPLAYS_PARQUET` somewhere else if the dumps move:

    KAGGLE_REPLAYS_PARQUET=/path/to/replays_parquet \\
      python tests/day_layer/corpus/build_real_days.py

**An op is kept only if the tile it was aimed at actually changed at that hour.** `hands_steps.op` is
what the agent ASKED for, and the engine refuses a malformed action in silence (F047) - so a
submitted op is not a fact. One day of one dump submits water, harvest and fertilize on a tile the
archive says was never harvested, and the tile is planted four times over the day because the agent
kept trying. `tiles_delta` records the tiles that changed, so the two together say which asks
happened: a unit submits one op a turn and the engine runs it on the unit's own tile, so a change at
that tile and that hour is that op having taken effect.

The hand hours come from `hands_steps` too - a hand hired in turn 2 acts from hour 3, not from hour 1,
so the fixture carries when each hand really began.
"""
import json
import os
import pathlib
import random

import duckdb
import pandas as pd

ROOT = os.environ.get(
    "KAGGLE_REPLAYS_PARQUET",
    "/chista/Chista/kaggriculture-episodes-analyses/data/replays_parquet",
)
OUT = pathlib.Path(__file__).parent / "real_days.json"
MOVES = {"NORTH", "SOUTH", "EAST", "WEST", "PASS", "PICKUP", "DROP"}
PER_SET = 30
SEED = 20260920
MID_DAY = 5
FIRST_DUMP, FIRST_DAYS, FIRST_EPISODES = "2026-09-16", (1, 3, 6, 10), 3


def dumps() -> list[str]:
    return sorted(p.name for p in pathlib.Path(ROOT).iterdir() if p.is_dir())


def extract(con, dump: str, episode: int, day: int):
    """One day's real land work, or None when the dump has no such day."""
    changed = con.sql(f"""
        SELECT DISTINCT x, y, step % 24 AS hour
        FROM '{ROOT}/{dump}/tiles_delta.parquet'
        WHERE episode_id = {episode} AND player = false AND step // 24 = {day}
    """).df()
    if changed.empty:
        return None
    happened = {(int(r.x), int(r.y), int(r.hour)) for r in changed.itertuples()}

    submitted = con.sql(f"""
        SELECT step % 24 AS hour, x, y, op
        FROM '{ROOT}/{dump}/hands_steps.parquet'
        WHERE episode_id = {episode} AND player = false AND step // 24 = {day}
        UNION ALL
        SELECT step % 24 AS hour, farmer_x AS x, farmer_y AS y, op
        FROM '{ROOT}/{dump}/farm_steps.parquet'
        WHERE episode_id = {episode} AND player = false AND step // 24 = {day}
        ORDER BY hour
    """).df()

    chains: dict[tuple[int, int], list[str]] = {}
    op_hours: dict[tuple[int, int], list[int]] = {}
    for row in submitted.itertuples():
        if not isinstance(row.op, str) or row.op in MOVES:
            continue
        cell = (int(row.x), int(row.y))
        if (cell[0], cell[1], int(row.hour)) not in happened:
            continue                       # the engine refused it: it never happened
        chains.setdefault(cell, []).append(row.op)
        op_hours.setdefault(cell, []).append(int(row.hour))

    # The crop or the animal the tile ended the day with, for the layer's entity.
    last = con.sql(f"""
        SELECT x, y, kind, crop, animal
        FROM '{ROOT}/{dump}/tiles_delta.parquet'
        WHERE episode_id = {episode} AND player = false AND step // 24 = {day}
        QUALIFY row_number() OVER (PARTITION BY x, y ORDER BY step DESC) = 1
    """).df()
    entity = {}
    for row in last.itertuples():
        if row.crop is not None and row.crop is not pd.NA:
            entity[(int(row.x), int(row.y))] = row.crop
        elif row.animal is not None and row.animal is not pd.NA:
            entity[(int(row.x), int(row.y))] = row.animal

    hours = [int(row[0]) for row in con.sql(f"""
        SELECT min(step % 24) FROM '{ROOT}/{dump}/hands_steps.parquet'
        WHERE episode_id = {episode} AND player = false AND step // 24 = {day}
        GROUP BY unit ORDER BY unit
    """).fetchall()]

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

    if not chains:
        return None
    return {
        "dump": dump,
        "episode": int(episode),
        "day": int(day),
        "hands": len(hours),
        "hire_times": sorted(hours),
        "available": available,
        "chains": [[list(cell), ops, entity.get(cell)]
                   for cell, ops in sorted(chains.items())],
        # The hour the GAME ran each op, in the same order as that cell's ops. The layer is charged
        # for its walks, its trips and its pickups, and this is the only record of what those cost
        # the agent that actually played the day - so a day the layer cannot carry can be read
        # against the day the game did, op by op, instead of argued about.
        "op_hours": [[list(cell), op_hours[cell]] for cell, _ops in sorted(chains.items())],
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

    # 0: the twelve the fixture started with.
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
