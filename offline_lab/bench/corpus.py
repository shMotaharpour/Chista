"""Build the day layer's benchmark corpus: a few winning games, sampled across the season.

Not a test and not a fixture - this is the lab's own corpus, so the day layer can be timed and tuned
on the shapes of day a season actually produces: how much land the game had bought by then, and how
many ops the day asked for. A winner buys the second and third quadrant somewhere in the first
fortnight, so one day per game cannot show that. The sample is a handful of winning games, each read
at a spread of days, and every entry carries the quadrants the game held at hour 0 of that day.

    KAGGLE_REPLAYS_PARQUET=/path/to/replays_parquet \\
      python offline_lab/bench/corpus.py

An op is kept only if the tile it was aimed at actually changed at that hour - a submitted op is not
a fact (F047). The extractor lives here because the archive reading belongs to the lab; the tests'
own builder imports it from this module rather than keeping a second copy.
"""
import json
import os
import pathlib

import duckdb
import pandas as pd

ROOT = os.environ.get(
    "KAGGLE_REPLAYS_PARQUET",
    "/chista/Chista/kaggriculture-episodes-analyses/data/replays_parquet",
)
OUT = pathlib.Path(__file__).parent / "days.json"
MOVES = {"NORTH", "SOUTH", "EAST", "WEST", "PASS", "PICKUP", "DROP"}
DAYS = (1, 3, 5, 8, 12, 16, 20, 24, 28)
GAMES = 4


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

    # The land the game held when the day started. A winner buys the second and third quadrant in the
    # first fortnight, so this is what makes a day's field wide or narrow, and it is the axis the
    # benchmark sorts by.
    held = con.sql(f"""
        SELECT quadrants FROM '{ROOT}/{dump}/farm_steps.parquet'
        WHERE episode_id = {episode} AND player = false
          AND step // 24 = {day} AND step % 24 = 0
    """).df()
    quadrants = [] if held.empty else str(held.iloc[0, 0]).split(",")

    if not chains:
        return None
    return {
        "dump": dump,
        "episode": int(episode),
        "day": int(day),
        "hands": len(hours),
        "hire_times": sorted(hours),
        "quadrants": quadrants,
        "ops": sum(len(ops) for ops in chains.values()),
        "available": available,
        "chains": [[list(cell), ops, entity.get(cell)]
                   for cell, ops in sorted(chains.items())],
        # The hour the GAME ran each op, in the same order as that cell's ops: the only record of what
        # the walks, trips and pickups cost the agent that actually played the day.
        "op_hours": [[list(cell), op_hours[cell]] for cell, _ops in sorted(chains.items())],
    }


def winners(con, dump: str, limit: int) -> list[int]:
    """The games our own seat won, best first."""
    rows = con.sql(f"""
        SELECT episode_id, reward0, reward1 FROM '{ROOT}/{dump}/episodes.parquet'
        WHERE status0 = 'DONE' AND status1 = 'DONE' ORDER BY reward0 DESC
    """).df()
    won = rows[rows.reward0 > rows.reward1]
    return [int(r.episode_id) for r in won.head(limit).itertuples()]


def main() -> None:
    con = duckdb.connect()
    con.execute("SET memory_limit='2GB'")
    newest = dumps()[-1]
    days = []
    for episode in winners(con, newest, GAMES):
        for day in DAYS:
            found = extract(con, newest, episode, day)
            if found:
                days.append(found)
    OUT.write_text(json.dumps(days, separators=(",", ":")))
    widths = sorted({len(d["quadrants"]) for d in days})
    print(f"{len(days)} days, {len({d['episode'] for d in days})} winning games, "
          f"quadrant counts {widths}, ops {min(d['ops'] for d in days)}-{max(d['ops'] for d in days)}, "
          f"{OUT.stat().st_size} bytes -> {OUT}")


if __name__ == "__main__":
    main()
