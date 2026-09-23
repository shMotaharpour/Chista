"""Rebuild the real-day corpus from the archive. Not a test - a tool, run by hand.

The archive is two gigabytes and lives in another repo, so the tests never read it: this writes the
land work of a spread of days as JSON, and `test_real_days.py` checks that in. Run it with an
interpreter that has duckdb - the analyses repo's own venv has one - and point
`KAGGLE_REPLAYS_PARQUET` somewhere else if the dumps move:

    KAGGLE_REPLAYS_PARQUET=/path/to/replays_parquet \\
      python tests/day_layer/corpus/build_real_days.py

The extractor itself lives in `offline_lab/bench/corpus.py`, next to the benchmark that reads the same
archive: one reader, not two. Its docstring says why an op is kept only when the tile it was aimed at
actually changed at that hour (F047), and why the hand hours come from `hands_steps`.
"""
import json
import pathlib
import random
import sys

import duckdb

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

from offline_lab.bench.corpus import ROOT, dumps, extract  # noqa: E402

OUT = pathlib.Path(__file__).parent / "real_days.json"
PER_SET = 30
SEED = 20260920
MID_DAY = 5
FIRST_DUMP, FIRST_DAYS, FIRST_EPISODES = "2026-09-16", (1, 3, 6, 10), 3


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
