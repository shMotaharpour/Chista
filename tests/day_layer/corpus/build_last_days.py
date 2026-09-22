"""Rebuild the winners' last-day corpus. Not a test - a tool, run by hand.

The archive's last day of a season is where the harvests and the drops are, and the games to take are
the ones our own seat WON - so the bar is the best play in the archive rather than an average one. The
days are pulled with `build_real_days.extract`, which reads the same `hands_steps` / `farm_steps` /
`tiles_delta` and keeps an op only when the tile it was aimed at changed at that hour (F047).

    KAGGLE_REPLAYS_PARQUET=/path/to/replays_parquet \\
      python tests/day_layer/corpus/build_last_days.py

Needs duckdb: the analyses repo's own venv has one.
"""
import importlib.util
import json
import os
import pathlib

ROOT = os.environ.get(
    "KAGGLE_REPLAYS_PARQUET",
    "/chista/Chista/kaggriculture-episodes-analyses/data/replays_parquet",
)
OUT = pathlib.Path(__file__).parent / "last_days.json"
DUMP = "2026-09-20"
DAY = 29
WANTED = 8

_spec = importlib.util.spec_from_file_location(
    "build_real_days", pathlib.Path(__file__).parent / "build_real_days.py")
builder = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(builder)

import duckdb   # noqa: E402 - after the builder, which is what needs it


def main() -> None:
    con = duckdb.connect()
    con.execute("SET memory_limit='2GB'")
    rows = con.sql(f"""
        SELECT episode_id, agent1, reward0, reward1
        FROM '{ROOT}/{DUMP}/episodes.parquet'
        WHERE status0 = 'DONE' AND status1 = 'DONE'
        ORDER BY reward0 DESC
    """).df()

    days = []
    for row in rows.itertuples():
        if row.reward0 <= row.reward1:
            continue                       # a day our seat lost is not the bar
        found = builder.extract(con, DUMP, int(row.episode_id), DAY)
        if not found:
            continue
        found["opponent"] = str(row.agent1)
        found["score"] = [int(row.reward0), int(row.reward1)]
        days.append(found)
        if len(days) >= WANTED:
            break

    OUT.write_text(json.dumps(days, separators=(",", ":")))
    print(f"{len(days)} winner days of {DUMP} day {DAY} -> {OUT} ({OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
