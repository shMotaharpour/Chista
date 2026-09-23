"""What the day search costs on the archive's own days, day by day. Not a test - a tool, run by hand.

The manager plans a day across 24 turns with a one-second act timeout (F046), and the layer's own
timing test pins the LARGEST day in one corpus. The cost is not a function of size alone: `search` is
three nested loops, each a full `_run`, and which of them turn depends on the day.

  the doors   `_settle` re-searches until the hands' doors stop moving (up to four `_run`s) - every
              day pays this one, carried or not
  the charge  `_fixed_point` tightens the pickup charge while the day is not carried (up to four more
              `_settle`s) - only days the search cannot carry pay this
  the pool    `search` walks the pools the caller allowed, one `_fixed_point` each - a non-carried day
              pays it, and stops early when a bigger pool places no more

So the slowest days are not simply the biggest ones. It measures both samples by default - the
archive's spread and the strong players' days - and `--corpus` takes any day file the builder wrote:

    .venv/bin/python offline_lab/search_cost.py
    .venv/bin/python offline_lab/search_cost.py --worst 12
    .venv/bin/python offline_lab/search_cost.py --corpus tests/day_layer/corpus/strong_days.json
"""
import argparse
import json
import pathlib
import sys
import time

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from agent.wsr import beam as B
from agent.wsr import tasks as T

#: The samples the layer is measured on: the archive's spread and the strong players' days. They are
#: the tests' fixtures, and `--corpus` takes any file the builder wrote.
SAMPLES = "tests/day_layer/corpus/real_days.json,tests/day_layer/corpus/strong_days.json"
#: The turn's second: one act timeout (F046). A day that takes longer than this cannot be planned
#: inside the turn it is planned in.
TURN_MS = 1000.0


def _day(entry):
    grid = [(tuple(cell), tuple(ops), entity) for cell, ops, entity in entry["chains"]]
    available = {good: int(hour) for good, hour in entry["available"].items()}
    tasks = T.build(grid, available=available)
    hire_times = tuple(entry["hire_times"]) or (1,) * entry["hands"]
    day = B.Day(chains=tuple(grid), available=available, hire_times=hire_times)
    return day, tasks


def measure(entry):
    """One day: what the search cost, and what it managed to place."""
    day, tasks = _day(entry)
    started = time.perf_counter()
    result = B.search(day, tasks, hands=max(entry["hands"] - 1, 0), max_hands=entry["hands"])
    return {
        "dump": entry["dump"], "episode": entry["episode"], "day": entry["day"],
        "agent": entry.get("agent", ""),
        "tasks": tasks.n, "hands": entry["hands"], "pool": result.pool,
        "complete": result.complete, "placed": len(result.route),
        "ms": (time.perf_counter() - started) * 1000.0,
    }


def report(name: str, rows: list[dict], worst: int) -> None:
    over = [row for row in rows if row["ms"] > TURN_MS]
    slowest = sorted(rows, key=lambda row: -row["ms"])[:worst]
    biggest = max(rows, key=lambda row: row["tasks"])
    print(f"\n=== {name}: {len(rows)} days ===")
    print(f"  over the turn's second: {len(over)} of {len(rows)}")
    print(f"  the biggest day      : {biggest['tasks']:>4} tasks, {biggest['ms']:7.1f} ms, "
          f"complete {biggest['complete']}   <- what the timing test pins")
    print(f"  {'dump':<11} {'tasks':>5} {'hands':>5} {'pool':>4} {'complete':>8} {'placed':>6} "
          f"{'ms':>8}")
    for row in slowest:
        print(f"  {row['dump']:<11} {row['tasks']:>5} {row['hands']:>5} {row['pool']:>4} "
              f"{str(row['complete']):>8} {row['placed']:>6} {row['ms']:>8.1f}")
    if over:
        share = sum(not row["complete"] for row in over) / len(over)
        print(f"  of the {len(over)} over the second, {share:.0%} are days the search did not carry")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--worst", type=int, default=8, help="how many of the slowest days to print")
    parser.add_argument("--corpus", default=SAMPLES,
                        help="day files to measure, comma separated, relative to the repo root")
    args = parser.parse_args()

    for name in args.corpus.split(","):
        path = pathlib.Path(name.strip())
        path = path if path.is_absolute() else REPO / path
        entries = json.loads(path.read_text())
        report(path.name, [measure(entry) for entry in entries], args.worst)


if __name__ == "__main__":
    main()
