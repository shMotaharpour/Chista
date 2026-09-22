"""Time the day layer over the benchmark corpus, and say what the seconds buy.

The corpus is `days.json` (built by `corpus.py`): a few winning games read at a spread of days, so
each entry carries the quadrants the game held and how many ops the day asked for. This is not a
test - it is the harness a change to the search is measured on before it becomes one.

    python offline_lab/bench/sweep.py [--beam N] [--budget S]

Every day is searched once with the layer's own choices and once with the override, and the two
answers are compared: a lever that buys time by placing less is not a lever, it is a loss of
accuracy, and the table has to show that.
"""
import argparse
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from agent.wsr import beam as B   # noqa: E402
from agent.wsr import tasks as T  # noqa: E402

HERE = pathlib.Path(__file__).parent


def load() -> list[dict]:
    return json.loads((HERE / "days.json").read_text())


def build(entry: dict):
    """The layer's own `Day` and its tasks, from one corpus entry."""
    grid = [(tuple(cell), tuple(ops), entity) for cell, ops, entity in entry["chains"]]
    available = {good: int(count) for good, count in entry["available"].items()}
    hire_times = tuple(entry["hire_times"]) or (1,) * entry["hands"]
    day = B.Day(chains=tuple(grid), available=available, hire_times=hire_times)
    return day, T.build(grid, available=available)


def search(entry: dict, beam: int | None, budget: float, hands: int | None):
    day, tasks = build(entry)
    if hands is None:
        hands = max(entry["hands"] - 1, 0)
    started = time.perf_counter()
    result = B.search(day, tasks, hands=hands, max_hands=entry["hands"],
                      beam=beam, budget_s=budget)
    return result, tasks, time.perf_counter() - started


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--beam", type=int, default=None, help="override the layer's own width")
    parser.add_argument("--budget", type=float, default=20.0, help="seconds per day")
    parser.add_argument("--days", type=int, default=0, help="only the first N days (0 = all)")
    args = parser.parse_args()

    days = load()
    if args.days:
        days = days[: args.days]

    print(f"{'day':>4} {'quad':>4} {'ops':>4} {'hands':>5} {'pool':>4} "
          f"{'placed':>9} {'secs':>7} {'beam':>5} {'budget?':>8}")
    totals = {"as is": [0.0, 0, 0], "override": [0.0, 0, 0]}
    by_width: dict[int, list[float]] = {}

    for entry in days:
        row = []
        for tag, beam in (("as is", None), ("override", args.beam)):
            if tag == "override" and args.beam is None:
                continue
            result, tasks, took = search(entry, beam, args.budget, None)
            totals[tag][0] += took
            totals[tag][1] += len(result.route)
            totals[tag][2] += tasks.n
            width = entry["hands"]
            if tag == "as is":
                by_width.setdefault(len(entry["quadrants"]), []).append(took)
                shown = (entry["day"], len(entry["quadrants"]), entry["ops"], entry["hands"],
                         result.pool, f"{len(result.route)}/{tasks.n}", took,
                         B.beam_for(tasks, entry["hands"]), result.out_of_time)
                row = shown
        if row:
            print(f"{row[0]:>4} {row[1]:>4} {row[2]:>4} {row[3]:>5} {row[4]:>4} "
                  f"{row[5]:>9} {row[6]:>6.1f}s {row[7]:>5} {str(row[8]):>8}")

    print()
    for tag, (took, placed, asked) in totals.items():
        print(f"{tag:9s}: {placed}/{asked} placed in {took:.1f}s")
    for quadrants, times in sorted(by_width.items()):
        print(f"{quadrants} quadrant(s): {len(times)} days, {sum(times):.1f}s total, "
              f"{sum(times) / len(times):.1f}s mean")


if __name__ == "__main__":
    main()
