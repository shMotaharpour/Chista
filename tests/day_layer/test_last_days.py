"""The last day of a season, from the games our own seat won: the days that end in drops.

The corpus in `real_days.json` is a spread over the season; this is the other end of it. A season's last
day is where the harvests are - 16 to 46 of them - so the layer has to build its own drops and bank the
right batches, and it comes with ten to twelve hands, some hired at hour 2, which is where the per-hand
door rule lives. Every day here is one the game WON, so the bar is the archive's best play.

The question this answers is whether the engine is responsive and trustworthy where it matters most: a
day with drops, a late hire, and no slack to spare.
"""

from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agent.wsr import beam as B
from agent.wsr import tasks as T
from agent.wsr.emit import check_route

LAST_DAYS = json.loads((Path(__file__).parent / "corpus" / "last_days.json").read_text())


def _pair(task_id: str) -> tuple[int, str]:
    """A task id back to the tile it works and the op it runs: `d24_place2` is the second PLACE."""
    tile, rest = task_id.split("_", 1)
    return int(tile[1:]), rest.rstrip("0123456789")


def test_the_corpus_is_the_winners_last_days() -> None:
    """Every day is a win, and every day has the harvests that make it a day with drops."""
    assert len(LAST_DAYS) >= 8, f"too few days to say anything: {len(LAST_DAYS)}"
    assert all(e["day"] == 29 for e in LAST_DAYS), "these are the last days of their seasons"
    assert all(e["score"][0] > e["score"][1] for e in LAST_DAYS), "a day our seat lost is not the bar"
    harvests = [sum(1 for _c, ops, _e in e["chains"] for op in ops if op == "HARVEST")
                for e in LAST_DAYS]
    assert min(harvests) >= 10, f"a day with nothing to drop is not this corpus: {harvests}"
    late = [e for e in LAST_DAYS if max(e["hire_times"]) > 1]
    assert late, "no day hires a hand after hour 1, which is the half of the day this is for"


@pytest.mark.parametrize(
    "entry", LAST_DAYS,
    ids=[f"{e['dump']}-{e['episode']}-d{e['day']}" for e in LAST_DAYS])
def test_a_winner_day_is_carried_as_the_game_carried_it(entry) -> None:
    """The land conversion is the game's, the pool is no larger than the game paid, and it compiles."""
    grid = [(tuple(cell), tuple(ops), entity) for cell, ops, entity in entry["chains"]]
    available = {good: int(hour) for good, hour in entry["available"].items()}
    tasks = T.build(grid, available=available)
    hire_times = tuple(entry["hire_times"]) or (1,) * entry["hands"]
    result = B.search(
        B.Day(chains=tuple(grid), available=available, hire_times=hire_times),
        tasks,
        hands=max(entry["hands"] - 1, 0),
        max_hands=entry["hands"],
        budget_s=20.0,
    )

    assert result.complete, (
        f"{len(result.route)} of {tasks.n} tasks with the {entry['hands']} hands the game used")
    assert result.pool <= entry["hands"], (
        f"the layer chose {result.pool} hands where the game paid {entry['hands']}")
    assert not check_route(B.Day(chains=tuple(grid), available=available, hire_times=hire_times),
                           tasks, result), "the route breaks a rule the engine enforces"

    placed = collections.Counter(_pair(task_id) for _turn, task_id, _worker in result.route
                                 if not task_id.endswith("_drop"))
    asked = collections.Counter((index, op.lower())
                                for index, (_cell, ops, _entity) in enumerate(grid) for op in ops)
    assert placed == asked, (
        f"placed {sum(placed.values())} of the day's {sum(asked.values())} tile ops; "
        f"missing {(asked - placed).most_common(3)}, extra {(placed - asked).most_common(3)}")
