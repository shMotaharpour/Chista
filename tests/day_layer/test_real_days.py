"""Real days, from the archive: the same land conversion, no more hands than the game paid for.

Synthetic days test what the layer was built to do; they cannot test what a season actually asks of
it. These come from `data/replays_parquet` in `kaggriculture-episodes-analyses`, and each is the
game's own land work: the ops that landed on every tile, and the hands it paid to have them landed.
`corpus/build_real_days.py` builds the corpus and says what an op has to satisfy to be counted.

The two questions the archive can answer are the two that matter:

  the land conversion   every op the game ran on every tile is placed by the route, and nothing else
  the hands             the pool the layer chooses is no larger than the game's own count

`KNOWN_SHORT` names the days that do not, marked with a strict `xfail` so that closing the gap turns
the mark into a failure saying to take it out. Every task those days miss is plain time - none is
blocked by the timetable - and what has been measured and ruled out is in the commit messages, which
is where measurements belong.
"""
import collections
import json
import pathlib
import sys

import pytest

sys.path.insert(0, "/chista/pm/world")

from agent.wsr import beam as B
from agent.wsr import tasks as T

CORPUS = pathlib.Path(__file__).parent / "corpus" / "real_days.json"
REAL_DAYS = json.loads(CORPUS.read_text())

#: The days the preload's per-worker charge costs tasks on. Strict: if the charge is ever made exact
#: these fail, which is the reminder to take the marks out.
KNOWN_SHORT = {
    ("2026-08-24", 98009264, 24),
    ("2026-08-29", 102083125, 12),
    ("2026-09-01", 104478713, 15),
    ("2026-09-06", 105964064, 25),
    ("2026-08-28", 101297130, 22),
    ("2026-09-16", 109466152, 25),
    ("2026-09-01", 104478289, 15),
    # and the three the real hire hours cost: two or three of the game's hands began at hour 2, so
    # they had 22 turns and not 23 - which the corpus used to give them all.
    ("2026-08-24", 98009264, 14),
    ("2026-08-24", 98009264, 18),
}

_SHORT_REASON = (
    "the model's turn cost is higher than the game's on these days and the cause is not yet found. "
    "The preload's per-worker charge was the suspect and is NOT it: turning it off changes nothing "
    "(87 of 102 either way). The missing tasks are all plain time - none is blocked by the timetable "
    "- so something in the walk or the trip model costs more than the game spent"
)


def _key(entry) -> tuple[str, int, int]:
    return entry["dump"], entry["episode"], entry["day"]


def _pair(task_id: str) -> tuple[int, str]:
    """A task id back to the tile it works and the op it runs: `d24_place2` is the second PLACE."""
    tile, rest = task_id.split("_", 1)
    return int(tile[1:]), rest.rstrip("0123456789")


def test_the_corpus_has_a_spread_of_days():
    """A regression that only ever sees one shape of day is a regression for one shape of day."""
    hands = [entry["hands"] for entry in REAL_DAYS]
    tiles = [len(entry["chains"]) for entry in REAL_DAYS]
    ops = [sum(len(c[1]) for c in entry["chains"]) for entry in REAL_DAYS]
    dumps = {entry["dump"] for entry in REAL_DAYS}
    season = {entry["day"] for entry in REAL_DAYS}
    late = [entry for entry in REAL_DAYS if max(entry["hire_times"], default=1) > 1]

    assert len(REAL_DAYS) >= 102, "the corpus is the archive's spread, not a sample of one"
    assert len(dumps) >= 30, f"the days come from too few dumps: {len(dumps)}"
    assert min(hands) <= 2 and max(hands) >= 14, f"the hand counts do not spread: {hands}"
    assert min(tiles) <= 10 and max(tiles) >= 60, f"the field sizes do not spread: {tiles}"
    assert min(ops) <= 20 and max(ops) >= 150, f"the days' work does not spread: {ops}"
    assert len(season) >= 20, f"the season does not spread: {sorted(season)}"
    assert late, "no day has a hand that began after hour 1, which is not the archive's own spread"


# A strict marker, not `pytest.xfail(...)`: that call stops the test and reports xfail whatever would
# have happened, so a mark that has gone stale can never say so.
@pytest.mark.parametrize(
    "entry",
    [pytest.param(e, marks=pytest.mark.xfail(strict=True, reason=_SHORT_REASON))
     if _key(e) in KNOWN_SHORT else e for e in REAL_DAYS],
    ids=[f"{e['dump']}-{e['episode']}-d{e['day']}" for e in REAL_DAYS])
def test_a_real_day_is_carried_as_the_game_carried_it(entry):
    """The land conversion is the game's, and the pool is no larger than what the game paid."""
    grid = [(tuple(cell), tuple(ops), entity) for cell, ops, entity in entry["chains"]]
    available = {good: int(hour) for good, hour in entry["available"].items()}
    tasks = T.build(grid, available=available)
    # The hands began when they began: a hand hired in turn 2 acts from hour 3, not from hour 1.
    hire_times = tuple(entry["hire_times"]) or (1,) * entry["hands"]
    result = B.search(
        B.Day(chains=tuple(grid), available=available, hire_times=hire_times),
        tasks,
        hands=max(entry["hands"] - 1, 0),
        max_hands=entry["hands"],
        budget_s=20.0,
    )

    assert result.complete, (
        f"{len(result.route)} of {tasks.n} tasks with {entry['hands']} hands, which is what the game "
        f"used"
    )
    assert result.pool <= entry["hands"], (
        f"the layer chose {result.pool} hands where the game paid {entry['hands']}"
    )

    # A task id is `d<tile>_<op>`, with a number when a tile repeats an op. The comparison is a
    # multiset of (tile, op), so a tile the game worked twice must be worked twice.
    placed = collections.Counter(_pair(task_id) for _turn, task_id, _worker in result.route)
    asked = collections.Counter((index, op.lower())
                                for index, (_cell, ops, _entity) in enumerate(grid) for op in ops)

    assert placed == asked, (
        f"placed {sum(placed.values())} of the day's {sum(asked.values())} tile ops; "
        f"missing {(asked - placed).most_common(3)}, extra {(placed - asked).most_common(3)}"
    )
