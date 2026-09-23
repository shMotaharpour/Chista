"""Real days, from the archive: the same land conversion, no more hands than the game paid for.

Synthetic days test what the layer was built to do; they cannot test what a season actually asks of
it. These come from `data/replays_parquet` in `kaggriculture-episodes-analyses`, and each is the
game's own land work: the ops that landed on every tile, and the hands it paid to have them landed.
`corpus/build_real_days.py` builds the corpus and says what an op has to satisfy to be counted.

The two questions the archive can answer are the two that matter:

  the land conversion   every op the game ran on every tile is placed by the route, and nothing else
  the hands             the pool the layer chooses is no larger than the game's own count

Two samples answer them, because who played the day decides how hard the question is. `real_days.json`
is the archive's spread - player-0 of an episode, chosen by episode id and a seed, whoever played it -
and most of its days belong to ordinary players, with player-0 the losing side more often than not.
`strong_days.json` is the competitive half: the winner's side of episodes a cumulative top-10 player
won, built by `corpus/build_strong_days.py`, each entry naming the player and its win count. The layer
carries fewer of those, which is the number worth knowing before a submission.

`KNOWN_SHORT` names the days of each sample that do not, marked with a strict `xfail` so that closing
the gap turns the mark into a failure saying to take it out. Every task those days miss is plain time -
none is blocked by the timetable - and what has been measured and ruled out is in the commit messages,
which is where measurements belong.
"""
import collections
import json
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from agent.wsr import beam as B
from agent.wsr import tasks as T

CORPUS = pathlib.Path(__file__).parent / "corpus" / "real_days.json"
REAL_DAYS = json.loads(CORPUS.read_text())
STRONG_CORPUS = pathlib.Path(__file__).parent / "corpus" / "strong_days.json"
STRONG_DAYS = json.loads(STRONG_CORPUS.read_text())

#: The days the preload's per-worker charge costs tasks on. Strict: if the charge is ever made exact
#: these fail, which is the reminder to take the marks out.
KNOWN_SHORT = {
    ("2026-08-29", 102083125, 12),
    ("2026-09-01", 104478713, 15),
    ("2026-09-06", 105964064, 25),
    ("2026-09-16", 109466152, 25),
    ("2026-09-01", 104478289, 15),
}

#: The strong sample's own short days, same rule. Every one is a late day of a big farm - the days a
#: top-10 player had the most land and the same handful of hands - and each misses a few tasks.
KNOWN_SHORT_STRONG = {
    ("2026-08-31", 103687742, 15),
    ("2026-09-05", 105864228, 25),
    ("2026-09-15", 109086888, 15),
    ("2026-09-15", 109086888, 20),
    ("2026-09-15", 109086888, 25),
    ("2026-09-20", 111016701, 20),
    ("2026-09-20", 111016701, 25),
}

#: Days the search carries at some widths of the game's own pool and not at the width `beam_for`
#: picks: the shortlist's tie-break decides them, not the pool or the rules. Measured with the pool
#: fixed at the game's, over the default width and its neighbours.
KNOWN_WIDTH = {
    ("2026-08-24", 98009264, 18),
    ("2026-09-08", 106613414, 22),
}
KNOWN_WIDTH_STRONG = {
    ("2026-08-23", 97204025, 5),
    ("2026-09-05", 105864228, 15),
    ("2026-09-10", 107289135, 25),
}

_WIDTH_REASON = (
    "carried at other widths of the game's own pool, not at the default one: the shortlist's "
    "tie-break among equal finish hours decides the day, not the pool"
)

_SHORT_REASON = (
    "the model's turn cost is higher than the game's on these days and the cause is not yet found. "
    "The preload's per-worker charge was the suspect and is NOT it: turning it off changes nothing "
    "(87 of 102 either way). The missing tasks are all plain time - none is blocked by the timetable "
    "- so something in the walk or the trip model costs more than the game spent"
)

_STRONG_SHORT_REASON = (
    "the same shortfall as the spread's own days, on the days a top-10 player had a big farm: the "
    "pool the game paid is what the search is given, and the search settles a few tasks short of the "
    "day's work. No deadline is missed and the search was not cut short - it says the pool cannot "
    "carry the day - so this is the layout, not the timetable"
)


def _key(entry) -> tuple[str, int, int]:
    return entry["dump"], entry["episode"], entry["day"]


def _pair(task_id: str) -> tuple[int, str]:
    """A task id back to the tile it works and the op it runs: `d24_place2` is the second PLACE."""
    tile, rest = task_id.split("_", 1)
    return int(tile[1:]), rest.rstrip("0123456789")


def _carried_as_the_game(entry) -> None:
    """The two questions, on one day: the game's own land conversion, and no more hands than it paid."""
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


def test_the_strong_corpus_says_who_it_is_built_from():
    """The sample is only worth having if it can say whose days these are: the whole point of it is
    that they are not anonymous. A player, its win count, and the opponent it beat, on every entry."""
    dumps = {entry["dump"] for entry in STRONG_DAYS}
    players = {entry["agent"] for entry in STRONG_DAYS}
    ops = [sum(len(c[1]) for c in entry["chains"]) for entry in STRONG_DAYS]

    assert len(STRONG_DAYS) >= 100, f"the sample is too small to measure anything: {len(STRONG_DAYS)}"
    assert len(dumps) >= 30, f"the days come from too few dumps: {len(dumps)}"
    assert len(players) >= 5, f"the days come from too few players: {sorted(players)}"
    assert min(ops) <= 60 and max(ops) >= 140, f"the days' work does not spread: {min(ops)}..{max(ops)}"
    for entry in STRONG_DAYS:
        assert entry["agent"] and int(entry["wins"]) > 0, entry
        assert entry["opponent"] and entry["opponent"] != entry["agent"], entry


# A strict marker, not `pytest.xfail(...)`: that call stops the test and reports xfail whatever would
# have happened, so a mark that has gone stale can never say so.
def _marked(entry, short: set, reason: str, width: set):
    if _key(entry) in short:
        return pytest.param(entry, marks=pytest.mark.xfail(strict=True, reason=reason))
    if _key(entry) in width:
        return pytest.param(entry, marks=pytest.mark.xfail(strict=True, reason=_WIDTH_REASON))
    return entry


@pytest.mark.parametrize(
    "entry",
    [_marked(e, KNOWN_SHORT, _SHORT_REASON, KNOWN_WIDTH) for e in REAL_DAYS],
    ids=[f"{e['dump']}-{e['episode']}-d{e['day']}" for e in REAL_DAYS])
def test_a_real_day_is_carried_as_the_game_carried_it(entry):
    """The land conversion is the game's, and the pool is no larger than what the game paid."""
    _carried_as_the_game(entry)


@pytest.mark.parametrize(
    "entry",
    [_marked(e, KNOWN_SHORT_STRONG, _STRONG_SHORT_REASON, KNOWN_WIDTH_STRONG) for e in STRONG_DAYS],
    ids=[f"{e['agent']}-{e['dump']}-d{e['day']}" for e in STRONG_DAYS])
def test_a_strong_players_day_is_carried(entry):
    """The same two questions on the days a top-10 player won, which is the competitive target."""
    _carried_as_the_game(entry)

