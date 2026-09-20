"""Real days, from the archive: the same land conversion, no more hands than the game paid for.

Synthetic days test what the layer was built to do; they cannot test what a season actually asks of
it. These twelve come from `data/replays_parquet/2026-09-16` in `kaggriculture-episodes-analyses` -
three episodes, four days each - and each is the game's own land work: the ops that landed on every
tile, and the hands it paid to have them landed. The fixture is sixteen kilobytes; the corpus stays
where it is.

The two questions the archive can answer are the two that matter:

  the land conversion   every op the game ran on every tile is placed by the route, and nothing else
  the hands             the pool the layer chooses is no larger than the game's own count

Measured when this landed: twelve of twelve carried whole, at one hand FEWER than the game on every
one of them, in 71 to 224 ms a day.

What the fixture does not carry is the argument of an op: `PLANT` arrives without its crop, so the
layer plans the tile work without pricing a seed for it. That is a limit of this extraction and not
of the layer, and it is the next thing to put in the fixture.
"""
import collections
import json
import pathlib
import sys

sys.path.insert(0, "/chista/pm/world")

from agent.wsr import beam as B
from agent.wsr import tasks as T

FIXTURE = pathlib.Path(__file__).parent / "fixtures" / "real_days.json"
REAL_DAYS = json.loads(FIXTURE.read_text())


def _day(entry):
    """The chains as the layer takes them: `(cell, ops, entity)` per tile, in the game's order."""
    grid = [(tuple(cell), tuple(ops), None) for cell, ops in entry["chains"]]
    available = {good: int(hour) for good, hour in entry["available"].items()}
    return grid, available


def _pair(task_id: str) -> tuple[int, str]:
    """A task id back to the tile it works and the op it runs."""
    tile, rest = task_id.split("_", 1)
    return int(tile[1:]), rest.rstrip("0123456789")

def _label(entry):
    return f"episode {entry['episode']} day {entry['day']}"


def test_the_fixture_has_a_spread_of_days():
    """A regression that only ever sees one shape of day is a regression for one shape of day."""
    hands = [entry["hands"] for entry in REAL_DAYS]
    tiles = [len(entry["chains"]) for entry in REAL_DAYS]

    assert len(REAL_DAYS) >= 12, "the fixture is the archive's spread, not a sample of one"
    assert min(hands) <= 5 and max(hands) >= 10, f"the hand counts do not spread: {hands}"
    assert min(tiles) <= 25 and max(tiles) >= 50, f"the field sizes do not spread: {tiles}"


def test_every_real_day_is_carried_with_no_more_hands_than_the_game():
    """The land conversion is the game's, and the pool must not exceed what the game paid."""
    for entry in REAL_DAYS:
        grid, available = _day(entry)
        tasks = T.build(grid, available=available)
        result = B.search(
            B.Day(chains=tuple(grid), available=available, hire_times=(1,) * max(entry["hands"], 1)),
            tasks,
            hands=max(entry["hands"] - 1, 0),
            max_hands=entry["hands"],
            budget_s=20.0,
        )

        assert result.complete, (
            f"{_label(entry)}: {len(result.route)} of {tasks.n} tasks with "
            f"{entry['hands']} hands, which is what the game used"
        )
        assert result.pool <= entry["hands"], (
            f"{_label(entry)}: the layer chose {result.pool} hands where the game paid "
            f"{entry['hands']}"
        )


def test_the_land_conversion_is_the_games_own():
    """Every op the game ran on every tile is placed, and no tile is worked that the game left."""
    for entry in REAL_DAYS:
        grid, available = _day(entry)
        tasks = T.build(grid, available=available)
        result = B.search(
            B.Day(chains=tuple(grid), available=available, hire_times=(1,) * max(entry["hands"], 1)),
            tasks,
            hands=max(entry["hands"] - 1, 0),
            max_hands=entry["hands"],
            budget_s=20.0,
        )

        # A task id is `d<tile>_<op>`, with a number when a tile repeats an op - `d24_place2` is the
        # second PLACE on tile 24. The comparison is a multiset of (tile, op), so a tile the game
        # worked twice must be worked twice and a tile it left alone must be left alone.
        placed = collections.Counter(_pair(task_id) for _turn, task_id, _worker in result.route)
        asked = collections.Counter((index, op.lower())
                                    for index, (_cell, ops, _entity) in enumerate(grid) for op in ops)

        assert placed == asked, (
            f"{_label(entry)}: placed {sum(placed.values())} of the day's {sum(asked.values())} "
            f"tile ops; missing {(asked - placed).most_common(3)}, "
            f"extra {(placed - asked).most_common(3)}"
        )
