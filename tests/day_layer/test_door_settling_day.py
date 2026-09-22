"""A real day whose door pass does not settle: what an unsettled pass has to hand back.

The hands land on the doors that are free when they are hired (F040), so the doors are a property of
the route and the search settles them by re-pricing until they agree. On one day of the archive they
never agree: the two configurations are each other's derivation, and the pass runs out of attempts.

What the answer carries then is not the doors the last attempt went in with - those describe the other
configuration - but the doors the ENGINE gives the route that is actually being written, which is the
one derivation the day itself agrees with. The compiler writes the day from what the answer carries, so
this is the difference between a plan the engine plays and a plan it silently scrambles (F047).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agent.wsr import beam as B
from agent.wsr import tasks as T
from agent.wsr.emit import check_route, compile_route

#: The day the archive has that the door pass cannot settle. Measured: the pass uses its three
#: attempts and the derivation still differs, and every hand of it is hired at the same hour, so the
#: doors are a permutation of the same four tiles.
OSCILLATING = ("2026-09-16", 109468286, 1)

REAL_DAYS = json.loads((Path(__file__).parent / "corpus" / "real_days.json").read_text())


def _day(entry):
    grid = [(tuple(cell), tuple(ops), entity) for cell, ops, entity in entry["chains"]]
    available = {good: int(hour) for good, hour in entry["available"].items()}
    hire_times = tuple(entry["hire_times"]) or (1,) * entry["hands"]
    day = B.Day(chains=tuple(grid), available=available, hire_times=hire_times)
    return day, T.build(grid, available=available)


def _entry():
    return next(e for e in REAL_DAYS
                if (e["dump"], e["episode"], e["day"]) == OSCILLATING)


def test_the_day_is_the_one_whose_doors_do_not_settle() -> None:
    """The fixture has to be the day it claims: every hand offered at the same hour, and late."""
    entry = _entry()
    hire_times = tuple(entry["hire_times"])
    assert hire_times and len(set(hire_times)) == 1, (
        f"the hands of this day are hired at different hours: {hire_times}")
    assert max(hire_times) > 1, f"the hands of this day are all hired in the first turn: {hire_times}"


def test_an_unsettled_door_pass_hands_back_the_doors_the_day_has() -> None:
    """The answer's doors are the engine's own derivation of its route, not the guess it went in with.

    `_hand_doors` IS that derivation, so the check is that the answer agrees with it - and that the
    answer compiles against it, which is what the manager actually does with it.
    """
    entry = _entry()
    day, tasks = _day(entry)
    result = B.search(day, tasks, hands=max(entry["hands"] - 1, 0), max_hands=entry["hands"],
                      budget_s=20.0)

    assert result.doors, "the search never settled the doors"
    assert tuple(result.doors) == B._hand_doors(day, tasks, result, result.pool), (
        "the answer carries doors the day does not have")
    assert not check_route(day, tasks, result), "the answer breaks a rule the engine enforces"
    compile_route(day, tasks, result)
