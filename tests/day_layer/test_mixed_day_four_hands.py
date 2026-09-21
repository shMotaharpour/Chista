"""The mixed day with one hand fewer: what the pool decides, and where the partial route stops.

The same day as `test_mixed_day.py` - a cow, a sheep and a goose on the shed's column, wheat across
the rest of the quadrant, every item in the shed from hour one - with four hands offered instead of
five. One hand is the difference between a day the search carries and a day it does not, so this is
where the pool's answer is pinned.

What is asserted is the answer the layer gives when the pool is short:

    four hands do not carry the day, and the search says so: a partial route, `complete=False`
    the shortfall is the POOL and not the clock: not `out_of_time`, not `can_improve`, not `infeasible`
    five hands carry the same day, so the hand the caller withheld is what the day was short of
    the partial route is still a legal route, which is what the caller is handed

One thing does not hold yet, and it is marked rather than hidden: the partial route, compiled and
replayed, is not the day it was priced as. The mark's reason names what was measured.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agent.tile_dp.chains import chain_id_of, chain_ops
from agent.wsr import beam as B
from agent.wsr import tasks as T
from agent.wsr.emit import check_route, compile_route, to_plan

#: The day, its timetable and its market are the five-hand test's - only the pool changes.
from tests.day_layer.test_mixed_day import (ANIMAL_TILES, AVAILABLE, ORDERS as FIVE_HAND_ORDERS,
                                            TILES, _board, _replay)

HANDS = 4
#: The same market, with the hands the caller is willing to pay for.
ORDERS = [order for order in FIVE_HAND_ORDERS if order[0] != "HIRE"] + [["HIRE"]] * HANDS

_SETTLED_REASON = (
    "the compiled day is not the day the search priced. `compile_route` writes a worker's first "
    "walk at its earliest free turn, so a first task with slack moves the worker in turn 0, while "
    "`_settled_after_first_turn` counts a turn-0 move only when the walk exactly fills the gap. On "
    "this day the farmer's first task is one tile away and six turns out, so the compiler writes "
    "WEST at turn 0 and the model has the farmer still on (4, 4) - which moves every hand's door by "
    "one, and the board comes back with the animals unplaced and most of the wheat bare"
)


def _day(hands: int = HANDS):
    """The day as the search sees it, with the pool the caller is willing to pay for."""
    chains = [(cell, chain_ops(chain_id_of(ops)), entity) for cell, ops, entity in TILES]
    tasks = T.build(chains, available=AVAILABLE)
    day = B.Day(chains=tuple(chains), available=AVAILABLE, hire_times=(1,) * hands)
    return day, tasks


@pytest.fixture(scope="module")
def short():
    """The four-hand search: the day it was given, and the answer it gives back."""
    day, tasks = _day()
    result = B.search(day, tasks, beam=64, hands=HANDS, max_hands=HANDS)
    return day, tasks, result


def test_four_hands_do_not_carry_the_day(short) -> None:
    """The pool is short and the search says so, with a route rather than with silence.

    `complete=False` is the honest answer about the pool; an empty route would be the answer that
    threw the work away, and the two are different decisions for the caller.
    """
    _day_, tasks, result = short
    assert not result.complete, f"four hands carried the day after all: {len(result.route)} tasks"
    assert 0 < len(result.route) < tasks.n, (
        f"the partial route is {len(result.route)} of {tasks.n} tasks - not partial, or not a route")
    assert result.pool == HANDS, f"the answer was searched with {result.pool} hands, not {HANDS}"


def test_the_shortfall_is_the_pool_and_not_the_clock(short) -> None:
    """Which of the three ways a day can come back short this one is.

    A deadline would mean more budget finds more, and an arithmetic ceiling would mean no allowed
    pool can carry the day at all. Neither is this: the pool the caller allowed is simply too small,
    which is what `can_improve=False` says in the name of the decision it feeds.
    """
    _day_, _tasks, result = short
    assert not result.out_of_time, "the search stopped at a deadline, not at the end of its work"
    assert not result.can_improve, "more budget would find more, so this is not the pool's answer"
    assert not result.infeasible, "the day was called impossible for any allowed pool"


def test_one_more_hand_carries_the_same_day() -> None:
    """The same day, one hand more - so the withheld hand is what the day was short of.

    Without this the first test would pass on a day no pool can carry, and the pool would not be
    what was measured.
    """
    day, tasks = _day(hands=HANDS + 1)
    result = B.search(day, tasks, beam=64, hands=HANDS + 1, max_hands=HANDS + 1)
    assert result.complete, (
        f"{HANDS + 1} hands did not carry the day either: {len(result.route)} of {tasks.n} tasks")
    assert result.pool == HANDS + 1, f"the answer was searched with {result.pool} hands"


def test_the_partial_route_is_still_a_legal_route(short) -> None:
    """What the caller is handed when the day is not carried: a route, and one the engine's own
    rules accept - `check_route` is the layer's promise about it, and it is worth asserting that the
    promise survives the shortfall rather than only holding on days that fit.
    """
    day, tasks, result = short
    complaints = check_route(day, tasks, result)
    assert not complaints, f"the partial route breaks a rule the engine enforces: {complaints}"


# A strict marker, not `pytest.xfail(...)`: that call reports xfail whatever would have happened, so
# a mark that has gone stale can never say so. This one fails loudly when the day is fixed.
@pytest.mark.xfail(strict=True, reason=_SETTLED_REASON)
def test_the_partial_route_replays_as_the_day_it_was_priced_as(short) -> None:
    """The day the caller keeps is the day the search priced - asserted on the engine's counters.

    The plantings the route planned have to land, and the animal tiles have to hold their species:
    the engine refuses a misplaced op in silence (F047), so this is the only place the difference
    between a day that was carried out and a day that was merely written down shows up.
    """
    day, tasks, result = short
    plan = to_plan(compile_route(day, tasks, result))
    board = _board(_replay(plan, ORDERS))

    planned = sum(1 for _hour, task_id, _worker in result.route if task_id.endswith("_plant"))
    planted = sum(1 for record in board.values() if record.get("kind") == "PLANT")
    assert planted == planned, (
        f"the partial route planned {planned} plantings and the board holds {planted}")

    for cell, _ops, entity in ANIMAL_TILES:
        assert board.get(cell, {}).get("animal") == entity, (
            f"{cell} was built for {entity} and holds {board.get(cell, {}).get('animal')!r}")
