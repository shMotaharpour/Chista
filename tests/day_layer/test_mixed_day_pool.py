"""The mixed day with fewer hands than it needs: what the pool decides, and where the partial route stops.

The same day as `test_mixed_day.py` - a cow, a sheep and a goose on the shed's column, wheat across
the rest of the quadrant, every item in the shed from hour one - with two hands offered instead of
five, and the three-hand day beside it. The pool is what decides whether the day is carried, so this
is where the boundary between the two answers is pinned:

    two hands do not carry the day, and the search says so: a partial route, `complete=False`
    three hands do carry it, so the hand the caller withheld is what the day was short of
    the search finds the three-hand day on its own, and a hand-built reference agrees it exists
    the shortfall is the POOL and not the clock: not `out_of_time`, not `can_improve`, not `infeasible`
    the partial route is still a legal route, which is what the caller is handed
    the partial route replays as the day it was priced as - on the engine's own counters
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

#: The short pool. Measured at beam 64: two hands place 43 of 54, three carry all 54.
HANDS = 2
#: The smallest pool that carries the day, and the pool the hand-built reference route is written for.
REFERENCE_HANDS = HANDS + 1
#: The same market, with the hands the caller is willing to pay for.
ORDERS = [order for order in FIVE_HAND_ORDERS if order[0] != "HIRE"] + [["HIRE"]] * HANDS


def _day(hands: int = HANDS):
    """The day as the search sees it, with the pool the caller is willing to pay for."""
    chains = [(cell, chain_ops(chain_id_of(ops)), entity) for cell, ops, entity in TILES]
    tasks = T.build(chains, available=AVAILABLE)
    day = B.Day(chains=tuple(chains), available=AVAILABLE, hire_times=(1,) * hands)
    return day, tasks


@pytest.fixture(scope="module")
def short():
    """The short-pool search: the day it was given, and the answer it gives back."""
    day, tasks = _day()
    result = B.search(day, tasks, beam=64, hands=HANDS, max_hands=HANDS)
    return day, tasks, result


@pytest.fixture(scope="module")
def late():
    """The same day with the third hand offered at hour three: the doors are decided twice."""
    chains = [(cell, chain_ops(chain_id_of(ops)), entity) for cell, ops, entity in TILES]
    tasks = T.build(chains, available=AVAILABLE)
    day = B.Day(chains=tuple(chains), available=AVAILABLE, hire_times=(1, 1, 3))
    return day, tasks, B.search(day, tasks, beam=64, hands=3, max_hands=3)


def test_three_hands_do_not_carry_the_day(short) -> None:
    """The pool is short and the search says so, with a route rather than with silence.

    `complete=False` is the honest answer about the pool; an empty route would be the answer that
    threw the work away, and the two are different decisions for the caller.
    """
    _day_, tasks, result = short
    assert not result.complete, f"two hands carried the day after all: {len(result.route)} tasks"
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


def test_the_partial_route_replays_as_the_day_it_was_priced_as(short) -> None:
    """The day the caller keeps is the day the search priced - asserted on the engine's counters.

    The plantings the route planned have to land, and the animal tiles have to hold their species:
    the engine refuses a misplaced op in silence (F047), so this is the only place the difference
    between a day that was carried out and a day that was merely written down shows up. It was the
    mark that used to sit here: the search priced the day from doors the compiler then did not write,
    and the board came back with the animals unplaced and most of the wheat bare.
    """
    day, tasks, result = short
    plan = to_plan(compile_route(day, tasks, result))
    board = _board(_replay(plan, ORDERS, weeds=False))

    # What a planting leaves on the board: the crop, or a weed when the day never reached its water -
    # an unwatered planting is what the engine turns to weed. Either is the plant op having landed; a
    # bare tile is one that did not. The night's random weeds are switched off, so no other tile grows.
    planned = {tuple(int(v) for v in tasks.cells[tasks.ids.index(task_id)])
               for _hour, task_id, _worker in result.route if task_id.endswith("_plant")}
    grown = {cell for cell, record in board.items() if record.get("kind") in ("PLANT", "WEED")}
    assert grown == planned, (
        f"the partial route planned {len(planned)} plantings and the board holds {len(grown)} tiles "
        f"that grew something: {sorted(planned - grown)} planned but bare, "
        f"{sorted(grown - planned)} grown but never planned")

    for cell, _ops, entity in ANIMAL_TILES:
        assert board.get(cell, {}).get("animal") == entity, (
            f"{cell} was built for {entity} and holds {board.get(cell, {}).get('animal')!r}")


def test_a_hand_hired_late_is_priced_from_where_the_field_stands_then(late) -> None:
    """A hand hired at hour three lands from where the field stands at hour three (F040).

    The engine settles each hire at its own moment, so a unit that walks off its door between the
    first turn and a late hire leaves that door for the late hand. Pricing every hand from the first
    turn's snapshot puts it a door out, and the engine - which refuses a misplaced op in silence -
    then plays a day the search never priced.
    """
    day, tasks, result = late
    first_turn = tuple(tuple(int(v) for v in cell)
                       for cell in B._start_positions(day, result.pool, result.settled)[len(day.units):])
    assert result.doors, "the search never settled the doors"
    assert first_turn != result.doors, (
        "this day does not tell the two moments apart, so it cannot pin the rule")
    assert result.doors == B._hand_doors(day, tasks, result, result.pool)


def test_the_spare_leaves_the_wait_for_the_goods_as_room() -> None:
    """A worker that begins before the shed opens spends its pickups, not the wait for them.

    The compiler writes one PICKUP per good at `max(hour, arrival)` and every turn before that is a
    PASS - room a manager may lay work into. Charging the wait as spent under-reports the room: on the
    mixed day the farmer begins at hour 0 and the shed opens at hour 1, and the search reported three
    turns where the compiled day leaves four.
    """
    chains = [(cell, chain_ops(chain_id_of(ops)), entity) for cell, ops, entity in ANIMAL_TILES]
    tasks = T.build(chains, available=AVAILABLE)
    day = B.Day(chains=tuple(chains), available=AVAILABLE, hire_times=())
    place = next(task_id for task_id in tasks.ids if task_id.endswith("_place"))
    result = B.Result(pool=0, route=[(2, place, 0)], complete=False)

    ops = compile_route(day, tasks, result)
    hours = B._start_hours(day, result.pool)
    passes = [sum(1 for op in row[int(hours[worker]):] if op == ("PASS",))
              for worker, row in enumerate(ops.units)]
    assert B.remaining_turns(day, tasks, result) == passes


#: A three-hand day for this fixture: built by hand, run on the engine, and read back off its replay.
#: Every op lands and the three animals are housed, so it is a day the day itself allows - and the
#: search does not find it (measured: 53 of 54 on its own, 54 of 54 when warmed with this).
REFERENCE_ROUTE = [
    (0, 'd0_build_pasture', 0),
    (3, 'd24_plant', 1),
    (4, 'd23_plant', 2),
    (4, 'd24_water', 1),
    (5, 'd0_place', 0),
    (5, 'd23_water', 2),
    (6, 'd20_plant', 1),
    (6, 'd22_plant', 3),
    (7, 'd19_plant', 2),
    (7, 'd1_build_coop', 0),
    (7, 'd20_water', 1),
    (7, 'd22_water', 3),
    (8, 'd19_water', 2),
    (8, 'd1_place', 0),
    (9, 'd16_plant', 1),
    (9, 'd18_plant', 3),
    (9, 'd1_feed', 0),
    (10, 'd15_plant', 2),
    (10, 'd16_water', 1),
    (10, 'd18_water', 3),
    (10, 'd1_care', 0),
    (11, 'd15_water', 2),
    (12, 'd11_plant', 1),
    (12, 'd14_plant', 3),
    (12, 'd2_build_pasture', 0),
    (13, 'd10_plant', 2),
    (13, 'd11_water', 1),
    (13, 'd14_water', 3),
    (13, 'd2_place', 0),
    (14, 'd10_water', 2),
    (14, 'd2_feed', 0),
    (15, 'd13_plant', 3),
    (15, 'd2_care', 0),
    (15, 'd6_plant', 1),
    (16, 'd13_water', 3),
    (16, 'd6_water', 1),
    (16, 'd9_plant', 2),
    (17, 'd12_plant', 0),
    (17, 'd9_water', 2),
    (18, 'd12_water', 0),
    (18, 'd17_plant', 3),
    (18, 'd5_plant', 1),
    (19, 'd17_water', 3),
    (19, 'd5_water', 1),
    (19, 'd8_plant', 2),
    (20, 'd7_plant', 0),
    (20, 'd8_water', 2),
    (21, 'd21_plant', 3),
    (21, 'd4_plant', 1),
    (21, 'd7_water', 0),
    (22, 'd21_water', 3),
    (22, 'd3_plant', 2),
    (22, 'd4_water', 1),
    (23, 'd3_water', 2),
]


def test_the_reference_three_hand_day_is_a_day_the_rules_allow() -> None:
    """The hand-built three-hand day is legal, and the search takes it when it is handed over.

    Two separate facts: `check_route` says no rule is broken, and a search warmed with the route comes
    back carrying all 54 - which is what makes the day's own allowance the answer and the search the
    thing that is short.
    """
    # The reference is the THREE-hand day: its route names workers 0..3. Building it with the
    # fixture's default (2 hands) makes `check_route` report a worker the day does not have.
    day, tasks = _day(hands=HANDS + 1)
    route = [(turn, task_id, worker) for turn, task_id, worker in REFERENCE_ROUTE]
    assert len(route) == tasks.n, f"the reference covers {len(route)} of the day's {tasks.n} tasks"
    reference = B.Result(pool=HANDS + 1, route=route, complete=True)

    assert not check_route(day, tasks, reference), "the reference breaks a rule the engine enforces"
    warmed = B.search(day, tasks, hands=HANDS + 1, max_hands=HANDS + 1, warm=reference)
    assert warmed.complete, (
        f"warmed with the reference the search placed {len(warmed.route)} of {tasks.n}")


def test_the_search_finds_the_three_hand_day_by_itself() -> None:
    """Three hands are enough for this day - the reference proves it - so the search has to find it."""
    day, tasks = _day(hands=REFERENCE_HANDS)
    result = B.search(day, tasks, beam=64, hands=REFERENCE_HANDS, max_hands=REFERENCE_HANDS)
    assert result.complete, f"the search placed {len(result.route)} of {tasks.n}"


def test_the_remaining_capacity_is_the_day_the_compiler_wrote(short) -> None:
    """The spare the search reports, against the PASS turns in the compiled day.

    The two are counted from opposite sides - the search from the route it placed, the compiler from
    the ops it writes - so this is where the accounting is checked rather than assumed. A manager
    reads `spare` to decide whether to lay more work on the same hands, and a number that does not
    match the day it will actually dispatch is worse than no number at all.
    """
    day, tasks, result = short
    ops = compile_route(day, tasks, result)
    # A row is the whole horizon long, so a hand's row carries the turns before its own hour as PASS
    # too - turn 0 is not part of a day that begins at hour 1. The comparison starts where the day
    # does.
    hours = B._start_hours(day, result.pool)
    idle = [sum(1 for op in row[int(hours[w]):] if op == ("PASS",))
            for w, row in enumerate(ops.units)]
    assert B.remaining_turns(day, tasks, result) == idle, (
        "the search's remaining capacity and the compiler's PASS turns disagree: "
        f"{B.remaining_turns(day, tasks, result)} against {idle}")
    assert result.spare == sum(idle), (
        f"the answer reports {result.spare} unspent turns and the day holds {sum(idle)}")


def test_one_more_hand_leaves_more_of_the_day_unspent(short) -> None:
    """The pool, read from the other side: a hand more is a day with more room left in it.

    Without this the spare could be a constant and the first test would still pass.
    """
    _day_, _tasks, result = short
    day, tasks = _day(hands=HANDS + 1)
    bigger = B.search(day, tasks, beam=64, hands=HANDS + 1, max_hands=HANDS + 1)
    assert bigger.spare > result.spare, (
        f"{HANDS + 1} hands leave {bigger.spare} turns unspent, {HANDS} leave {result.spare}")
