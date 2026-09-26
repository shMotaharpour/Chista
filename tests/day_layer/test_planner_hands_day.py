"""The hands the planner offers are the hands the day layer searches with - and zero is an offer.

`planner/day.py` hands wsr the pool the master priced. The farmer is always on the field; `hands` is
only the hired ones, and each starts at the hour the engine allows (`rules.hire_hour`). A plan priced
with no hands that comes back walked by two workers is a day the market never pays for: the
dispatcher writes a second unit's ops and nobody is hired to run them.

The day: fourteen wheat tiles, PLANT+WATER, more than the farmer can walk alone.

What is asserted:

    fit(hands=0) searches the farmer alone: pool 0, and the route names worker 0 only
    compile(hands=0) writes one unit and hires nobody
    fit(hands=2) is still allowed its two hands
    Day(hands=k) starts its hands at the engine's own hours, ten a turn
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agent.planner import day as D
from agent.world.rules import MAX_MARKET_ORDERS_PER_TURN, hire_hour
from agent.wsr import beam as B

CHAINS = tuple(((x, y), ("PLANT", "WATER"), "WHEAT")
               for y in range(4) for x in range(4) if (x, y) != (3, 3))[:14]
AVAILABLE = {"WHEAT": 1}
OBS = {"player": 0, "hour": 0, "day": 0,
       "private": {"seeds": {"WHEAT": 20}, "shed": {}},
       "farms": [{"hires_today": 0}], "market": {"prices": {}}}


def test_the_premise_the_farmer_alone_cannot_carry_this_day() -> None:
    fitted = D.fit(CHAINS, hands=0, available=AVAILABLE)
    assert not fitted.complete, "the day fits the farmer alone, so it cannot show a hand appearing"


def test_fit_with_no_hands_searches_the_farmer_alone() -> None:
    fitted = D.fit(CHAINS, hands=0, available=AVAILABLE)
    assert fitted.pool == 0, (
        f"the master priced no hands and the day layer searched with {fitted.pool}: a hand the "
        f"market never hires")


def test_fit_is_still_allowed_the_hands_it_was_offered() -> None:
    fitted = D.fit(CHAINS, hands=2, available=AVAILABLE)
    assert 1 <= fitted.pool <= 2, f"two hands were offered and the search used {fitted.pool}"


def test_compile_with_no_hands_writes_one_unit_and_hires_nobody() -> None:
    fitted = D.DayFit(CHAINS, 0, 28, 0, True, 0.0, 0.0)
    plan = D.DayPlan(master=None, choices=[], mixes={}, day=fitted, rounds=1, overhead=1.0, hands=0)
    out = D.compile(plan, OBS, hands=0)
    assert len(out["units"]) == 1, f"a plan with no hands wrote {len(out['units'])} units"
    hires = [o for row in out["market"] for o in row if o and o[0] == "HIRE"]
    assert not hires, f"a plan with no hands hires {len(hires)}"


def test_the_hours_are_the_callers_and_the_tuple_is_the_count() -> None:
    """`Day` derives nothing: the tuple IS the day's labour.

    One source. Its length is the count, each entry is the hour that hand is
    AVAILABLE, and a caller that knows better (the hourly layer) passes its own
    hours straight through. The engine's earliest is a bound the CALLER asks for,
    which is why it is built out here and not inside `Day`.
    """
    from agent.world.rules import earliest_hire_times

    bound = earliest_hire_times(MAX_MARKET_ORDERS_PER_TURN + 2)
    day = B.Day(chains=CHAINS, available=AVAILABLE, hire_times=bound)
    assert day.hire_times == bound
    assert day.hands == len(bound) == MAX_MARKET_ORDERS_PER_TURN + 2
    assert bound[0] == 1 and bound[-1] == 2, bound      # ten orders a turn (F031)
    empty = B.Day(chains=CHAINS, available=AVAILABLE)
    assert empty.hire_times == () and empty.hands == 0  # the farmer walks alone
    assert B.Day(chains=CHAINS, available=AVAILABLE,
                 hire_times=(5, 5, 1)).hands == 3       # a caller's own hours survive


def test_a_complete_day_reports_the_hands_it_needs_not_the_offer() -> None:
    """Offer 5, use 1: wsr's own number, and no second search to find it.

    The search starts at the arithmetic floor and grows to the offer
    (`hands=min(floor, hands)`), so a complete answer's `pool` already IS the
    least it carried the day with. The manager reads it instead of paying for
    another search per hand.
    """
    from agent.planner import day as D

    fitted = D.fit(CHAINS, hands=5, available=AVAILABLE)
    assert fitted.complete, fitted.reason
    assert fitted.pool < 5, "the answer echoed the offer instead of the need"
    assert 0 <= fitted.floor <= fitted.pool, (fitted.floor, fitted.pool)


def test_the_day_takes_the_secretarys_hours_when_it_has_them() -> None:
    """The hours are an INPUT: the hourly secretary's timetable, or the bound.

    `DayMarket.hire_hours` is the settlement turn plus one (F040). Handing it to
    `fit` must reach the search unchanged — a hand available from hour 5 has 18
    turns of work in it, not 23, and the capacity arithmetic has to know.
    """
    from agent.planner import day as D

    given = D.fit(CHAINS, hands=2, available=AVAILABLE, hire_times=(5, 5))
    bound = D.fit(CHAINS, hands=2, available=AVAILABLE)
    # The claim is that the input REACHES the search, so the observable is that
    # the day is not the same day: hands that start at hour 5 walk a different
    # day from hands that start at hour 1. Equal numbers here would mean the
    # tuple was dropped on the way in.
    assert given.tasks == bound.tasks
    assert given.hours_used != bound.hours_used, (given.hours_used, bound.hours_used)
