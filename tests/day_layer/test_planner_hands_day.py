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
from agent.world.rules import MAX_ORDERS_PER_TURN, hire_hour
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


def test_a_day_s_hands_start_at_the_engine_s_own_hours() -> None:
    day = B.Day(chains=CHAINS, available=AVAILABLE, hands=MAX_ORDERS_PER_TURN + 2)
    assert day.hire_times == tuple(hire_hour(k) for k in range(MAX_ORDERS_PER_TURN + 2))
    assert day.hire_times[0] == 1 and day.hire_times[-1] == 2, day.hire_times
    assert B.Day(chains=CHAINS, available=AVAILABLE, hands=0).hire_times == ()
