"""`wsr_check` is the difference between a check and a commitment.

The manager iterates on the check — the hourly secretary lays the day out, wsr
answers with its free slots or its shortfall, the manager changes the plan — and
commits once. The HIRE orders are what makes that a commitment: they spend the
purse and put hands on the field.
"""

from __future__ import annotations

from agent.planner import market as K
from offline_lab.fast_sim import FastSim


def _obs():
    return FastSim({"episodeSteps": 25}).observations()[0]


def _hires(rows) -> int:
    return sum(1 for hour in rows for o in hour if o and o[0] == "HIRE")


def test_a_check_lays_the_day_out_without_the_hires() -> None:
    obs = _obs()
    check = K.build(obs, (), hands=3, wsr_check=False)
    commit = K.build(obs, (), hands=3, wsr_check=True)
    assert _hires(check.rows) == 0, "a check committed hires"
    assert _hires(commit.rows) == 3, _hires(commit.rows)
    # NOT a bill assertion: `HAND_COST_MULT = 0` (rules.hire_cost) makes the hire
    # bill zero until the planner is given a labour-cost model, so the two bills
    # are equal by the owner's order and would say nothing about the flag.


def test_zero_hands_commits_nobody() -> None:
    """Zero is an offer: the farmer walks alone, and the purse pays no hire."""
    obs = _obs()
    rows = K.build(obs, (), hands=0, wsr_check=True)
    assert _hires(rows.rows) == 0


def test_the_queue_reports_when_its_hires_are_available() -> None:
    """The timetable is READ off the queue, not assumed from the engine's bound.

    Ten orders fill turn 0 (F031), so the hire behind them settles in turn 1 and
    the hand is available from hour 2 (F040) — not hour 1, which is what a bound
    that spends the whole turn on hires would claim.
    """
    from agent.planner.market import settle_hours

    rows = [[["SELL", "MILK", 3]] * 10,
            [["HIRE"], ["BUY_SEED", "WHEAT", 2]]]
    hands, goods = settle_hours(rows)
    assert hands == (2,), hands
    assert goods == (("WHEAT", 2),), goods


def test_a_queue_with_room_settles_its_hire_at_once() -> None:
    from agent.planner.market import settle_hours

    hands, goods = settle_hours([[["HIRE"], ["BUY_ANIMAL", "COW"]]])
    assert hands == (1,), hands          # turn 0 + F040
    assert goods == (("COW", 1),), goods
