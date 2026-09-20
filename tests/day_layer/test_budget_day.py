"""The budget's contract: a slice returns the work it placed, and a carried day asks for nothing.

Two bugs lived here, both of them invisible to a test that only asked whether a search returns a
route. The fixed point runs the search again from the settled positions, and it returned the LAST
attempt rather than the best - so once the deadline passed, each remaining attempt started a run cut
at its first step and handed back an EMPTY route, which replaced the work the attempt before it had
placed. A run that placed 24 came back through the fixed point as 0, and the whole feature was
unusable at the sizes it exists for. And `can_improve` was the deadline flag itself, so a search that
carried the whole day but crossed its deadline told the caller to come back to a finished day.

The clock is the test's own, not the machine's: a real deadline that lands between two attempts is a
race, and a test that fails for the hardware is worth less than no test.
"""
import sys

sys.path.insert(0, "/chista/pm/world")

from agent.tile_dp.chains import chain_id_of, chain_ops
from agent.wsr import beam as B
from agent.wsr import tasks as T

OPS = chain_ops(chain_id_of(("WATER", "HARVEST", "PLANT", "FERTILIZE", "WATER")))
AVAILABLE = {"WHEAT": 1, "FERTILIZER": 1}
HANDS = 8


def _day():
    chains = tuple(((x, y), OPS, "WHEAT") for y in range(8) for x in range(8))[:50]
    return chains, T.build(chains, available=AVAILABLE)


def test_a_rerun_the_deadline_cuts_does_not_replace_the_work(monkeypatch):
    """The first attempt placed work; a later one cut short must not hand back an empty route.

    The clock sits before the deadline for the first attempt and past it for every one after, so the
    fixed point has to answer with what it already has. Returning the last attempt instead - which is
    what it did - answers with nothing at all.
    """
    chains, tasks = _day()
    day = B.Day(chains=chains, available=AVAILABLE, hire_times=(1,) * HANDS)
    clock = {"now": -1.0, "attempts": 0}
    real_run = B._run

    def counted(*args, **kwargs):
        clock["attempts"] += 1
        if clock["attempts"] > 1:
            clock["now"] = 999.0
        return real_run(*args, **kwargs)

    monkeypatch.setattr(B.time, "perf_counter", lambda: clock["now"])
    monkeypatch.setattr(B, "_run", counted)

    result = B._fixed_point(day, tasks, B.beam_for(tasks, HANDS + 1), HANDS, 0.0)

    assert clock["attempts"] > 1, "the fixed point has to have tried a second time"
    assert len(result.route) > 0, (
        "the deadline replaced the work the first attempt placed with an empty route"
    )


def test_a_carried_day_does_not_ask_to_be_ground():
    """A route can be complete and still have crossed its deadline; it has no work left to place."""
    carried = B.Result(pool=2, route=[(0, "a", 0)], complete=True, out_of_time=True)
    partial = B.Result(pool=2, route=[(0, "a", 0)], complete=False, out_of_time=True)

    assert carried.can_improve is False, "a carried day has nothing left to place"
    assert partial.can_improve is True, "an incomplete day cut short is what the flag is for"


def test_the_better_route_rule_orders_a_carried_route_above_a_longer_partial():
    """The rule the pool loop and the fixed point share: carried first, then more work."""
    carried = B.Result(pool=4, route=[(0, "a", 0)], complete=True)
    longer = B.Result(pool=4, route=[(0, "a", 0), (1, "b", 0)], complete=False)
    shorter = B.Result(pool=4, route=[(0, "a", 0)], complete=False)

    assert B._better_route(carried, longer) is True, "carrying the day beats placing more of it"
    assert B._better_route(longer, shorter) is True, "among equals, more work wins"
    assert B._better_route(shorter, longer) is False
