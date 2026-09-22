"""The budget's contract: a bigger budget never places less, and a route says where it was priced.

Three defects lived here, and the first two were invisible to a test that only asked whether a search
returns a route.

A re-run the deadline cuts at step 8 or 16 came back non-empty and much worse, and it replaced the
attempt before it - so a larger budget could return a WORSE route, which is the property the branch
claimed and did not have. Measured: 60 ms placed 40 tasks where 80 ms placed 8. The guard caught only
a re-run that came back empty. The fix is the reserve: an attempt costs about what the last one cost,
so one that cannot be finished is not started.

And `Result` carried no settled positions, so a caller holding one could not compile it: the compiler
re-derived them from the route, which agrees with the search only when the fixed point converged.
Cut short, it raised in the agent's hot path - one season in four.

The clock here is the test's own. A real deadline that lands between two attempts is a race, and a
test that fails for the hardware is worth less than no test: each call to the clock costs one unit,
so the sweep is a sweep.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from agent.tile_dp.chains import chain_id_of, chain_ops
from agent.wsr import beam as B
from agent.wsr import tasks as T

OPS = chain_ops(chain_id_of(("PLANT", "WATER")))
AVAILABLE = {"WHEAT": 1}
HANDS = 8


def _day():
    chains = tuple(((x, y), OPS, "WHEAT") for y in range(8) for x in range(8))[:50]
    return chains, T.build(chains, available=AVAILABLE)


def _day_for(chains):
    return B.Day(chains=chains, available=AVAILABLE, hire_times=(1,) * HANDS)


def test_a_bigger_budget_never_places_less(monkeypatch):
    """The property the branch claimed: more budget is never a worse route.

    Every budget is swept on one instance, and the count of tasks placed must not fall as the budget
    rises. It did: the fixed point started an attempt it could not finish, and that attempt's
    half-built route replaced a fuller one.
    """
    chains, tasks = _day()
    clock = {"now": 0.0}

    def tick():
        clock["now"] += 1.0
        return clock["now"]

    monkeypatch.setattr(B.time, "perf_counter", tick)

    placed = []
    for budget in (3.0, 6.0, 9.0, 12.0, 20.0, 40.0, 100.0):
        result = B.search(_day_for(chains), tasks, hands=HANDS, max_hands=HANDS, budget_s=budget)
        placed.append(len(result.route))

    assert placed == sorted(placed), f"a larger budget placed less: {placed}"


def test_a_result_always_says_where_it_was_priced_from():
    """Never None: the compiler's None means derive from the route, which is a different question."""
    chains, tasks = _day()
    result = B.search(_day_for(chains), tasks, hands=HANDS, max_hands=HANDS, budget_s=0.001)

    assert result.settled, "a Result must carry the positions its route was priced from"


def test_a_carried_day_does_not_ask_to_be_ground():
    """A route can be complete and still have crossed its deadline; it has no work left to place."""
    carried = B.Result(pool=2, route=[(0, "a", 0)], complete=True, out_of_time=True)
    partial = B.Result(pool=2, route=[(0, "a", 0)], complete=False, out_of_time=True)

    assert carried.can_improve is False, "a carried day has nothing left to place"
    assert partial.can_improve is True, "an incomplete day cut short is what the flag is for"


def test_the_better_route_rule_orders_a_carried_route_above_a_longer_partial():
    """The rule the pool loop and the halving share: carried first, then more work."""
    carried = B.Result(pool=4, route=[(0, "a", 0)], complete=True)
    longer = B.Result(pool=4, route=[(0, "a", 0), (1, "b", 0)], complete=False)
    shorter = B.Result(pool=4, route=[(0, "a", 0)], complete=False)

    assert B._better_route(carried, longer) is True, "carrying the day beats placing more of it"
    assert B._better_route(longer, shorter) is True, "among equals, more work wins"
    assert B._better_route(shorter, longer) is False
