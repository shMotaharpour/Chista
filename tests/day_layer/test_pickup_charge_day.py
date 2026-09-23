"""The pickup charge the search iterates grows by the TURN a worker's walk starts, not by its goods.

`_fixed_point` charges each worker the goods it loads at its door and re-searches until the charge
stops growing. Grown by the union of goods, two routes that each load one different good charge two
pickup turns where either needed one, and a worker charged a turn nobody spends starts every walk a
turn late. The guards are the rule itself and the mixed three-hand day, which that turn costs.
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from agent.wsr import beam as B
from tests.day_layer import test_mixed_day_pool as P

#: Measured: the count rule (before the charge became a set) placed 53 of 54 at beam 64 with three
#: hands, and the union placed 47.
MIXED_PLACED = 53


def test_the_charge_grows_by_the_later_walk_not_by_the_union_of_goods() -> None:
    """Two routes that each load one different good charge one pickup turn, not two."""
    day, tasks = P._day()
    arrival = B.good_hours(tasks)
    first, second = sorted(arrival)[:2]
    assert arrival[first] == arrival[second], "the premise: both goods land at the same hour"
    hour = 1
    one = B.first_walk_turn(hour, frozenset({first}), arrival)
    other = B.first_walk_turn(hour, frozenset({second}), arrival)
    grown = B.grow_charge(hour, frozenset({first}), frozenset({second}), arrival)
    assert B.first_walk_turn(hour, grown, arrival) == max(one, other), (
        f"the grown charge starts the walk at {B.first_walk_turn(hour, grown, arrival)}; the two "
        f"routes it grew from start at {one} and {other}")


def test_the_mixed_day_is_placed_as_far_as_the_count_rule_placed_it() -> None:
    day, tasks = P._day(hands=P.REFERENCE_HANDS)
    result = B.search(day, tasks, beam=64, hands=P.REFERENCE_HANDS, max_hands=P.REFERENCE_HANDS)
    assert len(result.route) >= MIXED_PLACED, (
        f"the search placed {len(result.route)} of {tasks.n} where it placed {MIXED_PLACED} before "
        f"the charge became a set - a worker is charged a pickup turn nobody spends")
