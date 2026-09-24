"""The spare capacity: the turns the route leaves, and the walks come off it.

A manager reads `spare` after `complete` says a day is doable, to decide whether to lay more work on
the same hands. So the guard is that a turn spent walking is NOT spare: one task on a far tile costs
its own turn plus the walk, and a spare that counted only the tasks would say the day is nearly empty.
"""
import pathlib
import sys

sys.path.insert(0, "/chista/pm/world")

from agent.wsr import beam as B
from agent.wsr import tasks as T

#: The walk from the farmer's own spawn to the (0, 0) tile, which is where these days put their work.
WALK = abs(B.FARMER_START[0]) + abs(B.FARMER_START[1])


def _day(chains, hire_times=()):
    return B.Day(chains=tuple(chains), available={}, hire_times=hire_times)


def test_the_walk_is_not_spare():
    """One task eight tiles away: the day has 24 turns, the task takes one and the walk takes eight."""
    chains = [((0, 0), ("WATER",), None)]
    tasks = T.build(chains, available={})
    result = B.search(_day(chains), tasks, hands=0, max_hands=0)

    assert result.complete, "a single task cannot fail to fit in a day"
    assert len(result.route) == 1
    assert result.spare == 24 - (1 + WALK), (
        f"spare {result.spare} for one task {WALK} tiles away: the walk is not spare")


def test_more_work_leaves_less_spare():
    """The number a manager acts on has to move the right way when the day gets more work."""
    one = [((0, 0), ("WATER",), None)]
    two = [((0, 0), ("WATER",), None), ((0, 1), ("WATER",), None)]

    small = B.search(_day(one), T.build(one, available={}), hands=0, max_hands=0)
    bigger = B.search(_day(two), T.build(two, available={}), hands=0, max_hands=0)

    assert small.complete and bigger.complete, "two adjacent tiles fit one hand's day"
    assert bigger.spare < small.spare, (
        f"a second task raised the spare from {small.spare} to {bigger.spare}")


def test_an_unhired_hand_is_not_capacity():
    """The spare is counted against the hands the pool paid for, not against the offer."""
    chains = [((0, 0), ("WATER",), None)]
    tasks = T.build(chains, available={})
    offered = B.Day(chains=tuple(chains), available={}, hire_times=(1, 1, 1, 1))
    result = B.search(offered, tasks, hands=0, max_hands=0)

    assert result.pool == 0, f"the search hired {result.pool} hands where it was told to hire none"
    assert result.spare == 24 - (1 + WALK), (
        f"spare {result.spare}: four hands were offered, none hired, and the day is the farmer's")
