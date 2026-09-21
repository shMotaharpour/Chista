"""A deadline's walking, and a good that should never go through the shed.

Two things the layer has code for and no test that would notice if they stopped being true.

The first is a deadline: `build(..., drop_by=[...])` derives a DROP task and the floor charges twice
the distance from a shed door to the furthest tile, because the unit reaches it and comes back. That
term only shows on a day where the doubled reach beats the spanning tree - a few tiles, all of them
far - so the day here is three tiles in the corner, worked hard.

The second is a good a worker grew or collected itself. `COLLECT_FERTILIZER` puts fertilizer in the bag
and a `HARVEST` puts the crop there, so a `FERTILIZE` or a `FEED` that follows one - on the same tile
or another, as long as it is the same worker - needs no trip at all. Measured today, the search puts
the producer and the consumer on DIFFERENT workers, so the collected fertilizer is wasted and the
consumer pays for a trip the layer charges one turn for and no walk. Those two tests are marked with
that measurement; they are the shape the layer should have, not the shape it has.
"""
import pathlib
import sys

import pytest

sys.path.insert(0, "/chista/pm/world")

from agent.wsr import beam as B
from agent.wsr import tasks as T

#: Three tiles in the far corner, each worked hard: the doubled reach beats the tree here, which is
#: what a deadline's term is for.
FAR_CORNER = [(0, 0), (0, 1), (0, 2)]
PER_TILE = 12
NEAR, FAR = (5, 5), (1, 1)
TIMETABLE = {"FERTILIZER": 0, "WHEAT": 0}

_NOT_TIED = (
    "measured on this branch: the search puts the producer and the consumer on different workers "
    "(a COLLECT on worker 1 and its FERTILIZE on worker 0), so the good is wasted and the consumer "
    "pays a trip. Nothing ties them - `pred` is empty between the two and the trip is charged one "
    "turn with no walk to the door"
)


def test_a_deadline_makes_the_walking_longer():
    """A unit with a deadline reaches the furthest tile and comes back to a door."""
    chains = [(cell, ("HARVEST",) * PER_TILE, None) for cell in FAR_CORNER]
    plain = T.build(chains, available=TIMETABLE)
    banked = T.build(chains, available=TIMETABLE, drop_by=[20] * len(chains))

    assert banked.drop_rows.size > 0, "a chain with a deadline gets a DROP and this one did not"
    assert plain.drop_rows.size == 0

    reach = int(T.DISTANCE[T.SHED_INDEX].min(axis=0)[banked.cell_index].max())
    assert T.day_walking(banked) >= 2 * reach, (
        f"the walking is {T.day_walking(banked)} where the round trip to the furthest tile is "
        f"{2 * reach}")
    assert T.day_walking(banked) > T.day_walking(plain), (
        f"the walking is {T.day_walking(plain)} either way: the deadline is not in it")


@pytest.mark.xfail(strict=True, reason=_NOT_TIED)
def test_a_collected_good_reaches_its_consumer():
    """The worker that collects the fertilizer is the worker that spreads it: the bag is not shared."""
    chains = [(NEAR, ("COLLECT_FERTILIZER",), None), (FAR, ("FERTILIZE",), None)]
    tasks = T.build(chains, available=TIMETABLE)
    result = B.search(B.Day(chains=tuple(chains), available=TIMETABLE, hire_times=(1, 1)),
                      tasks, hands=2, max_hands=2)

    assert result.complete
    worker_of = {task_id: worker for _turn, task_id, worker in result.route}
    assert worker_of["d0_collect_fertilizer"] == worker_of["d1_fertilize"], (
        f"the fertilizer was collected by {worker_of['d0_collect_fertilizer']} and spread by "
        f"{worker_of['d1_fertilize']}, so the collection was wasted")


@pytest.mark.xfail(strict=True, reason=_NOT_TIED)
def test_a_harvested_crop_reaches_its_consumer():
    """The same for wheat: a harvest puts it in the bag and a feed takes it out of that bag."""
    chains = [(NEAR, ("HARVEST",), None), (FAR, ("FEED",), None)]
    tasks = T.build(chains, available=TIMETABLE)
    result = B.search(B.Day(chains=tuple(chains), available=TIMETABLE, hire_times=(1, 1)),
                      tasks, hands=2, max_hands=2)

    assert result.complete
    worker_of = {task_id: worker for _turn, task_id, worker in result.route}
    assert worker_of["d0_harvest"] == worker_of["d1_feed"], (
        f"the wheat was harvested by {worker_of['d0_harvest']} and fed by "
        f"{worker_of['d1_feed']}, so the harvest was wasted")
