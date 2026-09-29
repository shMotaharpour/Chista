"""A deadline's walking, and a good that never goes through the shed.

Two things the layer has code for and no test that would notice if they stopped being true.

The first is a deadline: `build(..., drop_by=[...])` derives a DROP task and the floor charges twice
the distance from a shed door to the furthest tile, because the unit reaches it and comes back. That
term only shows on a day where the doubled reach beats the spanning tree - a few tiles, all of them
far - so the day here is three tiles in the corner, worked hard.

The second is a good a worker grew or collected itself. `COLLECT_FERTILIZER` puts fertilizer in the
bag and a `HARVEST` puts the crop there, so a `FERTILIZE` or a `FEED` that follows one - on the same
tile or another, as long as it is the same worker - needs no trip at all. The engine's own bag rule
makes the constraint when the shed holds none of the good: a FEED takes the wheat from the FEEDING
unit's inventory, a FERTILIZE the fertilizer from its own, and a unit with an empty bag is refused
in silence (F047). So the builder binds every consumer of an unstocked grown good to that good's
producer (`T.build`, the `available` pass) - and the tests here read the engine's rule back: no
shed stock, no shed pickup, one worker for the whole take-and-eat set.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from agent.wsr import beam as B
from agent.wsr import tasks as T

#: Three tiles in the far corner, each worked hard: the doubled reach beats the tree here, which is
#: what a deadline's term is for. The corner has no shed stock of anything, so the grown-and-eaten
#: goods in the other tests are bound to their producers - there is nothing to pick up.
FAR_CORNER = [(0, 0), (0, 1), (0, 2)]
PER_TILE = 12
NEAR, FAR = (5, 5), (1, 1)
TIMETABLE = {"FERTILIZER": 0, "WHEAT": 0}
#: The shed holds nothing: the bag tests' days grow what they eat, which is the engine's own rule
#: for a bag - a FEED by a worker that harvested nothing has no wheat to spend.
EMPTY = {}



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


def test_an_unstocked_good_binds_its_consumers_to_its_producer():
    """No shed stock: the day grows what it eats, and the engine's own bag rule makes the tie.

    The engine's FEED takes the wheat from the feeding unit's own inventory - a FEED by a worker
    that harvested nothing spends nothing (the op is refused in silence, F047), and a shed pickup
    cannot help because the shed holds no wheat. So with the timetable empty, both FEEDs belong
    to the worker whose HARVEST filled its bag.
    """
    chains = [((5, 5), ("HARVEST",), "WHEAT"),
              ((1, 1), ("FEED",), None),
              ((1, 2), ("FEED",), None)]
    tasks = T.build(chains, available=EMPTY, harvests=[("WHEAT", 2), None, None])
    result = B.search(B.Day(chains=tuple(chains), available=EMPTY, hire_times=(1, 1)),
                      tasks, hands=2, max_hands=2)

    assert result.complete
    worker_of = {task_id: worker for _turn, task_id, worker in result.route}
    harvester = worker_of["d0_harvest"]
    assert harvester == worker_of["d1_feed"] == worker_of["d2_feed"], (
        f"the harvest went to worker {harvester} but the feeds to "
        f"{worker_of['d1_feed']} and {worker_of['d2_feed']}: with no shed stock those FEEDs "
        f"have no wheat in their bag and the engine refuses them in silence")


def test_a_collected_good_reaches_its_consumer():
    """No shed stock: the collected fertilizer is spent by its own collector - the bag is not shared."""
    chains = [(NEAR, ("COLLECT_FERTILIZER",), None), (FAR, ("FERTILIZE",), None)]
    tasks = T.build(chains, available=EMPTY)
    result = B.search(B.Day(chains=tuple(chains), available=EMPTY, hire_times=(1, 1)),
                      tasks, hands=2, max_hands=2)

    assert result.complete
    worker_of = {task_id: worker for _turn, task_id, worker in result.route}
    assert worker_of["d0_collect_fertilizer"] == worker_of["d1_fertilize"], (
        f"the fertilizer was collected by {worker_of['d0_collect_fertilizer']} and spread by "
        f"{worker_of['d1_fertilize']}, so with no shed stock the spreading has nothing to spend")


def test_a_harvested_crop_reaches_its_consumer():
    """The same for wheat: no shed stock, so the harvest's own worker does the feeding."""
    chains = [(NEAR, ("HARVEST",), "WHEAT"), (FAR, ("FEED",), None)]
    tasks = T.build(chains, available=EMPTY)
    result = B.search(B.Day(chains=tuple(chains), available=EMPTY, hire_times=(1, 1)),
                      tasks, hands=2, max_hands=2)

    assert result.complete
    worker_of = {task_id: worker for _turn, task_id, worker in result.route}
    assert worker_of["d0_harvest"] == worker_of["d1_feed"], (
        f"the wheat was harvested by {worker_of['d0_harvest']} and fed by "
        f"{worker_of['d1_feed']}, so with no shed stock the feeding has nothing to spend")
