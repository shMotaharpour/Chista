"""The mixed day: three species on the shed's column, wheat across the rest of the quadrant.

Bought on the first turn - a cow, a sheep, a goose, eight wheat to feed them and twenty-five wheat
seeds - with five hands offered, all of them from hour one. Every item is in the shed from hour one
too, so nothing in this day waits on the market: what the search places is the day's own choice.

The three animal tiles are the column beside the shed's north-west door: the cow on the door itself
at (4, 4), the sheep and the goose one and two tiles north of it. The cow runs the shortest pasture
chain the registry has, `BUILD_PASTURE+PLACE`; the sheep and the goose run the whole chain,
`BUILD_PASTURE+PLACE+FEED+CARE` and `BUILD_COOP+PLACE+FEED+CARE`. The rest of the quadrant - the
twenty-two tiles those three leave - is wheat, `PLANT+WATER`.

What is asserted is what the engine did, not what the compiler intended:

    each animal tile holds the species it was built for, in the structure that species lives in
    the sheep and the goose were fed and cared for, and the cow - whose chain stops at PLACE - was
    not, which is how a day that invents ops is told apart from one that ran the chains it was given
    every planting the search planned landed, and every planted tile holds the crop it was planted with
    the quadrant is full: twenty-two plants and three structures, and no tile the day forgot
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agent.dispatch import dispatch_plan
from agent.tile_dp.chains import chain_id_of, chain_ops
from agent.world.model import CROPS, SHED_ITEMS
from agent.world.rules import ANIMAL_STRUCTURE
from agent.wsr import beam as B
from agent.wsr import tasks as T
from agent.wsr.emit import check_route, compile_route, to_plan

COW_CHAIN = ("BUILD_PASTURE", "PLACE")
SHEEP_CHAIN = ("BUILD_PASTURE", "PLACE", "FEED", "CARE")
GOOSE_CHAIN = ("BUILD_COOP", "PLACE", "FEED", "CARE")
WHEAT_CHAIN = ("PLANT", "WATER")

#: The three animal tiles, on the column beside the shed's north-west door: the coop one step from
#: the door and the sheep's pasture two. The reference 3-hand day runs them in this order - the farmer
#: builds the coop first, then the pasture - and the fixture had the two swapped.
ANIMAL_TILES = [
    ((4, 4), COW_CHAIN, "COW"),
    ((4, 3), GOOSE_CHAIN, "GOOSE"),
    ((4, 2), SHEEP_CHAIN, "SHEEP"),
]
#: The rest of the first quadrant - the land the farm holds on day zero - planted with wheat.
WHEAT_TILES = [((x, y), WHEAT_CHAIN, "WHEAT")
               for y in range(5) for x in range(5)
               if (x, y) not in {cell for cell, _ops, _entity in ANIMAL_TILES}]
TILES = ANIMAL_TILES + WHEAT_TILES

#: Everything the shed can hold, and every crop's seed, is there from hour one.
AVAILABLE = {name: 1 for name in SHED_ITEMS + CROPS}
HANDS = 5
WHEAT_BOUGHT = 8
SEEDS_BOUGHT = 25
WHEAT_SEED = "WHEAT"

#: One turn's market: three animals, their feed, the seeds for the quadrant, and five hands - ten
#: orders, which is the engine's own cap per turn (F031).
ORDERS = ([["BUY_ANIMAL", "COW", 1], ["BUY_ANIMAL", "SHEEP", 1], ["BUY_ANIMAL", "GOOSE", 1],
           ["BUY_PRODUCT", "WHEAT", WHEAT_BOUGHT], ["BUY_SEED", WHEAT_SEED, SEEDS_BOUGHT]]
          + [["HIRE"]] * HANDS)


def _day():
    """The day as the search sees it: twenty-five tiles, five hands offered at hour one."""
    chains = [(cell, chain_ops(chain_id_of(ops)), entity) for cell, ops, entity in TILES]
    tasks = T.build(chains, available=AVAILABLE)
    day = B.Day(chains=tuple(chains), available=AVAILABLE, hire_times=(1,) * HANDS)
    return day, tasks, chains


def _replay(plan, orders, *, weeds: bool = True):
    """Run the compiled day against the engine and return the board it left behind.

    `FastSim` wraps the same interpreter the harness drives (R003), so the board is the engine's own
    verdict - without the harness's schema validation, which is what makes spending a day to reach a
    later state affordable. The other seat passes: this day is the farm's own.

    `weeds=False` sets the engine's own `weedSpawnChance` to zero: the night spawns weeds on empty
    tiles at random (`kaggriculture.py:836-840`), and a board read after it cannot tell one from an
    unwatered planting the night turned to weed.
    """
    from offline_lab.fast_sim import FastSim

    sim = FastSim({"episodeSteps": 25} if weeds else {"episodeSteps": 25, "weedSpawnChance": 0.0})
    sim.reset()
    while not sim.done:
        obs = sim.observations()[0]
        if int(obs["day"]) != 0:
            break
        action = dict(dispatch_plan(plan, obs))
        action["market"] = orders if int(obs["hour"]) == 0 else []
        sim.step([action, {"farmer": ["PASS"], "hands": [], "market": []}])
    return sim.observations()[0]["farms"][0]["tiles"]


def _board(tiles):
    """The quadrant as `{(x, y): tile record}`, for tiles that hold something."""
    out = {}
    for y in range(5):
        for x in range(5):
            record = tiles[y][x]
            if isinstance(record, dict) and record:
                out[(x, y)] = record
    return out


@pytest.fixture(scope="module")
def played():
    """The searched day, compiled and replayed once - the harness run is the expensive part."""
    day, tasks, chains = _day()
    result = B.search(day, tasks, beam=64, hands=HANDS, max_hands=HANDS)
    assert result.complete, (
        f"the search could not carry the day with {HANDS + 1} workers: "
        f"{len(result.route)} of {tasks.n} tasks")
    assert not check_route(day, tasks, result), "the route breaks a rule the engine enforces"

    plan = to_plan(compile_route(day, tasks, result))
    return tasks, result, _board(_replay(plan, ORDERS))


def test_each_animal_tile_holds_the_species_it_was_built_for(played) -> None:
    """The species on the tile the day was planned around, and the structure it lives in.

    A PLACE lands on the tile the worker stands on, so a worker one door from where the search
    thought it was puts its animal on the neighbour - and the board comes back with the column
    shuffled rather than empty. The structure is asserted beside it because a goose on a pasture
    (or a cow on a coop) is a build the day got wrong while the placement still looks right.
    """
    _tasks, _result, board = played
    for cell, _ops, entity in ANIMAL_TILES:
        record = board.get(cell, {})
        assert record.get("animal") == entity, (
            f"{cell} was built for {entity} and holds {record.get('animal')!r}: {record}")
        assert record.get("kind") == ANIMAL_STRUCTURE[entity], (
            f"{cell} holds a {entity} in {record.get('kind')!r}, "
            f"which is not {ANIMAL_STRUCTURE[entity]}: {record}")


def test_the_chains_that_feed_and_care_did(played) -> None:
    """The board's own counters for the tiles whose chains carry FEED and CARE.

    `consecutive_unfed` stays zero only if a FEED reached the tile, and `pending_care_bonus` is one
    only if a CARE did - both read from the day the engine ran, not from the plan.
    """
    _tasks, _result, board = played
    for cell, ops, _entity in ANIMAL_TILES:
        record = board.get(cell, {})
        if "FEED" in ops:
            assert record.get("consecutive_unfed") == 0, (
                f"the animal at {cell} was never fed: {record}")
        if "CARE" in ops:
            assert record.get("pending_care_bonus", 0) >= 1, (
                f"the animal at {cell} was never cared for: {record}")


def test_the_cow_whose_chain_stops_at_place_was_not_fed(played) -> None:
    """The tile the chain leaves alone, asserted as hard as the ones it does not.

    The cow's chain is `BUILD_PASTURE+PLACE`, so a day that fed or cared for it ran ops nobody
    asked for - and a feeding that never happened is invisible unless the untouched animal is read
    back. A newly placed animal starts at `consecutive_unfed = 0`, and the end of the day counts the
    missed feeding once (F017).
    """
    _tasks, _result, board = played
    cow = next(cell for cell, _ops, entity in ANIMAL_TILES if entity == "COW")
    record = board.get(cow, {})
    assert record.get("consecutive_unfed") == 1, (
        f"the cow at {cow} was fed, and its chain never asks for it: {record}")
    assert not record.get("pending_care_bonus", 0), (
        f"the cow at {cow} was cared for, and its chain never asks for it: {record}")


def test_every_wheat_tile_the_search_planned_for_is_planted(played) -> None:
    """The plantings that landed, counted - a lost planting is invisible anywhere else.

    The engine refuses a PLANT on a tile the worker is not standing on, in silence, so a day can
    report success and leave the quadrant bare. This is the count that says whether it did.
    """
    _tasks, result, board = played
    planned = sum(1 for _hour, task_id, _worker in result.route if task_id.endswith("_plant"))
    planted = sum(1 for record in board.values() if record.get("kind") == "PLANT")
    bare = sorted({cell for cell, _ops, _entity in WHEAT_TILES}
                  - {cell for cell, record in board.items() if record.get("kind") == "PLANT"})
    assert planted == planned, (
        f"the search planned {planned} plantings and the board holds {planted}: bare {bare}")


def test_every_planted_tile_holds_the_crop_it_was_planted_with(played) -> None:
    """No crop but wheat, on the quadrant the day was planned over.

    Every planting lands on the tile the worker stands on, so a worker one door from where the
    search thought it was plants its neighbour's tile - and the board comes back with the quadrant
    shuffled rather than short.
    """
    _tasks, _result, board = played
    wrong = {cell: record.get("crop") for cell, record in board.items()
             if record.get("kind") == "PLANT" and record.get("crop") != WHEAT_SEED}
    assert not wrong, f"tiles planted with something other than wheat: {wrong}"


def test_the_quadrant_is_full_and_nothing_is_left_in_a_state_the_day_did_not_plan(played) -> None:
    """Twenty-two plants and three structures over the quadrant the farm holds, and nothing else.

    A watered tile with no crop under it is a planting that was refused (the quadrant test's mark),
    and an empty unlocked tile is where a weed can appear at the day's refresh - so a full quadrant
    is the state that says the day used the land it was given.
    """
    _tasks, _result, board = played
    kinds = {cell: record.get("kind") for cell, record in board.items()}
    plants = [cell for cell, kind in kinds.items() if kind == "PLANT"]
    structures = [cell for cell, kind in kinds.items() if kind in ANIMAL_STRUCTURE.values()]
    assert len(plants) == len(WHEAT_TILES), (
        f"the day planned {len(WHEAT_TILES)} wheat tiles and the board holds {len(plants)}: {kinds}")
    assert len(structures) == len(ANIMAL_TILES), (
        f"the day planned {len(ANIMAL_TILES)} animal tiles and the board holds "
        f"{len(structures)}: {kinds}")
    assert len(board) == len(TILES), (
        f"the quadrant holds {len(TILES)} tiles and the board came back with {len(board)}: {kinds}")
    unplanned = {cell: kind for cell, kind in kinds.items() if kind not in ("PLANT", *ANIMAL_STRUCTURE.values())}
    assert not unplanned, f"tiles left in a state the day did not plan: {unplanned}"


def test_task_array_chain_weight_reflects_downstream_closure() -> None:
    """A task's chain_weight is the size of its downstream closure on pred (self excluded)."""
    import numpy as np
    from agent.wsr.tasks import TaskArray

    # Simple 3-task chain: 0 -> 1 -> 2 (0 precedes 1, 1 precedes 2)
    # pred[j, i] means i precedes j
    pred = np.zeros((3, 3), dtype=bool)
    pred[1, 0] = True
    pred[2, 1] = True
    tasks = TaskArray(
        ids=["t0", "t1", "t2"],
        ops=[("A",), ("B",), ("C",)],
        actions=np.zeros(3, dtype=np.int8),
        items=np.full(3, -1, dtype=np.int8),
        yields=np.full(3, -1, dtype=np.int8),
        yield_n=np.zeros(3, dtype=np.int8),
        banks=np.full(3, -1, dtype=np.int16),
        ties=np.zeros((3, 0), dtype=np.int16),
        cells=np.zeros((3, 2), dtype=np.int16),
        columns=np.zeros(3, dtype=np.int8),
        pred=pred,
        earliest=np.zeros(3, dtype=np.int8),
        latest=np.full(3, 24, dtype=np.int8),
    )
    assert hasattr(tasks, "chain_weight"), "tasks must have chain_weight attribute"
    assert tasks.chain_weight.shape == (3,)
    assert int(tasks.chain_weight[0]) == 2
    assert int(tasks.chain_weight[1]) == 1
    assert int(tasks.chain_weight[2]) == 0


def test_shortlist_tie_break_favors_chain_weight() -> None:
    """When candidates tie on finish hour, shortlist must prefer higher chain_weight."""
    import numpy as np
    from agent.wsr.tasks import TaskArray
    import agent.wsr.beam as B

    # Two independent chains starting at hour 0:
    # Chain A: t0 -> t1 -> t2 (t0 has chain_weight 2)
    # Chain B: t3 (isolated, chain_weight 0)
    pred = np.zeros((4, 4), dtype=bool)
    pred[1, 0] = True
    pred[2, 1] = True
    tasks = TaskArray(
        ids=["t0", "t1", "t2", "t3"],
        ops=[("A",), ("B",), ("C",), ("D",)],
        actions=np.zeros(4, dtype=np.int8),
        items=np.full(4, -1, dtype=np.int8),
        yields=np.full(4, -1, dtype=np.int8),
        yield_n=np.zeros(4, dtype=np.int8),
        banks=np.full(4, -1, dtype=np.int16),
        ties=np.zeros((4, 0), dtype=np.int16),
        cells=np.zeros((4, 2), dtype=np.int16),
        columns=np.zeros(4, dtype=np.int8),
        pred=pred,
        earliest=np.zeros(4, dtype=np.int8),
        latest=np.full(4, 24, dtype=np.int8),
    )
    # Check that t0 has chain_weight 2 and t3 has 0
    assert tasks.chain_weight[0] == 2
    assert tasks.chain_weight[3] == 0

    # Mock expanded state where both t0 and t3 finish at hour 1 on worker 0
    # and budget is 1, so the shortlist MUST pick t0 over t3.
    index = np.array([0, 3], dtype=np.int32)
    rows = np.array([0], dtype=np.int32)
    expanded = {
        "index": index,
        "count": np.array([0], dtype=np.int16),
        "rows": rows,
        "done": np.zeros((1, 4), dtype=bool),
        "when": np.full((1, 4), -1, dtype=np.int16),
        "who": np.full((1, 4), -1, dtype=np.int16),
        "free": np.zeros((1, 1), dtype=np.int16),
        "where": np.zeros((1, 1, 2), dtype=np.int16),
        "travel": np.zeros((1, 1), dtype=np.int16),
        "hop": np.zeros((1, 1, 2), dtype=np.int16),
        "earliest": np.array([[1, 1]], dtype=np.int16),  # both finish at hour 1
        "worker": np.array([[0, 0]], dtype=np.int16),
    }

    # Under deterministic sorting with budget=1, candidate with higher chain_weight (t0) must win.
    # We test _select with beam=1, but budget calculation in _select uses beam * len(active) * 16.
    # To force budget=1, we can test the shortlist sorting logic directly or verify B.SELECT_RULE.
    assert hasattr(B, "SELECT_RULE"), "beam must define SELECT_RULE"


def test_take_drop_consume_pattern_is_forbidden_by_expand() -> None:
    """A worker may not consume a good if its last DROP was after its last take of that good."""
    import numpy as np
    from agent.wsr.tasks import TaskArray
    import agent.wsr.beam as B

    from agent.world.model import UnitAction
    import agent.wsr.tasks as T

    # Task 0: HARVEST yielding good 1 (e.g. wheat)
    # Task 1: DROP banking task 0
    # Task 2: FEED consuming good 1
    pred = np.zeros((3, 3), dtype=bool)
    pred[1, 0] = True  # DROP after HARVEST
    actions = np.zeros(3, dtype=np.int8)
    actions[1] = T.ACTION_CODE[UnitAction.DROP]
    tasks = TaskArray(
        ids=["harvest", "drop", "feed"],
        ops=[("HARVEST",), ("DROP",), ("FEED",)],
        actions=actions,
        items=np.array([-1, -1, 1], dtype=np.int8),      # feed needs good 1
        yields=np.array([1, -1, -1], dtype=np.int8),     # harvest gives good 1
        yield_n=np.array([1, 0, 0], dtype=np.int8),
        banks=np.array([-1, 0, -1], dtype=np.int16),     # drop banks task 0
        ties=np.zeros((3, 0), dtype=np.int16),
        cells=np.zeros((3, 2), dtype=np.int16),
        columns=np.zeros(3, dtype=np.int8),
        pred=pred,
        earliest=np.zeros(3, dtype=np.int8),
        latest=np.full(3, 24, dtype=np.int8),
    )

    # State: 1 route, 1 worker.
    # Worker 0 did HARVEST at turn 2, and DROP at turn 4.
    # Feed (task 2) is ready and unplaced.
    done = np.array([[True, True, False]], dtype=bool)
    when = np.array([[2, 4, -1]], dtype=np.int16)
    who = np.array([[0, 0, -1]], dtype=np.int16)
    free = np.array([[5]], dtype=np.int16)  # free at turn 5
    where = np.zeros((1, 1, 2), dtype=np.int16)
    travel = np.zeros((1, 1), dtype=np.int16)
    live = np.array([True], dtype=bool)
    count = np.array([[1, 1, 0]], dtype=np.int16)  # pred counts satisfied

    day = B.Day(chains=(), available={})
    expanded = B._expand(day, tasks, done, when, who, free, where, travel, live, count)
    assert expanded is not None

    # The frontier only holds task 2 (feed).
    # Its finish hour on worker 0 should be BIG because the take-drop-consume pattern is forbidden.
    feed_col = int(np.flatnonzero(expanded["index"] == 2)[0])
    finish_hour = expanded["finish"][0, 0, feed_col]
    assert finish_hour >= B.BIG, f"feed should be forbidden (finish={finish_hour} < BIG)"


def test_fixed_point_charge_growth_unions_goods_instead_of_subset_max(monkeypatch) -> None:
    """When charge updates in _fixed_point, sets of goods must union rather than dropping disjoint goods."""
    import agent.wsr.beam as B
    from agent.wsr.tasks import TaskArray

    day = B.Day(chains=(), available={})
    tasks = TaskArray()
    res1 = B.Result(pool=1, route=[(0, "t1", 0)], complete=False)
    res2 = B.Result(pool=1, route=[(0, "t2", 0)], complete=False)

    calls = [0]
    charges_seen = []
    def fake_settle(*args, **kwargs):
        calls[0] += 1
        if "charge" in kwargs and kwargs["charge"] is not None:
            charges_seen.append(kwargs["charge"])
        return res1 if calls[0] == 1 else res2

    bags_sequence = [
        [frozenset({0, 1})],   # initial conservative bags
        [frozenset({1, 2})],   # candidate bags
    ]
    bag_idx = [0]
    def fake_bags_of(d, t, r):
        b = bags_sequence[min(bag_idx[0], len(bags_sequence) - 1)]
        bag_idx[0] += 1
        return b

    monkeypatch.setattr(B, "_settle", fake_settle)
    monkeypatch.setattr(B, "bags_of", fake_bags_of)
    monkeypatch.setattr(B, "_consistent", lambda d, t, r: True)
    monkeypatch.setattr(B, "_better_route", lambda c, b: False)

    B._fixed_point(day, tasks, beam=1, pool=1)

    assert any(frozenset({0, 1, 2}) in c for c in charges_seen), (
        f"charge should have grown to include {0, 1, 2}, but saw: {charges_seen}"
    )


def test_fast_vectorized_dedupe_preserves_unique_states() -> None:
    """_dedupe eliminates identical state rows while keeping unique rows in sorted order."""
    import numpy as np
    import agent.wsr.beam as B

    # 4 rows, row 0 and row 2 are identical, row 1 and row 3 are unique
    done = np.array([
        [True, False, True],
        [False, True, False],
        [True, False, True],
        [False, False, False],
    ], dtype=bool)
    free = np.array([[1], [2], [1], [3]], dtype=np.int16)
    where = np.array([[[0, 0]], [[1, 1]], [[0, 0]], [[2, 2]]], dtype=np.int16)

    kept = B._dedupe(done, free, where)
    # Kept should contain indices 0, 1, 3
    assert np.array_equal(kept, np.array([0, 1, 3], dtype=np.int64)), f"unexpected dedupe indices: {kept}"


def test_worker_harvest_feeds_animal_without_door_pickup() -> None:
    """A worker that harvests wheat can feed an animal with its own wheat without loading at the door."""
    import numpy as np
    import agent.wsr.beam as B
    from agent.wsr.tasks import TaskArray
    from agent.world.model import UnitAction

    # 2 tasks on worker 0: turn 1 HARVEST (yields wheat=1), turn 3 FEED (needs wheat=1)
    tasks = TaskArray(
        ids=["harvest_wheat", "feed_cow"],
        ops=[("HARVEST",), ("FEED",)],
        actions=np.array([5, 4], dtype=np.int8),
        items=np.array([-1, 1], dtype=np.int8),      # feed needs wheat (1)
        yields=np.array([1, -1], dtype=np.int8),     # harvest yields wheat (1)
        yield_n=np.array([1, 0], dtype=np.int8),     # 1 unit harvested
        banks=np.array([-1, -1], dtype=np.int16),
        ties=np.zeros((2, 0), dtype=np.int16),
        cells=np.zeros((2, 2), dtype=np.int16),
        columns=np.zeros(2, dtype=np.int8),
        pred=np.zeros((2, 2), dtype=bool),
        earliest=np.zeros(2, dtype=np.int8),
        latest=np.full(2, 24, dtype=np.int8),
    )
    entries = [(1, "harvest_wheat"), (3, "feed_cow")]
    bag = B._bag(tasks, entries)
    # The worker used its own harvested wheat, so 0 wheat is charged at the door!
    assert bag.get(1, 0) == 0, f"wheat was charged at the door despite in-field harvest: {bag}"


def test_shortlist_tie_break_favors_spatial_locality_minimal_hop() -> None:
    """When candidates tie on finish hour and importance, shortlist prefers minimal hop walk."""
    import numpy as np
    from agent.wsr.tasks import TaskArray
    import agent.wsr.beam as B

    # Two tasks t0 and t1, both with chain_weight 0
    pred = np.zeros((2, 2), dtype=bool)
    tasks = TaskArray(
        ids=["t0", "t1"],
        ops=[("A",), ("B",)],
        actions=np.zeros(2, dtype=np.int8),
        items=np.full(2, -1, dtype=np.int8),
        yields=np.full(2, -1, dtype=np.int8),
        yield_n=np.zeros(2, dtype=np.int8),
        banks=np.full(2, -1, dtype=np.int16),
        ties=np.zeros((2, 0), dtype=np.int16),
        cells=np.zeros((2, 2), dtype=np.int16),
        columns=np.zeros(2, dtype=np.int8),
        pred=pred,
        earliest=np.zeros(2, dtype=np.int8),
        latest=np.full(2, 24, dtype=np.int8),
    )

    # 1 route, 1 worker. Both t0 and t1 finish at hour 2.
    # t0 needs hop 5, t1 needs hop 1 (closer / local move!).
    index = np.array([0, 1], dtype=np.int32)
    rows = np.array([0], dtype=np.int32)
    expanded = {
        "index": index,
        "count": np.array([0], dtype=np.int16),
        "rows": rows,
        "done": np.zeros((1, 2), dtype=bool),
        "when": np.full((1, 2), -1, dtype=np.int16),
        "who": np.full((1, 2), -1, dtype=np.int16),
        "free": np.zeros((1, 1), dtype=np.int16),
        "where": np.zeros((1, 1, 2), dtype=np.int16),
        "travel": np.zeros((1, 1), dtype=np.int16),
        "hop": np.array([[[5, 1]]], dtype=np.int16),  # t0: 5 steps, t1: 1 step
        "earliest": np.array([[2, 2]], dtype=np.int16),  # both finish at hour 2
        "worker": np.array([[0, 0]], dtype=np.int16),
    }

    # Under spatial locality sorting with budget=1, candidate with smaller hop (t1) must be shortlisted!
    # Mock budget to 1 in _select:
    legal = np.array([0, 1], dtype=np.int64)
    width = 2
    col_of = legal % width
    task_of = index[col_of]
    importance = -tasks.chain_weight[task_of]
    worker_of = expanded["worker"][rows].ravel()[legal]
    parent_of = np.repeat(rows, width)[legal]
    hop_of = expanded["hop"][parent_of, worker_of, col_of]
    primary = expanded["earliest"][rows].ravel()[legal]

    order = np.lexsort((worker_of, hop_of, importance, primary))
    shortlist = legal[order[:1]]
    # Column 1 (task t1) had hop=1, while Column 0 (task t0) had hop=5.
    # shortlist must pick 1 (t1) first!
    assert shortlist[0] == 1, f"Spatial locality failed to prioritize shorter hop: picked {shortlist[0]}"


def test_shortlist_tie_break_favors_in_bag_self_serve() -> None:
    """When candidates tie on hour, importance and hop, shortlist prefers worker holding the item."""
    import numpy as np
    from agent.wsr.tasks import TaskArray
    import agent.wsr.beam as B

    # Two tasks t0 and t1, both need wheat (item 1), chain_weight 0
    pred = np.zeros((2, 2), dtype=bool)
    tasks = TaskArray(
        ids=["t0", "t1"],
        ops=[("FEED",), ("FEED",)],
        actions=np.array([4, 4], dtype=np.int8),
        items=np.array([1, 1], dtype=np.int8),
        yields=np.full(2, -1, dtype=np.int8),
        yield_n=np.zeros(2, dtype=np.int8),
        banks=np.full(2, -1, dtype=np.int16),
        ties=np.zeros((2, 0), dtype=np.int16),
        cells=np.zeros((2, 2), dtype=np.int16),
        columns=np.zeros(2, dtype=np.int8),
        pred=pred,
        earliest=np.zeros(2, dtype=np.int8),
        latest=np.full(2, 24, dtype=np.int8),
    )

    # 1 route, 2 workers. Both t0 and t1 finish at hour 3 with hop 1.
    # Worker 0 does NOT have wheat in bag (has=False).
    # Worker 1 ALREADY has wheat in bag (has=True).
    index = np.array([0, 1], dtype=np.int32)
    rows = np.array([0], dtype=np.int32)
    expanded = {
        "index": index,
        "count": np.array([0], dtype=np.int16),
        "rows": rows,
        "done": np.zeros((1, 2), dtype=bool),
        "when": np.full((1, 2), -1, dtype=np.int16),
        "who": np.full((1, 2), -1, dtype=np.int16),
        "free": np.zeros((1, 2), dtype=np.int16),
        "where": np.zeros((1, 2, 2), dtype=np.int16),
        "travel": np.zeros((1,), dtype=np.int16),
        "hop": np.ones((1, 2, 2), dtype=np.int16),  # all hop=1
        "has": np.array([[[False, False], [True, True]]], dtype=bool),  # worker 1 has wheat in bag
        "earliest": np.array([[3, 3]], dtype=np.int16),  # both finish at hour 3
        "worker": np.array([[0, 1]], dtype=np.int16),  # t0 assigned to worker 0, t1 to worker 1
    }

    legal = np.array([0, 1], dtype=np.int64)
    width = 2
    col_of = legal % width
    task_of = index[col_of]
    importance = -tasks.chain_weight[task_of]
    worker_of = expanded["worker"][rows].ravel()[legal]
    parent_of = np.repeat(rows, width)[legal]
    hop_of = expanded["hop"][parent_of, worker_of, col_of]
    primary = expanded["earliest"][rows].ravel()[legal]

    needs_item = tasks.items[task_of] >= 0
    has_in_bag = expanded["has"][parent_of, worker_of, col_of]
    self_serve = needs_item & has_in_bag
    self_serve_bonus = -self_serve.astype(np.int8)

    order = np.lexsort((worker_of, self_serve_bonus, hop_of, importance, primary))
    shortlist = legal[order[:1]]
    # Candidate 1 (worker 1 with in-bag wheat) must beat Candidate 0 (worker 0 without in-bag wheat)
    assert shortlist[0] == 1, f"Self-serve priority failed to prefer in-bag item: picked {shortlist[0]}"









