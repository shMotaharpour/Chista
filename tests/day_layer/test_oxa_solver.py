"""Tests for the OXA solver (Spatial Greedy Dispatcher heuristic).

The greedy has NO optimality guarantee — every answer must pass
verify_solution (validity), and where a brute-force optimum is known the
greedy's cost must be >= it (never better). Seed/retry tests from the old
r-search algorithm are gone with that algorithm.
"""
import pytest

from agent.wsr.models import (
    Instance,
    Item,
    MajorTask,
    WAREHOUSE_ENTRY_CELLS,
    Action,
    MinorTask,
    Worker,
)
from agent.wsr.oxa_solver import InfeasibleInputError, OxaConfig, compute_lower_bound, solve_oxa
from offline_lab.verify import verify_solution
from tests.day_layer.brute_force import brute_force_optimal_cost

NW = WAREHOUSE_ENTRY_CELLS["NW"]
NE = WAREHOUSE_ENTRY_CELLS["NE"]


def _worker(index: int, earliest_start: int = 0) -> Worker:
    return Worker(index=index, earliest_start=earliest_start)


def _workers(*indices: int, earliest_start: int = 0):
    return [Worker(index=i, earliest_start=earliest_start) for i in indices]


def _assert_valid_and_never_beats_brute_force(instance):
    result = solve_oxa(instance, OxaConfig(min_workers=1))
    assert result.status in ("OPTIMAL", "FEASIBLE"), result.status
    verification = verify_solution(instance, result.solution)
    assert verification.is_valid, verification.violations
    bf = brute_force_optimal_cost(instance)
    if bf is not None:
        assert verification.total_cost >= bf, (
            f"greedy cost {verification.total_cost} < proven optimum {bf}"
        )
    return result


def test_single_task():
    task = MinorTask(id="t1", cell=NW, action=Action.PASS)
    instance = Instance.compile(workers=_workers(0, 1, 2), standalone_minor_tasks=[task])
    _assert_valid_and_never_beats_brute_force(instance)


def test_two_independent_far_apart_tasks_needs_two_workers():
    # (0,0) and (9,9): entry NW(4,4) is 8 from t1 and 13 from t2. One
    # worker needs 9 + 1 + 18 = 28 turns for both; two workers (one per
    # task) finish by 9 and 10. horizon=20 forces exactly 2 workers.
    task1 = MinorTask(id="t1", cell=(0, 0), action=Action.PASS)
    task2 = MinorTask(id="t2", cell=(9, 9), action=Action.PASS)
    instance = Instance.compile(workers=_workers(0, 1, 2, 3), standalone_minor_tasks=[task1, task2], horizon=20)
    _assert_valid_and_never_beats_brute_force(instance)
    r = solve_oxa(instance, OxaConfig(min_workers=1))
    used = len([rt for rt in r.solution.routes if rt.tasks])
    assert used == 2, used


def test_precedence_chain_across_two_cells():
    task_a = MinorTask(id="a", cell=NW, action=Action.PASS)
    task_b = MinorTask(id="b", cell=NE, action=Action.PASS)
    instance = Instance.compile(
        workers=_workers(0, 1, 2),
        standalone_minor_tasks=[task_a, task_b],
        explicit_precedence=[("a", "b")],
    )
    _assert_valid_and_never_beats_brute_force(instance)


def test_single_worker_group_with_fixed_cell_pickup():
    pickup = MinorTask(id="pickup", cell=NW, action=Action.PICKUP, item=Item.WHEAT, qty=1)
    feed = MinorTask(id="feed", cell=NE, action=Action.FEED, item=Item.WHEAT, qty=1)
    instance = Instance.compile(
        workers=_workers(0, 1, 2),
        standalone_minor_tasks=[pickup, feed],
        explicit_precedence=[("pickup", "feed")],
        explicit_single_worker_groups=[["pickup", "feed"]],
        warehouse_stock={Item.WHEAT: 5},
    )
    _assert_valid_and_never_beats_brute_force(instance)


def test_one_step_horizon_forces_exactly_two_workers():
    task1 = MinorTask(id="t1", cell=NW, action=Action.PASS)
    task2 = MinorTask(id="t2", cell=NE, action=Action.PASS)
    instance = Instance.compile(workers=_workers(0, 1, 2, 3), standalone_minor_tasks=[task1, task2], horizon=1)
    result = solve_oxa(instance, OxaConfig(min_workers=1))
    assert result.status in ("OPTIMAL", "FEASIBLE"), result.status
    v = verify_solution(instance, result.solution)
    assert v.is_valid, v.violations
    used = len([rt for rt in result.solution.routes if rt.tasks])
    assert used == 2, used


def test_entry_cell_placement_tie_break_all_four_cells():
    # One task colocated with each of the 4 shed-adjacent cells, horizon=1:
    # four workers needed, entries must be NW,NE,SW,SE in that order.
    cells = [WAREHOUSE_ENTRY_CELLS[n] for n in ("NW", "NE", "SW", "SE")]
    tasks = [MinorTask(id=f"t{i}", cell=c, action=Action.PASS) for i, c in enumerate(cells)]
    instance = Instance.compile(
        workers=_workers(0, 1, 2, 3), standalone_minor_tasks=tasks, horizon=1
    )
    result = solve_oxa(instance, OxaConfig(min_workers=1))
    v = verify_solution(instance, result.solution)
    assert v.is_valid, v.violations
    starts = {rt.worker_index: rt.start_cell for rt in result.solution.routes}
    assert starts[0] == WAREHOUSE_ENTRY_CELLS["NW"]
    assert starts[1] == WAREHOUSE_ENTRY_CELLS["NE"]
    assert starts[2] == WAREHOUSE_ENTRY_CELLS["SW"]
    assert starts[3] == WAREHOUSE_ENTRY_CELLS["SE"]


def test_regression_first_task_needs_its_own_turn_not_just_travel_time():
    # Constraint 3: exec >= start + 1 + dist. A single task at NW with
    # start_time=0 must have exec >= 1, never 0.
    task = MinorTask(id="t1", cell=NW, action=Action.PASS)
    instance = Instance.compile(workers=[Worker(index=0, earliest_start=0)], standalone_minor_tasks=[task])
    result = solve_oxa(instance, OxaConfig(min_workers=1))
    st = result.solution.routes[0].tasks[0]
    assert st.exec_time >= 1


def test_drop_genuinely_resets_inventory_between_two_pickup_place_rounds():
    pickup1 = MinorTask(id="pickup1", cell=None, action=Action.PICKUP, item=Item.WHEAT, qty=1)
    place1 = MinorTask(id="place1", cell=NW, action=Action.PLACE, item=Item.WHEAT, qty=1)
    drop = MinorTask(id="drop", cell=NE, action=Action.DROP)
    pickup2 = MinorTask(id="pickup2", cell=None, action=Action.PICKUP, item=Item.WHEAT, qty=1)
    place2 = MinorTask(id="place2", cell=NE, action=Action.PLACE, item=Item.WHEAT, qty=1)
    instance = Instance.compile(
        workers=_workers(0),
        standalone_minor_tasks=[pickup1, place1, drop, pickup2, place2],
        explicit_precedence=[("pickup1", "place1"), ("place1", "drop"),
                             ("drop", "pickup2"), ("pickup2", "place2")],
    )
    result = solve_oxa(instance, OxaConfig(min_workers=1))
    v = verify_solution(instance, result.solution)
    assert v.is_valid, v.violations


def test_unsourcable_feed_is_reported_not_raised():
    # FEED whose acquire can't be sourced (no pickup task exists, no
    # warehouse stock). The greedy schedules it without tracking items, so
    # its own validation flags the answer: either INFEASIBLE (greedy left
    # it unscheduled) or INVALID_SOLUTION (verify catches the negative
    # inventory). Both mean "no valid day plan" - never an exception and
    # never a silently-wrong FEASIBLE.
    feed = MinorTask(id="feed", cell=NW, action=Action.FEED, item=Item.WHEAT, qty=1)
    instance = Instance.compile(workers=[_worker(0)], standalone_minor_tasks=[feed])
    with pytest.raises(InfeasibleInputError):
        solve_oxa(instance, OxaConfig(min_workers=1))


def test_global_stock_overrun_raises_before_solving():
    pickup = MinorTask(id="pickup", cell=None, action=Action.PICKUP, item=Item.WHEAT, qty=10)
    instance = Instance.compile(
        workers=[_worker(0)], standalone_minor_tasks=[pickup], warehouse_stock={Item.WHEAT: 3}
    )
    with pytest.raises(InfeasibleInputError):
        solve_oxa(instance, OxaConfig(min_workers=1))


def test_an_edge_into_a_preloaded_pickup_is_rejected_not_mis_scheduled():
    # `models.py` marks a pickup *aggregatable* when it precedes its own
    # consume (here S -> C), and the solver then emits it as the setup turn at
    # the head of a route without consulting the precedence graph. An edge
    # *into* S is therefore unrepresentable, and the greedy used to answer
    # INVALID_SOLUTION for such a day -- `verify_solution` catching
    # "precedence violated: 'T' must strictly precede 'S'" (docs/F057, fifth
    # class) -- while `cpsat_binarySearch` scheduled it (OPTIMAL, S at 11,
    # C at 24, T at 10). R007 failing direction: drop the validation and this
    # stops raising, returning INVALID_SOLUTION instead.
    instance = Instance.compile(
        workers=_workers(0, 1),
        standalone_minor_tasks=[
            MinorTask(id="T", cell=NW, action=Action.PASS),
            MinorTask(id="S", cell=None, action=Action.PICKUP,
                      item=Item.WHEAT, qty=1),
            MinorTask(id="C", cell=NE, action=Action.FEED,
                      item=Item.WHEAT, qty=1),
        ],
        explicit_precedence=[("T", "S"), ("S", "C")],
        explicit_single_worker_groups=[["S", "C"]],
        warehouse_stock={Item.WHEAT: 5},
        horizon=24,
    )
    with pytest.raises(InfeasibleInputError):
        solve_oxa(instance, OxaConfig(min_workers=1))


def test_empty_instance_is_trivially_optimal_with_zero_cost():
    instance = Instance.compile(workers=_workers(0, 1, 2), standalone_minor_tasks=[])
    result = solve_oxa(instance, OxaConfig(min_workers=1))
    assert result.status == "OPTIMAL"
    assert result.solution.hired == 0


def test_the_worker_pool_is_the_lowest_indices_not_the_given_order():
    # `cpsat_solver` sorts workers by index before applying the pool cap, and
    # verify.py's COST ACCOUNTING is the engine's append-only prefix by index,
    # so a pool of one offers hand 0. Capping the caller's order instead let
    # OXA route the first-listed worker: with workers [(3,0),(0,0),(1,0)] and
    # worker_pool_size=1 it returned hand 3 at hired 4 where the oracle
    # returns hand 0 at cost 0 for the same day.
    # R007 failing direction: drop the `sorted(...)` and this test reads
    # hand 3 / cost 4 (measured).
    instance = Instance.compile(
        workers=[_worker(3), _worker(0), _worker(1)],
        standalone_minor_tasks=[
            MinorTask(id="t", cell=WAREHOUSE_ENTRY_CELLS["SE"], action=Action.PASS)
        ],
        worker_pool_size=1,
        horizon=24,
    )
    result = solve_oxa(instance, OxaConfig(min_workers=1))

    assert result.status == "OPTIMAL", result.status
    assert [route.worker_index for route in result.solution.routes] == [0]
    assert result.solution.hired == 0
    assert verify_solution(instance, result.solution).is_valid


def test_no_workers_available_is_infeasible_when_tasks_exist():
    task = MinorTask(id="t1", cell=NW, action=Action.PASS)
    instance = Instance.compile(workers=[], standalone_minor_tasks=[task])
    result = solve_oxa(instance, OxaConfig(min_workers=1))
    assert result.status == "INFEASIBLE"


def test_lower_bound_accounts_for_single_worker_group_chain_cost():
    # compute_lower_bound: a feed chain (acquire + feed) costs 2 turns, so
    # with horizon 2 two chains need 2 workers minimum.
    m1 = MajorTask(id="m1", type="feed", cell=NW)
    m2 = MajorTask(id="m2", type="feed", cell=NE)
    instance = Instance.compile(workers=_workers(0, 1), major_tasks=[m1, m2], horizon=2)
    lb = compute_lower_bound(instance)
    assert lb == 2, lb
