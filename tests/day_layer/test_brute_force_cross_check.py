"""The most important optimality test: CP-SAT's answer, on small instances,
must exactly match a completely independent brute-force enumeration (see
tests/brute_force.py). Unlike the hand-solved fixtures (whose expected
cost is *my* reasoning about the model), this can catch a bug in the
CP-SAT formulation itself, not just its implementation.
"""
import pytest

from agent.wsr.models import Instance, Item, MajorTask, WAREHOUSE_ENTRY_CELLS, Action, MinorTask, Worker
from agent.wsr.solvers.cpsat_solver import CpSatConfig, solve_cpsat
from offline_lab.verify import verify_solution
from tests.day_layer.brute_force import brute_force_optimal_cost

NW = WAREHOUSE_ENTRY_CELLS["NW"]
NE = WAREHOUSE_ENTRY_CELLS["NE"]

# Optimizing mode: CP-SAT must prove the true optimum (Minimize on) for
# the comparison with brute force to be meaningful.
FAST = CpSatConfig(time_limit_seconds=10, feasibility_only=False)


def _workers(*indices: int, earliest_start: int = 0) -> list[Worker]:
    return [Worker(index=i, earliest_start=earliest_start) for i in indices]


def _assert_matches_brute_force(instance):
    expected = brute_force_optimal_cost(instance)
    result = solve_cpsat(instance, FAST)
    if expected is None:
        assert result.status == "INFEASIBLE", (result.status, expected)
        return
    assert result.status == "OPTIMAL", result.status
    assert result.solution.hired == expected
    verification = verify_solution(instance, result.solution)
    assert verification.is_valid, verification.violations
    assert verification.total_cost == expected


def test_single_task():
    task = MinorTask(id="t1", cell=NW, action=Action.PASS)
    instance = Instance.compile(workers=_workers(0, 1, 2), standalone_minor_tasks=[task])
    _assert_matches_brute_force(instance)


def test_two_independent_far_apart_tasks_with_tight_horizon_needs_two_workers():
    # (0,0) and (9,9) are 18 apart -- one worker can't do both within a
    # horizon of 5, so a second worker is required.
    task1 = MinorTask(id="t1", cell=(0, 0), action=Action.PASS)
    task2 = MinorTask(id="t2", cell=(9, 9), action=Action.PASS)
    instance = Instance.compile(workers=_workers(0, 1, 2, 3), standalone_minor_tasks=[task1, task2], horizon=5)
    _assert_matches_brute_force(instance)


def test_precedence_chain_across_two_cells():
    task_a = MinorTask(id="a", cell=NW, action=Action.PASS)
    task_b = MinorTask(id="b", cell=NE, action=Action.PASS)
    instance = Instance.compile(
        workers=_workers(0, 1, 2),
        standalone_minor_tasks=[task_a, task_b],
        explicit_precedence=[("a", "b")],
    )
    _assert_matches_brute_force(instance)


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
    _assert_matches_brute_force(instance)


def test_infeasible_instance_is_reported_as_infeasible_by_both():
    # Well-formed infeasible case (major-form): a feed major with a
    # horizon of 1 -- the acquire and the feed each need their own turn,
    # so no schedule fits. Both the brute force and CP-SAT must report
    # "no valid solution".
    major = MajorTask(id="feed1", type="feed", cell=NW)
    instance = Instance.compile(
        workers=_workers(0, 1), major_tasks=[major], horizon=1
    )
    _assert_matches_brute_force(instance)


def test_zero_horizon_forces_exactly_two_workers():
    task1 = MinorTask(id="t1", cell=NW, action=Action.PASS)
    task2 = MinorTask(id="t2", cell=NE, action=Action.PASS)
    instance = Instance.compile(workers=_workers(0, 1, 2, 3), standalone_minor_tasks=[task1, task2], horizon=0)
    _assert_matches_brute_force(instance)
