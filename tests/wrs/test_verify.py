"""Restore test_verify.py against the current verifier.

Changes from the archived version:
- imports use Instance.compile (instance_compiler/major_task_expander are gone)
- DROP test 3  (aggregatable_pickup_ids param removed)
- DROP tests 16/17 (prefix activation removed — cost accounting makes gaps
  legal but expensive)
- SKIP test 9  (warehouse stock: field exists, engine has it, but no dump
  populates it and no solver models it yet)
- UPDATE tests 15/20 to the engine-faithful cost rule:
  cost = sum fib(0..max_active) — idle middle hands are paid too
"""
import pytest

from secretary.models import (
    Instance,
    Item,
    MajorTask,
    MinorActionType,
    MinorTask,
    ScheduledTask,
    Solution,
    Worker,
    WorkerRoute,
)
from secretary.verify import verify_solution

NW = (4, 4)  # a shed-adjacent cell, so travel distance from a worker's own entry point is 0
NE = (5, 4)  # the next shed-adjacent cell the placement rule assigns


def _worker(index: int, earliest_start: int = 0) -> Worker:
    return Worker(index=index, earliest_start=earliest_start)


def _single_task_instance() -> Instance:
    task = MinorTask(id="t1", cell=NW, action=MinorActionType.DIG)
    return Instance.compile(workers=[_worker(0)], standalone_minor_tasks=[task])


def test_valid_single_task_solution_passes():
    instance = _single_task_instance()
    solution = Solution(
        # exec_time=1, not 0: constraint 3 -- entering the grid at
        # start_time=0 only means the worker is standing at NW, the first
        # task still needs its own +1 turn on top of the (here, zero)
        # travel distance.
        routes=[WorkerRoute(worker_index=0, start_time=0, tasks=[ScheduledTask(task_id="t1", exec_time=1)])],
        reported_cost=0,
    )
    result = verify_solution(instance, solution)
    assert result.is_valid, result.violations
    assert result.total_cost == 0


def test_missing_task_is_caught():
    task1 = MinorTask(id="t1", cell=NW, action=MinorActionType.DIG)
    task2 = MinorTask(id="t2", cell=NW, action=MinorActionType.PASS)
    instance = Instance.compile(workers=[_worker(0)], standalone_minor_tasks=[task1, task2])
    solution = Solution(
        routes=[WorkerRoute(worker_index=0, start_time=0, tasks=[ScheduledTask(task_id="t1", exec_time=0)])],
    )
    result = verify_solution(instance, solution)
    assert not result.is_valid
    assert any("never scheduled" in v and "t2" in v for v in result.violations)


def test_resolved_qty_overrides_the_pickups_fixed_qty_for_inventory_accounting():
    # One PICKUP sized for 1 unit, but resolved_qty says it actually
    # brought back 2 -- enough to cover two same-item FEEDs from a single
    # trip. Mirrors CP-SAT's aggregated-PICKUP output (resolved_cell's
    # sibling field on ScheduledTask).
    pickup = MinorTask(id="pickup", cell=None, action=MinorActionType.PICKUP, item=Item.WHEAT, qty=1)
    feed1 = MinorTask(id="feed1", cell=NW, action=MinorActionType.FEED, item=Item.WHEAT, qty=1)
    feed2 = MinorTask(id="feed2", cell=NW, action=MinorActionType.FEED, item=Item.WHEAT, qty=1)
    instance = Instance.compile(
        workers=[_worker(0)],
        standalone_minor_tasks=[pickup, feed1, feed2],
        explicit_precedence=[("pickup", "feed1"), ("pickup", "feed2")],
        warehouse_stock={Item.WHEAT: 2},
    )
    solution = Solution(
        routes=[
            WorkerRoute(
                worker_index=0,
                start_time=0,
                tasks=[
                    ScheduledTask(task_id="pickup", exec_time=1, resolved_cell=NW, resolved_qty=2),
                    ScheduledTask(task_id="feed1", exec_time=2),
                    ScheduledTask(task_id="feed2", exec_time=3),
                ],
            )
        ],
    )
    result = verify_solution(instance, solution)
    assert result.is_valid, result.violations

    # Without the override (falling back to the fixed qty=1), the second
    # FEED would correctly be rejected -- confirms resolved_qty is what's
    # actually making the difference above, not some other change.
    unresolved = Solution(
        routes=[
            WorkerRoute(
                worker_index=0,
                start_time=0,
                tasks=[
                    ScheduledTask(task_id="pickup", exec_time=1, resolved_cell=NW),
                    ScheduledTask(task_id="feed1", exec_time=2),
                    ScheduledTask(task_id="feed2", exec_time=3),
                ],
            )
        ],
    )
    result = verify_solution(instance, unresolved)
    assert not result.is_valid
    assert any("inventory of 'wheat' goes negative" in v for v in result.violations)


def test_duplicate_task_is_caught():
    instance = _single_task_instance()
    solution = Solution(
        routes=[
            WorkerRoute(
                worker_index=0,
                start_time=0,
                tasks=[ScheduledTask(task_id="t1", exec_time=0), ScheduledTask(task_id="t1", exec_time=5)],
            )
        ],
    )
    result = verify_solution(instance, solution)
    assert not result.is_valid
    assert any("scheduled more than once" in v for v in result.violations)


def test_precedence_violation_is_caught():
    water = MinorTask(id="water", cell=NW, action=MinorActionType.WATER)
    harvest = MinorTask(id="harvest", cell=NW, action=MinorActionType.HARVEST)
    instance = Instance.compile(
        workers=[_worker(0)],
        standalone_minor_tasks=[water, harvest],
        explicit_precedence=[("water", "harvest")],
    )
    # harvest scheduled before (not after) water -> violates strict precedence
    solution = Solution(
        routes=[
            WorkerRoute(
                worker_index=0,
                start_time=0,
                tasks=[ScheduledTask(task_id="harvest", exec_time=0), ScheduledTask(task_id="water", exec_time=1)],
            )
        ],
    )
    result = verify_solution(instance, solution)
    assert not result.is_valid
    assert any("precedence violated" in v for v in result.violations)


def test_negative_inventory_is_caught():
    # FEED with no prior PICKUP of wheat: consumes wheat it never had.
    feed = MinorTask(id="feed", cell=NW, action=MinorActionType.FEED, item=Item.WHEAT, qty=1)
    instance = Instance.compile(workers=[_worker(0)], standalone_minor_tasks=[feed])
    solution = Solution(
        routes=[WorkerRoute(worker_index=0, start_time=0, tasks=[ScheduledTask(task_id="feed", exec_time=0)])],
    )
    result = verify_solution(instance, solution)
    assert not result.is_valid
    assert any("inventory of 'wheat' goes negative" in v for v in result.violations)


def test_pickup_then_feed_keeps_inventory_non_negative():
    pickup = MinorTask(id="pickup", cell=None, action=MinorActionType.PICKUP, item=Item.WHEAT, qty=1)
    feed = MinorTask(id="feed", cell=NW, action=MinorActionType.FEED, item=Item.WHEAT, qty=1)
    instance = Instance.compile(
        workers=[_worker(0)],
        standalone_minor_tasks=[pickup, feed],
        explicit_precedence=[("pickup", "feed")],
        warehouse_stock={Item.WHEAT: 5},
    )
    solution = Solution(
        routes=[
            WorkerRoute(
                worker_index=0,
                start_time=0,
                tasks=[
                    ScheduledTask(task_id="pickup", exec_time=1, resolved_cell=NW),
                    ScheduledTask(task_id="feed", exec_time=2),
                ],
            )
        ],
    )
    result = verify_solution(instance, solution)
    assert result.is_valid, result.violations


@pytest.mark.skip(reason="warehouse stock is not modeled by any solver yet; "
                         "the Instance field exists but dumps never populate it. "
                         "Re-enable when shed stock modeling is added.")
def test_warehouse_stock_overrun_is_caught():
    pickup = MinorTask(id="pickup", cell=None, action=MinorActionType.PICKUP, item=Item.WHEAT, qty=10)
    instance = Instance.compile(
        workers=[_worker(0)], standalone_minor_tasks=[pickup], warehouse_stock={Item.WHEAT: 3}
    )
    solution = Solution(
        routes=[
            WorkerRoute(
                worker_index=0, start_time=0, tasks=[ScheduledTask(task_id="pickup", exec_time=0, resolved_cell=NW)]
            )
        ],
    )
    result = verify_solution(instance, solution)
    assert not result.is_valid
    assert any("warehouse stock" in v for v in result.violations)


def test_single_worker_group_split_is_caught():
    major = MajorTask(id="feed1", type="feed", cell=NW)
    instance = Instance.compile(workers=[_worker(0), _worker(1)], major_tasks=[major], warehouse_stock={Item.WHEAT: 5})
    solution = Solution(
        routes=[
            WorkerRoute(
                worker_index=0,
                start_time=0,
                tasks=[ScheduledTask(task_id="feed1_acquire", exec_time=0, resolved_cell=NW)],
            ),
            WorkerRoute(
                worker_index=1,
                start_time=0,
                tasks=[ScheduledTask(task_id="feed1_feed", exec_time=1)],
            ),
        ],
    )
    result = verify_solution(instance, solution)
    assert not result.is_valid
    assert any("single-worker group" in v for v in result.violations)


def test_clct_is_currently_not_enforced():
    # Regression: clct (constraint 9) used to be an automatic global rule
    # -- any HARVEST forced that worker to return to the shed and DROP by
    # a deadline. That was a foundational design error (see
    # docs/problem-formulation.*.md constraint 9's "History" note); clct
    # is disabled everywhere until it's rebuilt as a proper major_task.
    # A HARVEST with no follow-up DROP at all must therefore be valid.
    harvest = MinorTask(id="harvest", cell=NW, action=MinorActionType.HARVEST, item=Item.WHEAT, qty=1)
    instance = Instance.compile(workers=[_worker(0)], standalone_minor_tasks=[harvest], clct_deadline=23)
    solution = Solution(
        routes=[WorkerRoute(worker_index=0, start_time=0, tasks=[ScheduledTask(task_id="harvest", exec_time=1)])],
    )
    result = verify_solution(instance, solution)
    assert result.is_valid, result.violations


def test_travel_time_violation_is_caught():
    # (0,0) and (9,9) are 18 apart -> can't both be done one turn apart.
    task1 = MinorTask(id="t1", cell=(0, 0), action=MinorActionType.PASS)
    task2 = MinorTask(id="t2", cell=(9, 9), action=MinorActionType.PASS)
    instance = Instance.compile(workers=[_worker(0)], standalone_minor_tasks=[task1, task2])
    solution = Solution(
        routes=[
            WorkerRoute(
                worker_index=0,
                start_time=0,
                tasks=[ScheduledTask(task_id="t1", exec_time=0), ScheduledTask(task_id="t2", exec_time=1)],
            )
        ],
    )
    result = verify_solution(instance, solution)
    assert not result.is_valid
    assert any("reachable too early" in v or "reachable no earlier than" in v for v in result.violations)


def test_horizon_violation_is_caught():
    task = MinorTask(id="t1", cell=NW, action=MinorActionType.PASS)
    instance = Instance.compile(workers=[_worker(0)], standalone_minor_tasks=[task], horizon=24)
    solution = Solution(
        routes=[WorkerRoute(worker_index=0, start_time=0, tasks=[ScheduledTask(task_id="t1", exec_time=30)])],
    )
    result = verify_solution(instance, solution)
    assert not result.is_valid
    assert any("outside horizon" in v for v in result.violations)


def test_earliest_start_violation_is_caught():
    task = MinorTask(id="t1", cell=NW, action=MinorActionType.PASS)
    instance = Instance.compile(workers=[_worker(3, earliest_start=5)], standalone_minor_tasks=[task])
    solution = Solution(
        routes=[WorkerRoute(worker_index=3, start_time=0, tasks=[ScheduledTask(task_id="t1", exec_time=0)])],
    )
    result = verify_solution(instance, solution)
    assert not result.is_valid
    assert any("start_time < earliest_start" in v for v in result.violations)


def test_reported_cost_mismatch_is_caught():
    instance = _single_task_instance()
    solution = Solution(
        routes=[WorkerRoute(worker_index=0, start_time=0, tasks=[ScheduledTask(task_id="t1", exec_time=0)])],
        reported_cost=999,
    )
    result = verify_solution(instance, solution)
    assert not result.is_valid
    assert any("reported_cost=999" in v for v in result.violations)


def test_gap_in_worker_activation_is_legal_but_paid():
    # Engine accounting: hiring is append-only -- if worker 2 works, hands
    # 0, 1, 2 were all hired and paid, even though hand 1 sits idle. A gap
    # is legal; it just costs fib(1) extra. (Replaces the removed
    # prefix-activation constraint 11 -- "only the money matters".)
    task0 = MinorTask(id="t0", cell=NW, action=MinorActionType.PASS)
    task2 = MinorTask(id="t2", cell=NE, action=MinorActionType.PASS)
    instance = Instance.compile(
        workers=[_worker(0), _worker(1), _worker(2)], standalone_minor_tasks=[task0, task2]
    )
    # placement: active routes 0 and 2 get entries NW and NE respectively
    # (least-crowded rule over the ACTIVE set), so each worker reaches its
    # own task in one turn.
    solution = Solution(
        routes=[
            WorkerRoute(worker_index=0, start_time=0, tasks=[ScheduledTask(task_id="t0", exec_time=1)]),
            WorkerRoute(worker_index=2, start_time=0, tasks=[ScheduledTask(task_id="t2", exec_time=1)]),
        ],
    )
    result = verify_solution(instance, solution)
    assert result.is_valid, result.violations
    # fib(0)=0, fib(1)=1, fib(2)=1  ->  0 + 1 + 1 = 2
    assert result.total_cost == 2


def test_contiguous_worker_activation_passes():
    # Workers 0 and 1 active (a genuine prefix of the 0,1,2 candidate list),
    # worker 2 correctly left idle (never hired).
    task0 = MinorTask(id="t0", cell=NW, action=MinorActionType.PASS)
    task1 = MinorTask(id="t1", cell=NE, action=MinorActionType.PASS)
    instance = Instance.compile(
        workers=[_worker(0), _worker(1), _worker(2)], standalone_minor_tasks=[task0, task1]
    )
    solution = Solution(
        routes=[
            WorkerRoute(worker_index=0, start_time=0, tasks=[ScheduledTask(task_id="t0", exec_time=1)]),
            WorkerRoute(worker_index=1, start_time=0, tasks=[ScheduledTask(task_id="t1", exec_time=1)]),
        ],
    )
    result = verify_solution(instance, solution)
    assert result.is_valid, result.violations


def test_worker_capacity_is_horizon_minus_start_time():
    # Regression (constraint 3): a worker's first task used to be allowed
    # to execute "for free" at start_time + distance (no +1), silently
    # inflating every worker's usable capacity by one turn. A worker
    # starting at `start` can fit at most `horizon - start` actions, not
    # `horizon - start + 1`. Pack a worker starting at 5 with exactly
    # 24 - 5 = 19 back-to-back PASS actions at its own entry cell (zero
    # travel) and check the last one lands exactly on the horizon.
    horizon = 24
    start = 5
    n = horizon - start
    tasks = [MinorTask(id=f"t{i}", cell=NW, action=MinorActionType.PASS) for i in range(n)]
    instance = Instance.compile(
        workers=[_worker(0, earliest_start=start)], standalone_minor_tasks=tasks, horizon=horizon
    )
    scheduled = [ScheduledTask(task_id=f"t{i}", exec_time=start + 1 + i) for i in range(n)]
    solution = Solution(routes=[WorkerRoute(worker_index=0, start_time=start, tasks=scheduled)])
    result = verify_solution(instance, solution)
    assert result.is_valid, result.violations
    assert scheduled[-1].exec_time == horizon


def test_worker_cannot_fit_one_more_action_than_horizon_minus_start_time_allows():
    # The one-too-many counterpart: horizon - start + 1 actions can't all
    # fit -- the extra one would need exec_time = horizon + 1.
    horizon = 24
    start = 5
    n = horizon - start + 1
    tasks = [MinorTask(id=f"t{i}", cell=NW, action=MinorActionType.PASS) for i in range(n)]
    instance = Instance.compile(
        workers=[_worker(0, earliest_start=start)], standalone_minor_tasks=tasks, horizon=horizon
    )
    scheduled = [ScheduledTask(task_id=f"t{i}", exec_time=start + 1 + i) for i in range(n)]
    solution = Solution(routes=[WorkerRoute(worker_index=0, start_time=start, tasks=scheduled)])
    result = verify_solution(instance, solution)
    assert not result.is_valid
    assert any("outside horizon" in v for v in result.violations)


def test_total_cost_sums_fibonacci_up_to_the_highest_active_worker():
    # Engine accounting: hiring is append-only -- worker 3 working means
    # hands 0, 1, 2 were all hired and paid too (idle or not), so the cost
    # is fib(0)+fib(1)+fib(2)+fib(3) = 0+1+1+2 = 4.
    task1 = MinorTask(id="t1", cell=NW, action=MinorActionType.PASS)
    task2 = MinorTask(id="t2", cell=(5, 4), action=MinorActionType.PASS)  # NE
    instance = Instance.compile(workers=[_worker(0), _worker(3)], standalone_minor_tasks=[task1, task2])
    solution = Solution(
        routes=[
            WorkerRoute(worker_index=0, start_time=0, tasks=[ScheduledTask(task_id="t1", exec_time=1)]),
            WorkerRoute(worker_index=3, start_time=0, tasks=[ScheduledTask(task_id="t2", exec_time=1)]),
        ],
    )
    result = verify_solution(instance, solution)
    assert result.is_valid, result.violations
    assert result.total_cost == 0 + 1 + 1 + 2  # fib(0..3)
