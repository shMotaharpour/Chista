"""A small library of hand-crafted instances whose optimal cost is known by
manual/logical computation. These are the correctness ground truth for the
solvers built in later milestones: their answer's cost must equal
`optimal_cost` on every case here, and `example_optimal_solution` (when
given) must pass `day.verify.verify_solution` and match that cost
exactly (checked by tests/test_hand_solved_fixtures.py, independent of any
solver).

See docs/problem-formulation.en.md, "Phase 4" -> "b) Confirmed need: tests
validating solver-solution optimality", item 2.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from day.models import MajorTask
from day.models import (
    WAREHOUSE_ENTRY_CELLS,
    Instance,
    Item,
    Action,
    MinorTask,
    ScheduledTask,
    Solution,
    Worker,
    WorkerRoute,
)

NW = WAREHOUSE_ENTRY_CELLS["NW"]


@dataclass
class HandSolvedCase:
    name: str
    instance: Instance
    optimal_cost: int
    example_optimal_solution: Optional[Solution] = None


def _workers(*indices: int, earliest_start: int = 0) -> list[Worker]:
    return [Worker(index=i, earliest_start=earliest_start) for i in indices]


def _single_task_case() -> HandSolvedCase:
    # One task, one candidate worker. The cheapest possible worker (index 0,
    # cost 0) always suffices for a single task at a shed cell.
    task = MinorTask(id="t1", cell=NW, action=Action.PASS)
    instance = Instance.compile(workers=_workers(0, 1, 2), standalone_minor_tasks=[task])
    solution = Solution(
        routes=[WorkerRoute(worker_index=0, start_time=0, tasks=[ScheduledTask(task_id="t1", exec_time=1)])],
        reported_cost=0,
    )
    return HandSolvedCase("single_task_needs_only_worker_0", instance, optimal_cost=0, example_optimal_solution=solution)


def _two_independent_tasks_case() -> HandSolvedCase:
    # Two independent (no precedence) tasks at the same cell: a single
    # worker can do both, one turn apart, so the optimum is still worker 0
    # alone (cost 0) even though several workers are offered as candidates.
    task1 = MinorTask(id="t1", cell=NW, action=Action.PASS)
    task2 = MinorTask(id="t2", cell=NW, action=Action.PASS)
    instance = Instance.compile(workers=_workers(0, 1, 2), standalone_minor_tasks=[task1, task2])
    solution = Solution(
        routes=[
            WorkerRoute(
                worker_index=0,
                start_time=0,
                tasks=[ScheduledTask(task_id="t1", exec_time=1), ScheduledTask(task_id="t2", exec_time=2)],
            )
        ],
        reported_cost=0,
    )
    return HandSolvedCase(
        "two_independent_tasks_still_need_only_one_worker", instance, optimal_cost=0, example_optimal_solution=solution
    )


def _wet_harvst_single_worker_case() -> HandSolvedCase:
    # WATER -> HARVEST at the same cell: splittable between workers per the
    # formulation, but one worker is always cheapest and always feasible.
    #
    # History: this used to also require a standalone DROP (precedence-tied
    # to the HARVEST) because a HARVEST used to automatically trigger the
    # clct constraint (return to the shed and DROP by a deadline). clct is
    # currently disabled entirely (a foundational design error -- see
    # docs/problem-formulation.*.md constraint 9's "History" note), so
    # that DROP is no longer required; the fixture is simplified to just
    # the wet_harvst chain itself.
    major = MajorTask(id="m1", type="wet_harvst", cell=NW, harvested_item=Item.WHEAT, harvested_qty=3)
    instance = Instance.compile(workers=_workers(0, 1, 2), major_tasks=[major])
    solution = Solution(
        routes=[
            WorkerRoute(
                worker_index=0,
                start_time=0,
                tasks=[
                    ScheduledTask(task_id="m1_water", exec_time=1),
                    ScheduledTask(task_id="m1_harvest", exec_time=2),
                ],
            )
        ],
        reported_cost=0,
    )
    return HandSolvedCase(
        "wet_harvst_chain_needs_only_one_worker", instance, optimal_cost=0, example_optimal_solution=solution
    )


def _feed_single_worker_chain_case() -> HandSolvedCase:
    # `feed`'s single-worker requirement constrains *which* worker does
    # both steps, but never forces more workers to exist: worker 0 alone
    # can pick up the wheat and feed with it. Acquisition is always PICKUP
    # now -- see major_task_expander's "History" note.
    major = MajorTask(id="feed1", type="feed", cell=NW)
    instance = Instance.compile(workers=_workers(0, 1), major_tasks=[major], warehouse_stock={Item.WHEAT: 5})
    solution = Solution(
        routes=[
            WorkerRoute(
                worker_index=0,
                start_time=0,
                tasks=[
                    ScheduledTask(task_id="feed1_acquire", exec_time=1, resolved_cell=NW),
                    ScheduledTask(task_id="feed1_feed", exec_time=2),
                ],
            )
        ],
        reported_cost=0,
    )
    return HandSolvedCase(
        "feed_single_worker_chain_needs_only_worker_0", instance, optimal_cost=0, example_optimal_solution=solution
    )


def _one_step_horizon_forces_two_workers_case() -> HandSolvedCase:
    # With horizon=1, a worker starting at t=0 has room for exactly one
    # action (constraint 3: its first task needs exec_time >= start_time +
    # 1 + distance, so t=1 is the only value that still fits the horizon).
    # A second task at the same instant can't be squeezed onto that same
    # worker's route (its own next task would need t=2, past the horizon)
    # -- exactly 2 workers are required. The cheapest 2 are workers 0 and 1
    # (Fib(0)+Fib(1) = 0+1 = 1); any other pair costs more. The two tasks
    # sit at NW and NE respectively so each worker's own entry cell (per
    # the placement rule, simultaneous arrivals go NW then NE) reaches its
    # task at distance 0.
    #
    # History: this used to be horizon=0 with both tasks pinned to t=0 --
    # that assumed a worker's first action was free (no +1 turn), which was
    # exactly the off-by-one a user's manual audit caught (see docs
    # constraint 3). Under the corrected accounting, horizon=0 gives a
    # worker zero usable actions, not one, so the instance is now expressed
    # at horizon=1 instead; the "one action isn't enough for two tasks"
    # property this fixture exists to test is unchanged.
    task1 = MinorTask(id="t1", cell=NW, action=Action.PASS)
    task2 = MinorTask(id="t2", cell=WAREHOUSE_ENTRY_CELLS["NE"], action=Action.PASS)
    instance = Instance.compile(workers=_workers(0, 1, 2, 3), standalone_minor_tasks=[task1, task2], horizon=1)
    solution = Solution(
        routes=[
            WorkerRoute(worker_index=0, start_time=0, tasks=[ScheduledTask(task_id="t1", exec_time=1)]),
            WorkerRoute(worker_index=1, start_time=0, tasks=[ScheduledTask(task_id="t2", exec_time=1)]),
        ],
        reported_cost=1,
    )
    return HandSolvedCase(
        "one_step_horizon_forces_exactly_two_cheapest_workers", instance, optimal_cost=1, example_optimal_solution=solution
    )


HAND_SOLVED_CASES: list[HandSolvedCase] = [
    _single_task_case(),
    _two_independent_tasks_case(),
    _wet_harvst_single_worker_case(),
    _feed_single_worker_chain_case(),
    _one_step_horizon_forces_two_workers_case(),
]
