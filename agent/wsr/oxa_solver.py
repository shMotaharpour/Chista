"""
OXA solver (Refactored: Spatial Greedy Dispatcher)

A blazing-fast, pure Python heuristic solver optimized for dense grids and sub-second 
execution times. Based on Capacity-Constrained Nearest Neighbor and Setup Costs.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Optional

from agent.world.model import UnitAction

from .distances import assign_entry_cells, manhattan
from .fibonacci import fibonacci_cost
from .models import (
    CONSUME_ACTIONS,
    Action,
    WAREHOUSE_ENTRY_CELLS,
    Worker,
    Cell,
    Instance,
    MinorTask,
    ScheduledTask,
    Solution,
    WorkerRoute,
)
from .verify import verify_solution


class InfeasibleInputError(ValueError):
    """The instance is provably infeasible before the model is even built
    (e.g. more of an item is picked up across all tasks than the shed
    holds) -- a static property of the task list itself."""
    pass

@dataclass
class OxaConfig:
    """What the caller may tune.

    `validate` runs the independent verifier on the solver's own answer and reports
    INVALID_SOLUTION rather than returning an unverified schedule; it is the expensive half of
    the call and is meant to be switched off on the agent's path, where the answer is checked
    elsewhere.
    """

    min_workers: int = 1
    worker_pool_cap: Optional[int] = None
    validate: bool = True


@dataclass
class OxaResult:
    status: str
    solution: Optional[Solution] = None
    wall_time_seconds: float = 0.0
    matched_lower_bound: bool = False


def _build_worker_route(
    worker_index: int,
    earliest_start: int,
    entry_cell: Cell,
    remaining_targets: set[str],
    done_targets: set[str],
    exec_times: dict[str, int],
    target_preds: dict[str, list[str]],
    tasks_by_id: dict[str, MinorTask],
    target_needs_item: dict[str, str],
    item_to_pickups: dict[str, list[str]],
    horizon: int,
    instance: Optional[Instance] = None,
) -> list[ScheduledTask]:
    """Greedily fills a single worker's shift using Nearest Neighbor + Setup Costs."""
    current_t = earliest_start
    current_pos = entry_cell
    worker_items: set[str] = set()
    
    route_task_ids: list[str] = []
    local_exec_times: dict[str, int] = {}
    time_offset = 0
    
    acquire_partner: dict[str, str] = {}
    for group in instance.single_worker_groups:
        if len(group) == 2 and tasks_by_id[group[0]].action == UnitAction.PICKUP:
            acquire_partner[group[1]] = group[0]

    successor_gaps = _successor_gaps(instance, tasks_by_id)

    while True:
        # The tail table is constant across this iteration: the candidate
        # set only changes when a task is committed, and a commit ends it.
        pending = frozenset(remaining_targets)
        tail_turns: dict[str, int] = {}
        available = [
            tid for tid in remaining_targets 
            if all(p in done_targets or p in route_task_ids for p in target_preds[tid])
        ]
        
        best_task_id = None
        best_key = None
        best_actual_exec = -1
        best_setup = 0
        best_item = None
        
        for tid in available:
            task = tasks_by_id[tid]
            needed_item = target_needs_item.get(tid)
            
            setup = 1 if (needed_item and needed_item not in worker_items) else 0
            task_cell = task.cell if task.cell else WAREHOUSE_ENTRY_CELLS['NW']
            dist = manhattan(current_pos, task_cell)
            
            proposed_t = current_t + setup + 1 + dist
            
            ready_t = 0
            for p in target_preds[tid]:
                if p in route_task_ids:
                    ready_t = max(ready_t, local_exec_times[p] + setup + 1)
                else:
                    ready_t = max(ready_t, exec_times[p] + 1)
                    
            actual_exec = max(proposed_t, ready_t)
            
            if actual_exec > horizon:
                continue
            # Do not commit a task at a time where the rest of its own
            # precedence chain can no longer run inside the horizon: the
            # emitted exec_time becomes a hard global lower bound for its
            # successors (`ready_t = exec_times[p] + 1`), so no later
            # worker can rescue the chain and the leftover target is read
            # as INFEASIBLE with idle workers sitting right there. The gap
            # `_successor_gaps` charges an edge is a lower bound on any
            # completion of it, so skipping the task here can never lose a
            # schedule that fits.
            tail = _unfinished_tail_turns(tid, pending, successor_gaps, tail_turns)
            if actual_exec + tail > horizon:
                continue
                
            cost = actual_exec - current_t
            tie_breaker = -manhattan(entry_cell, task_cell)
            # A total order on the candidates: cheapest first, then the entry
            # cell's travel preference, then the MORE CONSTRAINED task (the
            # longer unfinished tail -- serving it late is what strands a
            # chain, fuzz seed 307), then the task id. The id makes the
            # winner independent of the order the candidates were visited
            # in, so the verdict is a function of the instance alone.
            key = (cost, tie_breaker, -tail, tid)
            if best_key is None or key < best_key:
                best_key = key
                best_task_id = tid
                best_actual_exec = actual_exec
                best_setup = setup
                best_item = needed_item
                
        if not best_task_id:
            break
            
        if best_setup:
            worker_items.add(best_item)
            time_offset += 1
            current_t += 1
            for rt in route_task_ids:
                local_exec_times[rt] += 1
            # the setup turn IS the acquire: insert the paired acquire into
            # the route here; it occupies the turn just consumed by setup
            acq_id = acquire_partner.get(best_task_id)
            if acq_id and acq_id not in route_task_ids:
                # insert immediately BEFORE best_task_id so the emitted
                # order stays chronological (acquire at the setup turn)
                idx = route_task_ids.index(best_task_id) if best_task_id in route_task_ids else len(route_task_ids)
                route_task_ids.insert(idx, acq_id)
                local_exec_times[acq_id] = current_t - 1
        route_task_ids.append(best_task_id)
        local_exec_times[best_task_id] = best_actual_exec
        current_t = best_actual_exec
        current_pos = tasks_by_id[best_task_id].cell or WAREHOUSE_ENTRY_CELLS['NW']
        
        remaining_targets.remove(best_task_id)

    scheduled_tasks: list[ScheduledTask] = []

    # ---- emission (verifier-consistent) -------------------------------
    # Preload phase: the worker stands at its entry cell and performs one
    # PICKUP per needed item (turns es+1 .. es+p). All pickups happen at
    # the SAME entry cell, so consecutive travel is zero.
    # The greedy's local_exec_times already include one 'setup' turn per
    # new item, so the route tasks keep their greedy times.
    acquire_partner: dict[str, str] = {}
    for group in instance.single_worker_groups:
        if len(group) == 2 and tasks_by_id[group[0]].action == UnitAction.PICKUP:
            acquire_partner[group[1]] = group[0]

    first_consume_time: dict[str, int] = {}
    route_demand: dict[str, int] = {}
    for tid in route_task_ids:
        task = tasks_by_id[tid]
        if task.action in CONSUME_ACTIONS and task.item is not None:
            route_demand[task.item] = route_demand.get(task.item, 0) + task.n
            if task.item not in first_consume_time:
                first_consume_time[task.item] = local_exec_times[tid]

    t = earliest_start + 1
    emitted_acquires: set[str] = set()
    for item in sorted(first_consume_time, key=lambda i: first_consume_time[i]):
        acq_id = None
        for tid2 in route_task_ids:
            t2 = tasks_by_id[tid2]
            if (t2.action in CONSUME_ACTIONS and t2.item == item
                    and tid2 in acquire_partner):
                acq_id = acquire_partner[tid2]
                break
        if acq_id is None:
            continue
        scheduled_tasks.append(ScheduledTask(
            task_id=acq_id,
            exec_time=t,
            resolved_cell=entry_cell,
            resolved_n=route_demand[item],
        ))
        emitted_acquires.add(acq_id)
        t += 1

    for tid in route_task_ids:
        if tid in emitted_acquires:
            continue
        task = tasks_by_id[tid]
        scheduled_tasks.append(ScheduledTask(
            task_id=tid,
            exec_time=local_exec_times[tid],
            resolved_cell=task.cell or entry_cell,
            resolved_n=task.n,
        ))
        exec_times[tid] = local_exec_times[tid]
        done_targets.add(tid)

    return scheduled_tasks


def _chain_cost_for_group(group: list[str], precedence: list[tuple[str, str]], tasks_by_id: dict[str, MinorTask]) -> int:
    """Workload of one single_worker_group, as an indivisible block (see
    module docstring). Exact for the current domain's size-2 groups
    (feed/frtz/place_animal); falls back to a plain task count for any
    larger group, which the design doc flags as a still-open
    generalization (a shortest-path-respecting-precedence over the
    group's members) if one is ever needed."""
    if len(group) != 2:
        return len(group)
    a, b = group
    order = (a, b)
    for pred, succ in precedence:
        if pred == a and succ == b:
            order = (a, b)
            break
        if pred == b and succ == a:
            order = (b, a)
            break
    first, second = order
    d = _min_dist(_cell_options(tasks_by_id[first]), _cell_options(tasks_by_id[second]))
    return 2 + d



def _min_dist(a_options: list[Cell], b_options: list[Cell]) -> int:
    return min(manhattan(a, b) for a in a_options for b in b_options)


_ENTRY_CELLS = frozenset(WAREHOUSE_ENTRY_CELLS.values())


def _cell_options(task: MinorTask) -> list[Cell]:
    if task.cell is not None:
        return [task.cell]
    return _ENTRY_CELLS



def _successor_gaps(
    instance: Instance, tasks_by_id: dict[str, MinorTask]
) -> dict[str, list[tuple[str, int]]]:
    """`instance.precedence` inverted over the target tasks, each edge carrying
    the *minimum turns that must separate the two tasks*.

    Two tasks in one `single_worker_group` are executed by the same worker, so
    the successor also pays the travel from the predecessor's cell
    (`1 + dist`). Every other edge is a plain precedence pair that any worker
    may serve: only `1` turn is forced (precedence is strict, one action per
    worker per turn) — charging travel there would reject commits another
    worker can finish, which is its own false-INFEASIBLE (see docs/F057).
    """
    gaps: dict[str, list[tuple[str, int]]] = {tid: [] for tid in instance.target_tasks}
    for pred, succ in instance.precedence:
        if pred not in gaps or succ not in gaps:
            continue
        gap = 1
        pred_group = instance.group_of.get(pred)
        if pred_group is not None and pred_group == instance.group_of.get(succ):
            gap += _min_dist(_cell_options(tasks_by_id[pred]), _cell_options(tasks_by_id[succ]))
        gaps[pred].append((succ, gap))
    return gaps


def _unfinished_tail_turns(
    tid: str,
    pending: frozenset[str],
    successor_gaps: dict[str, list[tuple[str, int]]],
    memo: dict[str, int],
) -> int:
    """Turns that must still fit *after* `tid` for every unfinished descendant
    of `tid` to run: the longest path through the successor DAG, each hop
    costing the minimum separation `_successor_gaps` assigns that edge.

    Every legal completion needs at least this much time, so refusing to
    commit a task that cannot afford it is lossless -- see docs/F057 (the
    false-INFEASIBLE defect).
    """
    cached = memo.get(tid)
    if cached is not None:
        return cached
    memo[tid] = 0  # breaks any accidental cycle; the DAG is validated upstream
    longest = 0
    for succ, gap in successor_gaps.get(tid, ()):
        if succ not in pending:
            continue
        longest = max(longest, gap + _unfinished_tail_turns(succ, pending, successor_gaps, memo))
    memo[tid] = longest
    return longest


def compute_lower_bound(instance: Instance) -> int:
    tasks_by_id = {t.id: t for t in instance.minor_tasks}
    grouped_ids = {tid for group in instance.single_worker_groups for tid in group}
    workload = sum(
        _chain_cost_for_group(group, instance.precedence, tasks_by_id) for group in instance.single_worker_groups
    )
    workload += sum(1 for t in instance.minor_tasks if t.id not in grouped_ids)
    # /horizon, not /(horizon+1): a worker starting at t=0 fits actions
    # completing at t=1..horizon -- horizon of them, not horizon+1 (see
    # module docstring's "Lower bound" note and docs constraint 3).
    return max(1, math.ceil(workload / instance.horizon)) if instance.horizon > 0 else max(1, workload)


# ---------------------------------------------------------------- worker mechanics


class _WorkerState:
    __slots__ = ("worker", "entry_cell", "last_cell", "last_time", "balance", "tasks")

    def __init__(self, worker: Worker, entry_cell: Cell):
        self.worker = worker
        self.entry_cell = entry_cell
        self.last_cell = entry_cell
        # History: was `earliest_start - 1`, which let the first task
        # execute for free at `earliest_start + dist` (no +1 for its own
        # turn) -- the same off-by-one fixed in greedy_decoder.py's
        # _WorkerState (see its docstring for the full story). Priming to
        # `earliest_start` (not `- 1`) makes the uniform
        # "next = last_time + 1 + dist" formula in `_earliest_time` correct
        # for the first task too.
        self.last_time = worker.earliest_start
        self.balance: dict[Item, int] = {}
        self.tasks: list[ScheduledTask] = []



def _validate_item_sources(instance: Instance) -> None:
    """Static check: every consume task must have an item source — either
    a paired acquire in its single-worker group, or any PICKUP task of
    the same item somewhere in the instance (preload-able). Raises
    InfeasibleInputError otherwise."""
    pickup_items = {t.item for t in instance.minor_tasks
                    if t.action == UnitAction.PICKUP and t.item is not None}
    for t in instance.minor_tasks:
        if t.action not in CONSUME_ACTIONS or t.item is None:
            continue
        has_partner = any(
            t.id in g and any(
                instance.tasks_by_id[tid].action == UnitAction.PICKUP and
                instance.tasks_by_id[tid].item == t.item
                for tid in g if tid != t.id
            )
            for g in instance.single_worker_groups
        )
        if not has_partner and t.item not in pickup_items:
            raise InfeasibleInputError(
                f"consume task {t.id!r} needs {t.item.value!r} but no PICKUP "
                f"of that item exists anywhere in the instance"
            )

def _validate_preloaded_pickups_have_no_predecessor(instance: Instance) -> None:
    """A preloaded pickup may not itself have a predecessor.

    `Instance.compile` marks a pickup *aggregatable* when it precedes its own
    consume inside a `single_worker_group` (models.py step 5): the solver then
    executes it as the setup turn at the head of a route and emits it without
    ever consulting the precedence graph -- such a pickup is not a target task,
    so `target_preds` does not carry the edge either. An edge *into* it is
    therefore unrepresentable here: the greedy commits it at its own hour and
    `verify_solution` rejects the day as `precedence violated`. `cpsat_solver`
    models the edge, so the input is not refusable at `Instance.compile`
    without taking that ability away; the formulation's limit is raised here
    instead of being answered with an unusable day (docs/F057, fifth class).
    """
    for pred, succ in instance.precedence:
        if succ in instance.aggregatable_pickups:
            raise InfeasibleInputError(
                f"precedence {pred!r} -> {succ!r} puts a task before a "
                f"preloaded pickup: {succ!r} is emitted as the setup turn at "
                f"the head of its route, so it cannot follow another task, "
                f"and this solver answers INVALID_SOLUTION when asked to"
            )

# How many times `solve_oxa` re-derives the entry-cell assignment from the
# routed set and re-runs the dispatch (docs/F057, third class). One dispatch is
# sub-millisecond on this domain's shapes; the audit converges in one or two
# attempts, and the cap keeps a pathological instance bounded.
_PLACEMENT_ATTEMPTS = 4

# How many dispatch passes each placement attempt runs (docs/F057, fourth
# class). Every pass covers the hands that have no route yet, so a further pass
# only helps when a later hand committed a predecessor for an earlier one;
# three covers every shape the audit produces.
_DISPATCH_ROUNDS = 3


def _dispatch(
    instance: Instance,
    candidates: list[Worker],
    entry_assignments: dict[int, str],
    remaining_targets: set[str],
    done_targets: set[str],
    exec_times: dict[str, int],
) -> list[WorkerRoute]:
    """One spatial-greedy pass over `candidates`, mutating the shared state.

    `remaining_targets`, `done_targets` and `exec_times` carry across passes:
    what an earlier pass committed is what makes a successor available to a
    hand in a later one. The caller owns `entry_assignments` and owns running
    the passes -- see `_dispatch_rounds` and the fixed point in `solve_oxa`.
    """
    routes: list[WorkerRoute] = []

    for worker in candidates:
        if not remaining_targets:
            break

        entry_cell = WAREHOUSE_ENTRY_CELLS[entry_assignments[worker.index]]
        sched_tasks = _build_worker_route(
            worker_index=worker.index,
            earliest_start=worker.earliest_start,
            entry_cell=entry_cell,
            remaining_targets=remaining_targets,
            done_targets=done_targets,
            exec_times=exec_times,
            target_preds=instance.target_preds,
            tasks_by_id=instance.tasks_by_id,
            target_needs_item=instance.target_needs_item,
            item_to_pickups=instance.item_to_pickups,
            horizon=instance.horizon,
            instance=instance,
        )

        if sched_tasks:
            routes.append(WorkerRoute(
                worker_index=worker.index,
                start_time=worker.earliest_start,
                tasks=sched_tasks,
                start_cell=WAREHOUSE_ENTRY_CELLS[entry_assignments[worker.index]],
            ))

    return routes


def _dispatch_rounds(
    instance: Instance,
    candidates: list[Worker],
    entry_assignments: dict[int, str],
) -> tuple[list[WorkerRoute], set[str]]:
    """The dispatch: as many passes as there are idle hands left to try.

    A task is only available once its predecessors are committed, and one pass
    gives every hand a single chance in index order -- so a day whose legal
    arrangement serves a successor from a lower-indexed hand while a LATER hand
    serves its predecessor is missed (docs/F057, fourth class: hand 0 can reach
    the successor but not the predecessor, hand 1 takes the predecessor, and
    hand 0's pass is already over). The pass therefore repeats for the hands
    that still have no route; a routed hand's route is never touched, because
    extending it would reroute it and neither the emission phase nor the
    verifier's per-route checks are built for that. Returns the routes and the
    targets left unscheduled (empty means the day was covered).
    """
    remaining_targets: set[str] = set(instance.target_tasks)
    done_targets: set[str] = set()
    exec_times: dict[str, int] = {}

    routes: list[WorkerRoute] = []
    for _round in range(_DISPATCH_ROUNDS):
        if not remaining_targets:
            break
        idle = [w for w in candidates if w.index not in {r.worker_index for r in routes}]
        if not idle:
            break
        routes.extend(_dispatch(
            instance, idle, entry_assignments, remaining_targets, done_targets, exec_times))

    return routes, remaining_targets


def solve_oxa(instance: Instance, config: OxaConfig = OxaConfig()) -> OxaResult:
    _validate_item_sources(instance)
    _validate_preloaded_pickups_have_no_predecessor(instance)

    # ---- static input validation: warehouse stock overrun ----------------
    if instance.warehouse_stock:
        picked: dict[Item, int] = {}
        for _t in instance.minor_tasks:
            if _t.action == UnitAction.PICKUP and _t.item is not None:
                picked[_t.item] = picked.get(_t.item, 0) + _t.n
        for _item, _amount in picked.items():
            _stock = instance.warehouse_stock.get(_item, 0)
            if _amount > _stock:
                raise InfeasibleInputError(
                    f"warehouse stock for {_item.value!r} is {_stock}, "
                    f"but {_amount} is picked up across all tasks"
                )
    start = time.perf_counter()
    
    if not instance.tasks_by_id:
        return OxaResult(status="OPTIMAL", solution=Solution(routes=[], reported_cost=0), matched_lower_bound=True)

    # The pool is the LOWEST-indexed hands, not the order the caller listed
    # them in: `cpsat_solver` sorts by index before applying the cap, and
    # verify.py's COST ACCOUNTING is the engine's append-only prefix by index,
    # so a cap of 1 offers hand 0. Capping the given order instead let OXA
    # route a high-indexed hand (and pay its payroll) for a day the oracle
    # does with hand 0 at cost 0.
    candidates = sorted(instance.workers, key=lambda worker: worker.index)
    if instance.worker_pool_size is not None:
        candidates = candidates[: instance.worker_pool_size]
    if config.worker_pool_cap is not None:
        candidates = candidates[: config.worker_pool_cap]
        
    if not candidates:
        return OxaResult(status="INFEASIBLE", wall_time_seconds=0.0)

    # 1. The placement fixed point, over the repeated dispatch.
    #
    # `verify_solution` recomputes the entry cells over the workers that are
    # actually ROUTED (constraint 10, `_check_placement`), not over every
    # candidate: a hand the greedy leaves idle does not hold a cell. The
    # dispatch used to assign over all candidates and never look again, so a
    # worker could be sent out from a cell it does not hold and the day came
    # back INVALID_SOLUTION -- "first task ... reachable too early from entry"
    # (docs/F057, third class, fuzz seeds 93, 182, 205). Which workers are
    # routed is only known after a dispatch, so the assignment is re-derived
    # from the routed set and the dispatch re-run until it stops moving; each
    # of those runs is itself the repeated pass of F057's fourth class
    # (`_dispatch_rounds`), so a successor stranded on an idle hand is picked up
    # before the placement is judged.
    entry_assignments = assign_entry_cells([(w.index, w.earliest_start) for w in candidates])
    routes: list[WorkerRoute] = []
    for _attempt in range(_PLACEMENT_ATTEMPTS):
        routes, remaining_targets = _dispatch_rounds(instance, candidates, entry_assignments)
        if remaining_targets:
            break
        routed = {route.worker_index for route in routes}
        corrected = assign_entry_cells(
            [(w.index, w.earliest_start) for w in candidates if w.index in routed]
        )
        if all(entry_assignments.get(index) == corrected.get(index) for index in routed):
            break
        entry_assignments = {**entry_assignments, **corrected}

    active_workers = len(routes)
    wall_time = time.perf_counter() - start

    if remaining_targets:
        return OxaResult(status="INFEASIBLE", wall_time_seconds=wall_time)
        
    # Engine cost accounting (same rule as `verify_solution`): hiring is
    # append-only, so if the highest hand used is m, hands 0..m were all paid
    # fib(0..m) -- an idle middle worker is legal but still on the payroll.
    # Summing only the routes that carry tasks under-reports whenever the
    # greedy leaves a gap, which the verifier then rejects as
    # INVALID_SOLUTION (fuzz seed 9 of the review audit; the other two
    # seeds that audit reports, 93 and 182/205, fail for the entry-cell
    # reason docs/F057 records as still open).
    max_active = max((route.worker_index for route in routes), default=-1)
    solution = Solution(
        routes=routes, 
        reported_cost=sum(fibonacci_cost(i) for i in range(max_active + 1))
    )
    
    if config.validate:
        try:
            verification = verify_solution(instance, solution)
            if not verification.is_valid:
                return OxaResult(status="INVALID_SOLUTION", solution=solution,
                                 wall_time_seconds=wall_time,
                                 matched_lower_bound=False)
        except Exception:
            return OxaResult(status="INVALID_SOLUTION", solution=solution,
                             wall_time_seconds=wall_time,
                             matched_lower_bound=False)
        
    is_optimal = active_workers <= config.min_workers
    return OxaResult(
        status="OPTIMAL" if is_optimal else "FEASIBLE",
        solution=solution,
        wall_time_seconds=wall_time,
        matched_lower_bound=is_optimal,
    )