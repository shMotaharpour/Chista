"""
Independent solution verifier
Checks constraints: Unique assignment, travel distance (Manhattan),
route start, strict precedence, single-worker groups, and per-worker item
flow (carried inventory never negative).

COST ACCOUNTING (engine): hiring is append-only — if the highest-index
hand used is m, hands 0..m were all hired and paid fib(0..m), whether or
not every one of them was given work. "Having work" is NOT required; only
the payroll matters. A solution whose workers have gaps (e.g. 4 active, 3
idle) is legal but pays for the idle middle workers too.

Item-flow semantics (aggregation-aware): a consume task (FEED/FERTILIZE/
PLACE) is supplied by a scheduled PICKUP of the same item on the same route
at any earlier-or-equal time, OR by its own paired acquire task if that
acquire is itself scheduled. A route's acquire tasks whose item is covered
by an aggregated scheduled pickup (resolved_qty >= pending demand) may stay
unscheduled — this mirrors what the CP-SAT and routing solvers emit when
they batch one shed pickup to serve several consumes.
"""
from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

from agent.world.board import manhattan
from agent.world.rules import spawn_assignments
from agent.wsr.models import (
    CONSUME_ACTIONS,
    PRODUCE_ACTIONS,
    Item,
    Action,
    
    Cell,
    Instance,
    MinorTask,
    ScheduledTask,
    Solution,
)

_ENTRY_CELLS = frozenset(SHED_ACCESS)

@dataclass
class VerificationResult:
    is_valid: bool
    violations: list[str] = field(default_factory=list)
    total_cost: Optional[int] = None

def _effective_cell(task: MinorTask, scheduled: ScheduledTask, violations: list[str]) -> Optional[Cell]:
    if task.cell is not None:
        return task.cell
    if scheduled.resolved_cell is None:
        violations.append(f"task {task.id!r} has a flexible location but no resolved_cell was given")
        return None
    if scheduled.resolved_cell not in _ENTRY_CELLS:
        violations.append(f"task {task.id!r}: resolved_cell {scheduled.resolved_cell} is not an entry cell")
        return None
    return scheduled.resolved_cell

def verify_solution(instance: Instance, solution: Solution,
                    entry_assign: Optional[dict[int, str]] = None) -> VerificationResult:
    violations: list[str] = []
    tasks_by_id = instance.tasks_by_id
    workers_by_index = {worker.index: worker for worker in instance.workers}
    
    assigned_worker: dict[str, int] = {}
    exec_time_of: dict[str, int] = {}
    seen_at: dict[str, int] = {}

    # 1. Check valid routes, horizons, and duplicate/missing tasks
    for route in solution.routes:
        worker = workers_by_index.get(route.worker_index)
        if worker is None:
            violations.append(f"route references unknown worker index {route.worker_index}")
            continue
        if route.start_time < worker.earliest_start:
            violations.append(f"worker {route.worker_index}: start_time < earliest_start")
        if route.start_time > instance.horizon:
            violations.append(f"worker {route.worker_index}: start_time beyond horizon")

        for scheduled in route.tasks:
            task = tasks_by_id.get(scheduled.task_id)
            if task is None:
                violations.append(f"worker {route.worker_index}: unknown task {scheduled.task_id!r}")
                continue
            if scheduled.task_id in seen_at:
                violations.append(f"task {scheduled.task_id!r} is scheduled more than once")
            
            seen_at[scheduled.task_id] = route.worker_index
            assigned_worker[scheduled.task_id] = route.worker_index
            exec_time_of[scheduled.task_id] = scheduled.exec_time
            
            if not (0 <= scheduled.exec_time <= instance.horizon):
                violations.append(f"task {scheduled.task_id!r}: exec_time outside horizon")

    # 1b. Item flow: every consume must be covered by carried inventory on
    # its own route. A route's acquire tasks of an item may stay unscheduled
    # when a scheduled pickup of that item on the same route carries enough
    # quantity (aggregation). Inventory from PRE-DAY preload is modeled as
    # the acquire tasks themselves; the check below treats each route's
    # scheduled pickups as its stock and requires every scheduled consume to
    # be covered in time order.
    route_demand: dict[int, dict[Item, int]] = defaultdict(lambda: defaultdict(int))
    route_supply: dict[int, dict[Item, int]] = defaultdict(lambda: defaultdict(int))
    for route in solution.routes:
        if route.worker_index not in workers_by_index:
            continue
        ordered = sorted(route.tasks, key=lambda s: (s.exec_time, s.task_id))
        balance: dict[Item, int] = defaultdict(int)
        for scheduled in ordered:
            task = tasks_by_id.get(scheduled.task_id)
            if task is None:
                continue
            if task.action == UnitAction.PICKUP and task.item is not None:
                qty = scheduled.resolved_qty if scheduled.resolved_qty is not None else task.qty
                balance[task.item] += qty
                route_supply[route.worker_index][task.item] += qty
            elif task.action in CONSUME_ACTIONS and task.item is not None:
                balance[task.item] -= task.qty
                route_demand[route.worker_index][task.item] += task.qty
                if balance[task.item] < 0:
                    violations.append(
                        f"worker {route.worker_index}: inventory of "
                            f"{getattr(task.item, 'value', task.item)!r} goes negative "
                        f"({balance[task.item]}) at task {scheduled.task_id!r} (t={scheduled.exec_time})"
                    )
            elif task.action == UnitAction.DROP:
                balance.clear()

    # missing tasks: unscheduled acquire tasks covered by an aggregated
    # pickup on their group's route are NOT missing
    aggregated_aways: set[str] = set()
    for task in instance.minor_tasks:
        if task.id in seen_at:
            continue
        if task.action != UnitAction.PICKUP or task.item is None:
            continue
        group = next((g for g in instance.single_worker_groups if task.id in g), None)
        if group is None:
            continue
        partner_w = {assigned_worker[t] for t in group
                     if t in seen_at and tasks_by_id[t].action != UnitAction.PICKUP}
        if len(partner_w) != 1:
            continue
        w = partner_w.pop()
        if route_supply[w][task.item] >= route_demand[w][task.item]:
            aggregated_aways.add(task.id)
        else:
            violations.append(
                f"acquire task {task.id!r} aggregated away but worker {w}'s pickup "
                f"qty {route_supply[w][task.item]} does not cover demand {route_demand[w][task.item]}"
            )

    missing = sorted(task.id for task in instance.minor_tasks
                     if task.id not in seen_at and task.id not in aggregated_aways)
    if missing:
        violations.append(f"tasks never scheduled: {missing}")

    # 1c. Unscheduled acquire tasks: allowed only when the SAME route
    # schedules a pickup of the same item whose qty covers all consumes.
    # (Aggregated-pickup emission by the solvers.)
    scheduled_ids = seen_at
    for task in instance.minor_tasks:
        if task.id in scheduled_ids:
            continue
        if task.action != UnitAction.PICKUP or task.item is None:
            continue
        # find the route that covers its consumes (single-worker group partner)
        group = next((g for g in instance.single_worker_groups if task.id in g), None)
        if group is None:
            violations.append(f"acquire task {task.id!r} never scheduled and not part of any group")
            continue
        partner_w = {assigned_worker[t] for t in group if t in scheduled_ids and
                     tasks_by_id[t].action != UnitAction.PICKUP}
        if len(partner_w) != 1:
            violations.append(f"acquire task {task.id!r} unscheduled but its group has no single scheduled consumer")
            continue
        w = partner_w.pop()
        # the route's scheduled pickup qty must cover ALL demand of that item
        if route_supply[w][task.item] < route_demand[w][task.item]:
            violations.append(
                f"acquire task {task.id!r} aggregated away but worker {w}'s pickup "
                f"qty {route_supply[w][task.item]} does not cover demand {route_demand[w][task.item]}"
            )

    # 2. Constraints 2 & 3: Sequencing and Travel Time
    for route in solution.routes:
        if route.worker_index not in workers_by_index:
            continue
        prev_cell: Optional[Cell] = None
        prev_time: Optional[int] = None
        for scheduled in route.tasks:
            task = tasks_by_id.get(scheduled.task_id)
            if task is None: continue
            cell = _effective_cell(task, scheduled, violations)
            if prev_cell is not None and prev_time is not None and cell is not None:
                min_time = prev_time + 1 + manhattan(prev_cell, cell)
                if scheduled.exec_time < min_time:
                    violations.append(f"worker {route.worker_index}: task {scheduled.task_id!r} reachable too early")
            prev_cell = cell
            prev_time = scheduled.exec_time

    # 3. Constraint 4: Strict Precedence
    for i, j in instance.precedence:
        if i in exec_time_of and j in exec_time_of:
            if not (exec_time_of[i] < exec_time_of[j]):
                violations.append(f"precedence violated: {i!r} must strictly precede {j!r}")

    # 4. Constraint 5: Single-worker groups
    for group in instance.single_worker_groups:
        workers_in_group = {assigned_worker[t] for t in group if t in assigned_worker}
        if len(workers_in_group) > 1:
            violations.append(f"single-worker group {group} was split across workers")

    # 5. Constraint 10: Placement (entry-cell reachability)
    violations.extend(_check_placement(instance, solution, tasks_by_id, entry_assign))

    # 6. Cost Calculation — engine accounting: hiring is append-only, so if
    # the highest-index worker used is m, hands 0..m were ALL hired and paid
    # (fib(0..m)), including any idle ones in the middle. A gap is legal but
    # costs money, which is exactly how the engine charges hires.
    active_workers = {route.worker_index for route in solution.routes if route.worker_index in workers_by_index}
    max_active = max(active_workers) if active_workers else -1
    total_cost = max(max_active - 1, 0)   # the same head count the scheduler reports
    
    if solution.hired is not None and solution.hired != total_cost:
        violations.append(f"hired={solution.hired} mismatch the schedule's own head count={total_cost}")

    return VerificationResult(is_valid=not violations, violations=violations, total_cost=total_cost)

def _check_placement(instance: Instance, solution: Solution, tasks_by_id: dict[str, MinorTask], entry_assign_override: Optional[dict[int, str]] = None) -> list[str]:
    violations: list[str] = []
    workers_by_index = {worker.index: worker for worker in instance.workers}
    worker_starts = [(route.worker_index, route.start_time) for route in solution.routes if route.worker_index in workers_by_index]
    if entry_assign_override is not None:
        entry_assignment = dict(entry_assign_override)
    else:
        entry_assignment = spawn_assignments(worker_starts)
    
    for route in solution.routes:
        if not route.tasks or route.worker_index not in workers_by_index:
            continue
        entry_name = entry_assignment.get(route.worker_index)
        if entry_name is None: continue
        entry_cell = SHED_ACCESS[entry_name]
        first = route.tasks[0]
        task = tasks_by_id.get(first.task_id)
        if task is None: continue
        first_cell = task.cell if task.cell is not None else first.resolved_cell
        if first_cell is None: continue
        
        min_time = route.start_time + 1 + manhattan(entry_cell, first_cell)
        if first.exec_time < min_time:
            violations.append(f"worker {route.worker_index}: first task {first.task_id!r} reachable too early from entry")
    return violations
