"""
CP-SAT exact solver for the Worker Distribution problem.

Builds the mathematical model directly using CP-SAT variables and constraints. 
This optimized version applies key domain assumptions to heavily reduce the 
search space and accelerate solve times.
"""

from __future__ import annotations
import itertools
import time
from dataclasses import dataclass, replace
from typing import Optional

from ortools.sat.python import cp_model

from ..distances import manhattan
from ..fibonacci import fibonacci_cost
from ..models import (
    CONSUME_ACTIONS,
    PRODUCE_ACTIONS,
    Instance,
    Item,
    MinorActionType,
    MinorTask,
    WAREHOUSE_ENTRY_CELLS,
    WAREHOUSE_ENTRY_ORDER,
    Cell,
    Instance,
    ScheduledTask,
    Solution,
    WorkerRoute,
)
from ..verify import verify_solution
from .base_config import BaseSolverConfig

@dataclass
class CpSatConfig(BaseSolverConfig):
    time_limit_seconds: Optional[float] = 10.0
    num_search_workers: int = 8
    linearization_level: int = 1
    cost_ceiling: Optional[int] = None
    feasibility_only: bool = True
    cost_floor: Optional[int] = None
    worker_pool_cap: Optional[int] = 11
    warm_start: Optional[Solution] = None


@dataclass
class CpSatResult:
    status: str
    solution: Optional[Solution] = None
    wall_time_seconds: float = 0.0
    pool_capped_at: Optional[int] = None


class SolverProducedInvalidSolutionError(AssertionError):
    pass

class InfeasibleInputError(ValueError):
    """The instance is provably infeasible before a CP-SAT model is even
    built (e.g. more of an item is picked up across all tasks than the
    shed holds) -- a static property of the task list itself."""
    pass



def _add_warm_start_hints(
    model: cp_model.CpModel,
    tasks: list,
    task_index: dict[str, int],
    workers: list,
    u: list,
    assign: list,
    tau: list,
    flex: dict[int, list],
    sched: dict[int, cp_model.IntVar],
    entry_cells: list,
    horizon: int,
    solution: Solution,
) -> None:
    worker_index_to_w = {workers[w].index: w for w in range(len(workers))}
    entry_cell_to_c = {cell: c for c, cell in enumerate(entry_cells)}
    scheduled_by_task: dict[str, tuple[int, ScheduledTask]] = {}
    active_worker_indices: set[int] = set()

    for route in solution.routes:
        if not route.tasks:
            continue
        active_worker_indices.add(route.worker_index)
        for scheduled in route.tasks:
            scheduled_by_task[scheduled.task_id] = (route.worker_index, scheduled)

    hint_vars: list = []
    hint_vals: list[int] = []

    for w in range(len(workers)):
        hint_vars.append(u[w])
        hint_vals.append(1 if workers[w].index in active_worker_indices else 0)

    for task_id, i in task_index.items():
        task = tasks[i]
        hit = scheduled_by_task.get(task_id)
        if hit is None:
            if i in sched:
                hint_vars.append(sched[i])
                hint_vals.append(0)
            continue

        worker_index, scheduled_task = hit
        w = worker_index_to_w.get(worker_index)
        if w is None:
            continue

        for w2 in range(len(workers)):
            hint_vars.append(assign[i][w2])
            hint_vals.append(1 if w2 == w else 0)

        if 0 <= scheduled_task.exec_time <= horizon:
            hint_vars.append(tau[i])
            hint_vals.append(scheduled_task.exec_time)

        if i in flex and scheduled_task.resolved_cell is not None:
            c = entry_cell_to_c.get(scheduled_task.resolved_cell)
            if c is not None:
                for c2 in range(4):
                    hint_vars.append(flex[i][c2])
                    hint_vals.append(1 if c2 == c else 0)

        if i in sched:
            hint_vars.append(sched[i])
            hint_vals.append(1)

    for var, val in zip(hint_vars, hint_vals):
        model.AddHint(var, val)


def _find_aggregatable_pickups(instance: Instance, task_index: dict[str, int]) -> frozenset[int]:
    """Identify PICKUP tasks eligible for aggregation (see module
    docstring): the PICKUP half of a size-2 `single_worker_group` whose
    other half consumes the same item, with a precedence edge from the
    PICKUP to it. Structural, not tied to any specific major_task recipe
    name -- feed/frtz/frtz_water/place_animal all happen to produce
    exactly this shape today.
    """
    tasks = instance.minor_tasks
    precedence_set = set(instance.precedence)
    aggregatable: set[int] = set()
    for group in instance.single_worker_groups:
        if len(group) != 2:
            continue
        i0, i1 = task_index[group[0]], task_index[group[1]]
        for p, c in ((i0, i1), (i1, i0)):
            pickup, consume = tasks[p], tasks[c]
            if (
                pickup.action == MinorActionType.PICKUP
                and pickup.cell is None
                and pickup.item is not None
                and consume.action in CONSUME_ACTIONS
                and consume.item == pickup.item
                and (pickup.id, consume.id) in precedence_set
            ):
                aggregatable.add(p)
    return frozenset(aggregatable)



def _worker_count_lower_bound(instance: Instance, aggregatable_pickups_indices: frozenset[int]) -> int:
    mandatory = sum(1 for i in range(len(instance.minor_tasks)) if i not in aggregatable_pickups_indices)
    if instance.horizon <= 0:
        return max(1, mandatory)
    return max(1, -(-mandatory // instance.horizon))


def _validate_item_sources(instance: Instance) -> None:
    """Static check: every consume task must have an item source — either
    a paired acquire in its single-worker group, or any PICKUP task of
    the same item somewhere in the instance (preload-able). Raises
    InfeasibleInputError otherwise."""
    pickup_items = {t.item for t in instance.minor_tasks
                    if t.action == MinorActionType.PICKUP and t.item is not None}
    for t in instance.minor_tasks:
        if t.action not in CONSUME_ACTIONS or t.item is None:
            continue
        has_partner = any(
            t.id in g and any(
                instance.tasks_by_id[tid].action == MinorActionType.PICKUP and
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

def solve_cpsat(instance: Instance, config: CpSatConfig = CpSatConfig()) -> CpSatResult:
    _validate_item_sources(instance)
    model = cp_model.CpModel()
    tasks = instance.minor_tasks
    n = len(tasks)
    
    # Use pre-computed lookups from Instance
    task_index = instance.task_index
    # Convert set of IDs to set of indices for CP-SAT internal arrays
    aggregatable_pickups = frozenset(task_index[tid] for tid in instance.aggregatable_pickups)

    # ---- static input validation: warehouse stock overrun ----------------
    # A static property of the task list: if total PICKUP demand for an
    # item exceeds the shed's stock, no schedule can ever fix it.
    if instance.warehouse_stock:
        picked: dict[Item, int] = {}
        for task in tasks:
            if (task.action == MinorActionType.PICKUP and task.item is not None):
                picked[task.item] = picked.get(task.item, 0) + task.qty
        for item, amount in picked.items():
            stock = instance.warehouse_stock.get(item, 0)
            if amount > stock:
                raise InfeasibleInputError(
                    f"warehouse stock for {item.value!r} is {stock}, "
                    f"but {amount} is picked up across all tasks"
                )

    workers = sorted(instance.workers, key=lambda w: w.index)
    offered = len(workers)
    if instance.worker_pool_size is not None:
        workers = workers[: instance.worker_pool_size]
    if config.worker_pool_cap is not None:
        workers = workers[: config.worker_pool_cap]

    k = len(workers)
    pool_capped_at = k if k < offered and config.worker_pool_cap is not None else None
    sigma = [w.earliest_start for w in workers]
    horizon = instance.horizon

    # ---- constraint 11: prefix worker activation
    u = [model.NewBoolVar(f"u_{workers[w].index}") for w in range(k)]
    for w in range(k - 1):
        model.Add(u[w] >= u[w + 1])

    lower_bound = _worker_count_lower_bound(instance, aggregatable_pickups)
    if 0 < lower_bound <= k:
        model.Add(u[lower_bound - 1] == 1)

    # ---- constraint 1: Task Assignment & Lightweight Aggregation
    assign = [[model.NewBoolVar(f"assign_{i}_{w}") for w in range(k)] for i in range(n)]
    sched: dict[int, cp_model.IntVar] = {}

    for i in aggregatable_pickups:
        s = model.NewBoolVar(f"sched_{i}")
        model.Add(s == sum(assign[i]))
        sched[i] = s

    for i in range(n):
        if i in aggregatable_pickups:
            model.AddAtMostOne(assign[i])
        else:
            model.AddExactlyOne(assign[i])
        
        for w in range(k):
            model.Add(assign[i][w] <= u[w])

    for w in range(k):
        model.Add(sum(assign[i][w] for i in range(n)) >= u[w])

    a = [model.NewIntVar(0, k, f"a_{i}") for i in range(n)]
    for i in range(n):
        placed = sum(w * assign[i][w] for w in range(k))
        if i in sched:
            model.Add(a[i] == placed + k * (1 - sched[i]))
        else:
            model.Add(a[i] == placed)

    same_worker: dict[tuple[int, int], cp_model.IntVar] = {}
    for i, j in itertools.combinations(range(n), 2):
        b = model.NewBoolVar(f"same_worker_{i}_{j}")
        model.Add(a[i] == a[j]).OnlyEnforceIf(b)
        model.Add(a[i] != a[j]).OnlyEnforceIf(b.Not())
        same_worker[(i, j)] = b

    tau = [model.NewIntVar(0, horizon, f"tau_{i}") for i in range(n)]

    # Lightweight Aggregation Logic & Symmetry Breaking
    all_items = {tasks[i].item for i in aggregatable_pickups if tasks[i].item is not None}
    worker_any_pickup = {w: [] for w in range(k)}

    for item in all_items:
        P_k = [i for i in aggregatable_pickups if tasks[i].item == item]
        C_k = [task_index[succ] for pred, succ in instance.precedence if task_index[pred] in P_k]
        
        for w in range(k):
            has_item = model.NewBoolVar(f"has_item_{w}_{item.name}")
            model.AddMaxEquality(has_item, [assign[j][w] for j in C_k])
            
            exec_on = []
            for i in P_k:
                e = model.NewBoolVar(f"exec_on_{i}_{w}")
                model.AddBoolAnd([sched[i], assign[i][w]]).OnlyEnforceIf(e)
                model.AddBoolOr([sched[i].Not(), assign[i][w].Not()]).OnlyEnforceIf(e.Not())
                exec_on.append(e)
                worker_any_pickup[w].append(e)
                
                for j in C_k:
                    model.Add(tau[i] < tau[j]).OnlyEnforceIf([e, assign[j][w]])

            model.Add(sum(exec_on) == 1).OnlyEnforceIf(has_item)
            model.Add(sum(exec_on) == 0).OnlyEnforceIf(has_item.Not())

    has_any_pickup_var = []
    for w in range(k):
        has_any = model.NewBoolVar(f"has_any_pickup_{w}")
        if worker_any_pickup[w]:
            model.AddMaxEquality(has_any, worker_any_pickup[w])
        else:
            model.Add(has_any == 0)
        has_any_pickup_var.append(has_any)

    for w in range(1, k):
        model.AddImplication(has_any_pickup_var[w], has_any_pickup_var[0])

    # ---- Flexible (shed-adjacent) cell resolution
    entry_names = WAREHOUSE_ENTRY_ORDER
    entry_cells = [WAREHOUSE_ENTRY_CELLS[name] for name in entry_names]

    flex: dict[int, list[cp_model.IntVar]] = {}
    for i, task in enumerate(tasks):
        if task.cell is None:
            bools = [model.NewBoolVar(f"flex_{i}_{c}") for c in range(4)]
            model.AddExactlyOne(bools)
            flex[i] = bools

    def cell_options(i: int) -> list[tuple[Optional[cp_model.IntVar], Cell]]:
        task = tasks[i]
        if task.cell is not None:
            return [(None, task.cell)]
        return [(flex[i][c], entry_cells[c]) for c in range(4)]

    # ---- constraint 10: exact worker entry-cell placement
    order = sorted(range(k), key=lambda w: (sigma[w], workers[w].index))
    counts: list[list[cp_model.IntVar]] = [[model.NewConstant(0) for _ in range(4)]]
    chosen_by_worker: list[Optional[list[cp_model.IntVar]]] = [None] * k

    for step, w in enumerate(order):
        prev = counts[-1]
        min_count = model.NewIntVar(0, k, f"min_count_{step}")
        model.AddMinEquality(min_count, prev)

        is_min = []
        for c in range(4):
            b = model.NewBoolVar(f"is_min_{w}_{c}")
            model.Add(prev[c] == min_count).OnlyEnforceIf(b)
            model.Add(prev[c] != min_count).OnlyEnforceIf(b.Not())
            is_min.append(b)

        chosen: list[cp_model.IntVar] = []
        for c in range(4):
            if c == 0:
                pc = is_min[0]
            else:
                pc = model.NewBoolVar(f"chosen_{w}_{c}")
                not_earlier = [ch.Not() for ch in chosen]
                model.AddBoolAnd([is_min[c]] + not_earlier).OnlyEnforceIf(pc)
                model.AddBoolOr([is_min[c].Not()] + chosen).OnlyEnforceIf(pc.Not())
            chosen.append(pc)
        model.AddExactlyOne(chosen)
        chosen_by_worker[w] = chosen

        new_counts = []
        for c in range(4):
            nc = model.NewIntVar(0, k, f"count_{step}_{c}")
            inc = model.NewBoolVar(f"inc_{w}_{c}")
            model.AddBoolAnd([chosen[c], u[w]]).OnlyEnforceIf(inc)
            model.AddBoolOr([chosen[c].Not(), u[w].Not()]).OnlyEnforceIf(inc.Not())
            model.Add(nc == prev[c] + inc)
            new_counts.append(nc)
        counts.append(new_counts)

    # ---- constraint 3 (route start)
    for i in range(n):
        for w in range(k):
            for c in range(4):
                entry_lit, entry_cell = chosen_by_worker[w][c], entry_cells[c]
                for cell_lit, cell in cell_options(i):
                    dist = manhattan(entry_cell, cell)
                    lits = [assign[i][w], entry_lit]
                    if cell_lit is not None:
                        lits.append(cell_lit)
                    model.Add(tau[i] >= sigma[w] + 1 + dist).OnlyEnforceIf(lits)

    # ---- constraint 2 (route sequencing)
    before_vars: dict[tuple[int, int], cp_model.IntVar] = {}
    for i, j in itertools.combinations(range(n), 2):
        before = model.NewBoolVar(f"before_{i}_{j}")
        before_vars[(i, j)] = before
        
        for cell_lit_i, cell_i in cell_options(i):
            for cell_lit_j, cell_j in cell_options(j):
                dist_ij = manhattan(cell_i, cell_j)
                common = [same_worker[(i, j)]]
                if cell_lit_i is not None: common.append(cell_lit_i)
                if cell_lit_j is not None: common.append(cell_lit_j)

                model.Add(tau[j] >= tau[i] + 1 + dist_ij).OnlyEnforceIf(common + [before])
                model.Add(tau[i] >= tau[j] + 1 + dist_ij).OnlyEnforceIf(common + [before.Not()])

    # ---- constraint 4: strict precedence
    for pred_id, succ_id in instance.precedence:
        p, s = task_index[pred_id], task_index[succ_id]
        if p in sched:
            model.Add(tau[p] < tau[s]).OnlyEnforceIf(sched[p])
        else:
            model.Add(tau[p] < tau[s])

    # ---- constraint 7: DROP-aware inventory segments --------------------
    # A scheduled DROP on a worker's route wipes their carried inventory.
    # A same-item consume scheduled after a scheduled DROP therefore
    # needs a scheduled pickup of that item BETWEEN the drop and the
    # consume (the route-aggregation shortcut only covers consumes in
    # the same segment as the pickup).
    drop_idxs = [i for i, tsk in enumerate(tasks) if tsk.action.name == "DROP"]
    if drop_idxs:
        pickup_idxs = [i for i, tsk in enumerate(tasks)
                       if tsk.action == MinorActionType.PICKUP and tsk.item is not None]
        for ci, tsk in enumerate(tasks):
            if tsk.action not in CONSUME_ACTIONS or tsk.item is None:
                continue
            same_item_pickups = [p for p in pickup_idxs if tasks[p].item == tsk.item]
            if not same_item_pickups:
                continue
            for w in range(k):
                for di, d in enumerate(drop_idxs):
                    # proper 3-way reification:
                    #   before_d == assign[d][w] AND assign[ci][w] AND (tau[d] < tau[ci])
                    same_w = model.NewBoolVar(f"c{ci}_w{w}_d{di}_samew")
                    model.AddBoolAnd([assign[d][w], assign[ci][w]]).OnlyEnforceIf(same_w)
                    model.AddBoolOr([assign[d][w].Not(), assign[ci][w].Not(), same_w])
                    before_raw = model.NewBoolVar(f"c{ci}_w{w}_d{di}_raw")
                    model.Add(tau[d] < tau[ci]).OnlyEnforceIf(before_raw)
                    model.Add(tau[d] >= tau[ci]).OnlyEnforceIf(before_raw.Not())
                    before_d = model.NewBoolVar(f"c{ci}_w{w}_d{di}_before")
                    model.AddBoolAnd([same_w, before_raw]).OnlyEnforceIf(before_d)
                    model.AddBoolOr([same_w.Not(), before_raw.Not(), before_d])
                    fresh_or = []
                    for p in same_item_pickups:
                        # fresh segment pickup: p in (tau[d], tau[ci]] on w
                        # p_on_w_after == assign[p][w] AND tau[d] < tau[p] AND tau[p] <= tau[ci]
                        pw = model.NewBoolVar(f"c{ci}_w{w}_d{di}_p{p}_w")
                        model.AddImplication(pw, assign[p][w])
                        pa = model.NewBoolVar(f"c{ci}_w{w}_d{di}_p{p}_a")
                        model.Add(tau[d] < tau[p]).OnlyEnforceIf(pa)
                        model.Add(tau[d] >= tau[p]).OnlyEnforceIf(pa.Not())
                        pb = model.NewBoolVar(f"c{ci}_w{w}_d{di}_p{p}_b")
                        model.Add(tau[p] <= tau[ci]).OnlyEnforceIf(pb)
                        model.Add(tau[p] > tau[ci]).OnlyEnforceIf(pb.Not())
                        on_w_after = model.NewBoolVar(f"c{ci}_w{w}_d{di}_p{p}")
                        model.AddBoolAnd([pw, pa, pb]).OnlyEnforceIf(on_w_after)
                        model.AddBoolOr([pw.Not(), pa.Not(), pb.Not(), on_w_after])
                        fresh_or.append(on_w_after)
                    if fresh_or:
                        fresh = model.NewBoolVar(f"c{ci}_w{w}_d{di}_fresh")
                        model.AddMaxEquality(fresh, fresh_or)
                        model.AddImplication(before_d, fresh)
                    else:
                        # no same-item pickup exists: a consume after a drop
                        # can never be supplied
                        model.Add(before_d == 0)

    # ---- constraint 5: single-worker groups
    for group in instance.single_worker_groups:
        idxs = [task_index[tid] for tid in group]
        agg_members = [i for i in idxs if i in aggregatable_pickups]
        anchor_pool = [i for i in idxs if i not in agg_members]
        ref = anchor_pool[0] if anchor_pool else idxs[0]
        
        for other in anchor_pool[1:]:
            for w in range(k):
                model.Add(assign[ref][w] == assign[other][w])
        
        for pickup_idx in agg_members:
            for w in range(k):
                model.Add(assign[pickup_idx][w] <= assign[ref][w])

    if config.warm_start is not None:
        _add_warm_start_hints(
            model, tasks, task_index, workers, u, assign, tau, flex, sched, entry_cells, horizon, config.warm_start
        )

    # ---- objective (only when optimizing) ------------------------------
    # feasibility_only=True (default): NO Minimize — CP-SAT only checks
    # feasibility; the minimum worker count is found by
    # solve_cpsat_prefix_search (binary search on pool size).
    # feasibility_only=False: Minimize(Σ fib·u) — the monolithic optimum.
    # (0f1af97 added this unconditionally, which broke both modes; caught
    # in the commit-by-commit review.)
    if not config.feasibility_only:
        model.Minimize(sum(fibonacci_cost(workers[w].index) * u[w] for w in range(k)))

    solver = cp_model.CpSolver()
    if config.time_limit_seconds is not None:
        solver.parameters.max_time_in_seconds = config.time_limit_seconds
    solver.parameters.num_search_workers = config.num_search_workers
    solver.parameters.linearization_level = config.linearization_level

    status = solver.Solve(model)
    status_name = solver.StatusName(status)

    if status_name not in ("OPTIMAL", "FEASIBLE"):
        return CpSatResult(
            status=status_name, solution=None, wall_time_seconds=solver.WallTime(), pool_capped_at=pool_capped_at
        )

    routes: list[WorkerRoute] = []
    for w in range(k):
        if solver.Value(u[w]) != 1:
            continue
        
        my_tasks = [i for i in range(n) if solver.Value(assign[i][w]) == 1]
        my_tasks.sort(key=lambda i: solver.Value(tau[i]))
        
        scheduled = []
        # aggregated-pickup qty: a scheduled acquire covers every consume of
        # the same item on this route (dropped sibling acquires included).
        route_demand: dict = {}
        for j in my_tasks:
            it = tasks[j].item
            if it is not None and tasks[j].action in CONSUME_ACTIONS:
                route_demand[it] = route_demand.get(it, 0) + tasks[j].qty
        for i in my_tasks:
            resolved_cell = None
            if i in flex:
                for c in range(4):
                    if solver.Value(flex[i][c]) == 1:
                        resolved_cell = entry_cells[c]
                        break
            resolved_qty = tasks[i].qty
            if i in aggregatable_pickups and tasks[i].item is not None:
                resolved_qty = max(resolved_qty, route_demand.get(tasks[i].item, tasks[i].qty))
            scheduled.append(
                ScheduledTask(
                    task_id=tasks[i].id,
                    exec_time=solver.Value(tau[i]),
                    resolved_cell=resolved_cell,
                    resolved_qty=resolved_qty,
                )
            )
        chosen_entry = None
        if chosen_by_worker[w]:
            for c in range(4):
                if solver.Value(chosen_by_worker[w][c]) == 1:
                    chosen_entry = entry_cells[c]
                    break
        routes.append(WorkerRoute(worker_index=workers[w].index, start_time=sigma[w], tasks=scheduled,
                                  start_cell=chosen_entry))

    total_cost = sum(fibonacci_cost(workers[w].index) for w in range(k) if solver.Value(u[w]) == 1)
    solution = Solution(routes=routes, reported_cost=total_cost)

    if config.validate:
        try:
            verification = verify_solution(instance, solution)
            if not verification.is_valid:
                return CpSatResult(
                    status="INVALID_SOLUTION",
                    solution=solution,
                    wall_time_seconds=solver.WallTime(),
                    pool_capped_at=pool_capped_at,
                )
        except Exception:
            return CpSatResult(
                status="INVALID_SOLUTION",
                solution=solution,
                wall_time_seconds=solver.WallTime(),
                pool_capped_at=pool_capped_at,
            )

    return CpSatResult(
        status=status_name,
        solution=solution,
        wall_time_seconds=solver.WallTime(),
        pool_capped_at=pool_capped_at,
    )




def cpsat_direct(instance: Instance, config: CpSatConfig = CpSatConfig()) -> CpSatResult:
    """Direct optimizing solve: Minimize(Σ fib·u) regardless of
    config.feasibility_only. Returns the provably optimal schedule."""
    optimizing = replace(config, feasibility_only=False)
    return solve_cpsat(instance, optimizing)

def cpsat_binarySearch(instance: Instance, config: CpSatConfig = CpSatConfig()) -> CpSatResult:
    started = time.perf_counter()

    def remaining() -> Optional[float]:
        if config.time_limit_seconds is None:
            return None
        return max(0.01, config.time_limit_seconds - (time.perf_counter() - started))

    def feasibility_probe(pool: int, share: int) -> CpSatResult:
        budget = remaining()
        if budget is not None:
            budget = max(0.01, budget / max(1, share))
        
        probe_config = replace(
            config,
            feasibility_only=True,
            worker_pool_cap=pool,
            time_limit_seconds=budget,
            warm_start=None,
        )
        return solve_cpsat(instance, probe_config)

    def finish(result: CpSatResult, status: str) -> CpSatResult:
        return CpSatResult(
            status=status,
            solution=result.solution,
            wall_time_seconds=time.perf_counter() - started,
            pool_capped_at=result.pool_capped_at,
        )

    offered = sorted(instance.workers, key=lambda w: w.index)
    if instance.worker_pool_size is not None:
        offered = offered[: instance.worker_pool_size]
    if config.worker_pool_cap is not None:
        offered = offered[: config.worker_pool_cap]

    k_max = len(offered)
    if k_max == 0:
        # no workers offered at all -- nothing to search over; the direct
        # optimizing solve is the only entry point that can answer (#14).
        return cpsat_direct(instance, config)

    aggregatable_pickups = frozenset(instance.task_index[tid] for tid in instance.aggregatable_pickups)

    budget_slots = max(2, k_max.bit_length() + 2)
    
    full = feasibility_probe(k_max, budget_slots)
    budget_slots -= 1
    
    if full.status == "INFEASIBLE":
        return finish(full, "INFEASIBLE")
    if full.solution is None:
        # the full-pool probe found nothing inside its slice of the budget:
        # fall back to the optimizing solve rather than inventing a verdict.
        return cpsat_direct(instance, config)

    lo = max(1, min(_worker_count_lower_bound(instance, aggregatable_pickups), k_max))
    hi = k_max
    best = full

    while lo < hi:
        mid = (lo + hi) // 2
        probe = feasibility_probe(mid, budget_slots)
        budget_slots = max(1, budget_slots - 1)

        if probe.solution is not None:
            best = probe
            hi = mid
        elif probe.status == "INFEASIBLE":
            lo = mid + 1
        else:
            return finish(best, "FEASIBLE")

    return finish(best, "OPTIMAL")

