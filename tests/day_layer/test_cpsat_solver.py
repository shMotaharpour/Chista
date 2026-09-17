"""CP-SAT solver tests against the hand-solved fixture library (see
tests/fixtures/hand_solved_instances.py) and a few solver-specific checks
(infeasibility detection, config handling). The independent brute-force
cross-check lives separately in test_brute_force_cross_check.py.
"""
import pytest

from day.models import (
    Instance,
    MajorTask,
    WAREHOUSE_ENTRY_CELLS,
    Item,
    Action,
    MinorTask,
    ScheduledTask,
    Solution,
    Worker,
    WorkerRoute,
)
from day.solvers.cpsat_solver import (
    CpSatConfig,
    InfeasibleInputError,
    _find_aggregatable_pickups,
    _worker_count_lower_bound,
    solve_cpsat,
    cpsat_binarySearch,
)
from day.verify import verify_solution
from tests.day_layer.fixtures.hand_solved_instances import HAND_SOLVED_CASES

NW = WAREHOUSE_ENTRY_CELLS["NW"]
FAST = CpSatConfig(time_limit_seconds=10, feasibility_only=False)


@pytest.mark.parametrize("case", HAND_SOLVED_CASES, ids=lambda case: case.name)
def test_cpsat_reaches_the_known_optimum(case):
    result = solve_cpsat(case.instance, FAST)
    assert result.status == "OPTIMAL", result.status
    assert result.solution.reported_cost == case.optimal_cost

    verification = verify_solution(case.instance, result.solution)
    assert verification.is_valid, verification.violations
    assert verification.total_cost == case.optimal_cost


@pytest.mark.parametrize("level", [0, 1, 2])
def test_linearization_level_is_honored_and_still_reaches_the_optimum(level):
    # linearization_level is just a search-tuning knob (see CpSatConfig's
    # comment for the benchmark that motivated exposing it) -- it must
    # never change *what* CP-SAT proves, only how fast. Cheap enough
    # instance that all 3 levels finish quickly regardless.
    (case,) = [c for c in HAND_SOLVED_CASES if c.name == "feed_single_worker_chain_needs_only_worker_0"]
    result = solve_cpsat(case.instance, CpSatConfig(time_limit_seconds=10, linearization_level=level))
    assert result.status == "OPTIMAL", result.status
    assert result.solution.reported_cost == case.optimal_cost

    verification = verify_solution(case.instance, result.solution)
    assert verification.is_valid, verification.violations


def test_feasibility_only_stops_at_the_first_schedule_meeting_the_ceiling():
    # "I already know N workers suffice, I just need the actual
    # schedule" -- feasibility_only drops Minimize entirely, so the
    # result may not even be the true optimum, just something valid at
    # or under the ceiling. Enough workers offered that the true optimum
    # (1 worker, cost 0) exists, but the ceiling is loose (5 workers'
    # worth) -- the point is this must never *exceed* the ceiling, not
    # that it hits the true optimum.
    major = MajorTask(id="feed1", type="feed", cell=NW)
    workers = [Worker(index=i, earliest_start=0) for i in range(8)]
    instance = Instance.compile(workers=workers, major_tasks=[major], warehouse_stock={Item.WHEAT: 5})
    ceiling = sum(w.cost for w in workers[:5])

    result = solve_cpsat(instance, CpSatConfig(time_limit_seconds=10, cost_ceiling=ceiling, feasibility_only=True))
    assert result.status == "OPTIMAL", result.status  # CP-SAT calls a feasibility-only model's first hit OPTIMAL
    assert result.solution.reported_cost <= ceiling

    verification = verify_solution(instance, result.solution)
    assert verification.is_valid, verification.violations


def test_feasibility_only_works_without_a_cost_ceiling():
    # The mode's whole point: "I have N workers, I don't want an
    # optimization, just give me a schedule that works." That must not
    # require a cost_ceiling to be set as well.
    #
    # History: the branch used to read `if cost_ceiling is not None and
    # feasibility_only`, so this exact call fell through to the Minimize
    # path and the flag did nothing -- silently, which is the worst way
    # for a feature to be unavailable.
    majors = [MajorTask(id=f"m{i}", type="feed", cell=(i % 10, (i // 10) % 10)) for i in range(6)]
    workers = [Worker(index=i, earliest_start=0) for i in range(4)]
    instance = Instance.compile(
        workers=workers, major_tasks=majors, warehouse_stock={Item.WHEAT: 100}, horizon=24
    )

    result = solve_cpsat(instance, CpSatConfig(time_limit_seconds=30, feasibility_only=True))
    assert result.solution is not None, result.status

    verification = verify_solution(
        instance,
        result.solution,
    )
    assert verification.is_valid, verification.violations

    # No objective was built, so this is "a" valid schedule, not the
    # cheapest one -- the optimizing solve is free to do strictly better.
    optimal = solve_cpsat(instance, CpSatConfig(time_limit_seconds=30))
    assert optimal.status == "OPTIMAL"
    assert result.solution.reported_cost >= optimal.solution.reported_cost


def test_cost_ceiling_without_feasibility_only_still_bounds_the_result():
    # Same idea, but keeping Minimize (a solution callback stops the
    # search early instead of dropping the objective) -- still must
    # never exceed the ceiling.
    major = MajorTask(id="feed1", type="feed", cell=NW)
    workers = [Worker(index=i, earliest_start=0) for i in range(8)]
    instance = Instance.compile(workers=workers, major_tasks=[major], warehouse_stock={Item.WHEAT: 5})
    ceiling = sum(w.cost for w in workers[:5])

    result = solve_cpsat(instance, CpSatConfig(time_limit_seconds=10, cost_ceiling=ceiling))
    assert result.status in ("OPTIMAL", "FEASIBLE"), result.status
    assert result.solution.reported_cost <= ceiling

    verification = verify_solution(instance, result.solution)
    assert verification.is_valid, verification.violations


def test_regression_an_active_worker_must_actually_carry_tasks():
    # Constraint 11's "second half" (see cpsat_solver.py): `u[w]` used to
    # be tied to `assign` in one direction only, so CP-SAT could set
    # `u[w] = 1` for a worker that received no tasks at all. The model
    # then considered a clean prefix of `u` to satisfy constraint 11 while
    # the *tasks* sat on a gappy subset -- which `verify.py` (which counts
    # a worker active only when its route has tasks) correctly rejected,
    # making `solve_cpsat` raise SolverProducedInvalidSolutionError.
    #
    # Reproduced before the fix at exactly this size: `u` was the prefix
    # 0..13 while tasks landed on {2,6,7,8,9,11,12,13}, leaving workers
    # 0,1,3,4,5,10 holding empty routes. It needs an instance big enough
    # that the solve stops at FEASIBLE rather than proving an optimum --
    # that is the regime where the objective stops masking the gap.
    majors = [MajorTask(id=f"m{i}", type="feed", cell=(i % 10, (i // 10) % 10)) for i in range(20)]
    workers = [Worker(index=i, earliest_start=0) for i in range(16)]
    instance = Instance.compile(
        workers=workers, major_tasks=majors, warehouse_stock={Item.WHEAT: 500}, horizon=24
    )

    result = solve_cpsat(instance, CpSatConfig(time_limit_seconds=20))  # raised before the fix
    assert result.solution is not None, result.status
    assert all(route.tasks for route in result.solution.routes), "an active worker carries no tasks"

    verification = verify_solution(
        instance, result.solution, entry_assign=None)
    assert verification.is_valid, verification.violations


def test_regression_feasibility_only_does_not_inflate_cost_with_empty_workers():
    # The same root cause, second symptom: `total_cost` sums fib over
    # `u`, so an active-but-empty worker was still charged. Under
    # `feasibility_only` there is no objective at all pushing `u` down,
    # so this was reachable at *any* size -- a 6-task instance reported
    # cost 2 for a schedule whose real cost was 1. A `cost_ceiling` could
    # then reject a solution genuinely under the ceiling.
    majors = [MajorTask(id=f"m{i}", type="feed", cell=(i % 10, i % 10)) for i in range(3)]
    workers = [Worker(index=i, earliest_start=0) for i in range(8)]
    instance = Instance.compile(
        workers=workers, major_tasks=majors, warehouse_stock={Item.WHEAT: 50}, horizon=24
    )
    ceiling = sum(w.cost for w in workers)

    result = solve_cpsat(
        instance, CpSatConfig(time_limit_seconds=10, cost_ceiling=ceiling, feasibility_only=True)
    )
    assert result.solution is not None, result.status
    assert all(route.tasks for route in result.solution.routes), "an active worker carries no tasks"

    # reported_cost must equal the cost of the workers that actually work
    real_cost = sum(Worker(index=r.worker_index, earliest_start=0).cost
                    for r in result.solution.routes if r.tasks)
    assert result.solution.reported_cost == real_cost


def test_cost_floor_forces_more_workers_than_the_unconstrained_optimum():
    # A hard cost >= floor constraint -- pure pruning, not a stopping
    # condition. The true optimum here is 1 worker (cost 0); a floor set
    # above that must force CP-SAT to use strictly more.
    #
    # History: this used to use a single `feed` major task, whose two
    # minor tasks are locked into one `single_worker_group` -- so only ONE
    # worker can ever carry work, and the only way to reach a cost of 1
    # was to activate a second, *empty* worker. That passed solely because
    # of the `u[w]`-without-tasks bug (see cpsat_solver.py's constraint-11
    # "second half" note); with that fixed the old form is correctly
    # INFEASIBLE. Two genuinely independent tasks are used instead, so a
    # real 2-worker split exists for the floor to force.
    tasks = [
        MinorTask(id="t1", cell=NW, action=Action.PASS),
        MinorTask(id="t2", cell=WAREHOUSE_ENTRY_CELLS["NE"], action=Action.PASS),
    ]
    workers = [Worker(index=i, earliest_start=0) for i in range(4)]
    instance = Instance.compile(workers=workers, standalone_minor_tasks=tasks)
    floor = workers[0].cost + workers[1].cost  # forces at least 2 active workers

    result = solve_cpsat(instance, CpSatConfig(time_limit_seconds=10, cost_floor=floor))
    assert result.status == "OPTIMAL", result.status
    assert result.solution.reported_cost >= floor
    # every reported route must be a worker that actually carries tasks
    assert all(route.tasks for route in result.solution.routes)
    assert len({r.worker_index for r in result.solution.routes}) >= 2

    verification = verify_solution(instance, result.solution)
    assert verification.is_valid, verification.violations


def test_warm_start_from_an_arbitrary_solution_object_still_reaches_the_optimum():
    # The whole point of warm_start is that it only reads the project's
    # own Solution/WorkerRoute/ScheduledTask shape -- it must work
    # exactly the same whether that Solution came from another solve, a
    # hand-built fixture, or (as here, standing in for "loaded from a
    # database") a Solution built from scratch by the test itself.
    (case,) = [c for c in HAND_SOLVED_CASES if c.name == "feed_single_worker_chain_needs_only_worker_0"]
    result = solve_cpsat(case.instance, CpSatConfig(time_limit_seconds=10, warm_start=case.example_optimal_solution))
    assert result.status == "OPTIMAL", result.status
    assert result.solution.reported_cost == case.optimal_cost

    verification = verify_solution(case.instance, result.solution)
    assert verification.is_valid, verification.violations


def test_warm_start_tolerates_stale_or_mismatched_task_and_worker_ids():
    # A warm start referencing a task id and a worker index that don't
    # exist in this instance/pool at all must be skipped, not crash --
    # this is exactly the "loaded from a cache built for a similar-but-
    # not-identical instance" case.
    major = MajorTask(id="feed1", type="feed", cell=NW)
    instance = Instance.compile(workers=[Worker(index=0, earliest_start=0)], major_tasks=[major], warehouse_stock={Item.WHEAT: 5})
    stale = Solution(
        routes=[
            WorkerRoute(
                worker_index=7,  # not in this instance's worker pool
                start_time=0,
                tasks=[
                    ScheduledTask(task_id="does_not_exist", exec_time=1),
                    ScheduledTask(task_id="feed1_acquire", exec_time=1, resolved_cell=NW, resolved_qty=1),
                ],
            )
        ],
        reported_cost=99,
    )
    result = solve_cpsat(instance, CpSatConfig(time_limit_seconds=10, warm_start=stale))
    assert result.status == "OPTIMAL", result.status
    assert result.solution.reported_cost == 0

    verification = verify_solution(instance, result.solution)
    assert verification.is_valid, verification.violations


def test_worker_count_lower_bound_never_exceeds_the_true_optimum():
    # The automatic lower bound is asserted into the model (`u[lb-1] == 1`),
    # so an over-estimate silently destroys optimality rather than failing
    # loudly. This is the instance where a first version did exactly that:
    # 20 `feed` majors, whose PICKUPs aggregate heavily. Counting each
    # single_worker_group as `len(group) + travel` (the way oxa_solver
    # does, correctly, for a solver that never aggregates) gave lb=7 and
    # forced a proven "optimum" of cost 20 -- against a true optimum of 1.
    majors = [MajorTask(id=f"m{i}", type="feed", cell=(i % 10, (i // 10) % 10)) for i in range(20)]
    workers = [Worker(index=i, earliest_start=0) for i in range(16)]
    instance = Instance.compile(
        workers=workers, major_tasks=majors, warehouse_stock={Item.WHEAT: 500}, horizon=24
    )
    aggregatable = _find_aggregatable_pickups(instance, {t.id: i for i, t in enumerate(instance.minor_tasks)})
    lb = _worker_count_lower_bound(instance, aggregatable)

    result = solve_cpsat(instance, CpSatConfig(time_limit_seconds=60))
    assert result.status == "OPTIMAL", result.status
    # cost == sum fib(0..max_active): 20 feeds across a 10x10 grid within
    # horizon 24 genuinely needs 11 workers; fib sum 0..10 == 143.
    from day.fibonacci import fibonacci_cost
    max_active = max(r.worker_index for r in result.solution.routes if r.tasks)
    assert result.solution.reported_cost == sum(
        fibonacci_cost(i) for i in range(max_active + 1)
    ), "cost must equal the engine's hire accounting"

    workers_used = len({r.worker_index for r in result.solution.routes if r.tasks})
    assert lb <= workers_used, f"lower bound {lb} exceeds the true optimum's {workers_used} workers"


def test_prefix_search_reaches_the_same_proven_optimum_as_the_monolithic_solve():
    # cpsat_binarySearch is exact, not a heuristic: constraint 11
    # makes the active set a prefix and fib is non-decreasing, so cost is
    # non-decreasing in prefix length and feasibility is monotone in it --
    # the smallest feasible prefix is provably optimal. It must therefore
    # agree with the monolithic Minimize wherever both prove optimality.
    for case in HAND_SOLVED_CASES:
        result = cpsat_binarySearch(case.instance, FAST)
        assert result.status == "OPTIMAL", f"{case.name}: {result.status}"
        assert result.solution.reported_cost == case.optimal_cost, case.name

        verification = verify_solution(case.instance, result.solution)
        assert verification.is_valid, (case.name, verification.violations)


def test_prefix_search_reports_infeasible_without_scanning_every_prefix():
    # No wheat anywhere, so FEED can never happen. The static source
    # check settles this before any solve (no prefix walk at all).
    feed = MinorTask(id="feed", cell=NW, action=Action.FEED, item=Item.WHEAT, qty=1)
    instance = Instance.compile(workers=[Worker(index=0, earliest_start=0)], standalone_minor_tasks=[feed])
    with pytest.raises(InfeasibleInputError):
        cpsat_binarySearch(instance, FAST)


def test_binary_search_pool_is_the_minimum_feasible_pool():
    """The min-worker objective, asserted directly (issue #14, F056).

    `cpsat_binarySearch` is the oracle: it probes *feasibility* under shrinking
    pool caps, so the smallest cap that admits a schedule **is** the optimum for
    this objective — that is the search's own invariant, and it needs no second
    solver to certify it. The check is therefore feasible at `k`, infeasible at
    `k − 1`.

    (The monolithic solve cannot be the reference here: in its default mode it
    builds no objective at all, so its reported cost is an arbitrary feasible
    schedule — 18 × 0 and 2 × 2 in 20 runs.)
    """
    # PICKUP aggregation interacts with the prefix search through the
    # lower bound, so exercise a genuinely aggregating instance too.
    majors = [MajorTask(id=f"m{i}", type="feed", cell=(i % 10, i % 10)) for i in range(4)]
    workers = [Worker(index=i, earliest_start=0) for i in range(6)]
    instance = Instance.compile(
        workers=workers, major_tasks=majors, warehouse_stock={Item.WHEAT: 50}, horizon=24
    )
    config = CpSatConfig(time_limit_seconds=30)
    prefix = cpsat_binarySearch(instance, config)

    assert prefix.status == "OPTIMAL" and prefix.solution is not None
    active = len(prefix.solution.routes)
    assert active >= 1
    verification = verify_solution(instance, prefix.solution)
    assert verification.is_valid, verification.violations
    assert prefix.solution.reported_cost is None or isinstance(
        prefix.solution.reported_cost, int)

    # one worker fewer must be infeasible: that is what makes `active` minimal
    smaller = solve_cpsat(instance, CpSatConfig(
        time_limit_seconds=30, feasibility_only=True,
        worker_pool_cap=active - 1, warm_start=None))
    assert smaller.solution is None, (
        f"{active - 1} workers also admit a schedule, so {active} was not "
        "the minimum pool")


def test_worker_pool_cap_truncates_the_pool_and_says_so():
    # The cap is an empirical estimate, not a proven bound, so an
    # INFEASIBLE produced under it must be distinguishable from a genuine
    # one -- otherwise a capped solve silently loses the optimum.
    # horizon=1 gives each worker exactly one action, and that action must
    # sit at distance 0 from its own entry cell. The placement rule hands
    # out NW, NE, SW, SE, NW, NE... in worker order, so one task on each of
    # those cells needs exactly 6 workers -- comfortably available from the
    # pool of 8, impossible under a cap of 3.
    cells = [
        WAREHOUSE_ENTRY_CELLS["NW"], WAREHOUSE_ENTRY_CELLS["NE"], WAREHOUSE_ENTRY_CELLS["SW"],
        WAREHOUSE_ENTRY_CELLS["SE"], WAREHOUSE_ENTRY_CELLS["NW"], WAREHOUSE_ENTRY_CELLS["NE"],
    ]
    tasks = [MinorTask(id=f"t{i}", cell=c, action=Action.PASS) for i, c in enumerate(cells)]
    workers = [Worker(index=i, earliest_start=0) for i in range(8)]
    instance = Instance.compile(workers=workers, standalone_minor_tasks=tasks, horizon=1)

    capped = solve_cpsat(instance, CpSatConfig(time_limit_seconds=10, worker_pool_cap=3))
    assert capped.status == "INFEASIBLE"
    assert capped.pool_capped_at == 3, "an INFEASIBLE under a cap must be flagged as such"

    uncapped = solve_cpsat(instance, CpSatConfig(time_limit_seconds=10, worker_pool_cap=None))
    assert uncapped.status == "OPTIMAL", "the same instance is solvable with the full pool"
    assert uncapped.pool_capped_at is None


def test_worker_pool_cap_is_not_flagged_when_it_truncates_nothing():
    # A cap larger than the offered pool changes nothing, so it must not
    # raise the "this answer is only valid under a cap" flag.
    major = MajorTask(id="feed1", type="feed", cell=NW)
    instance = Instance.compile(
        workers=[Worker(index=i, earliest_start=0) for i in range(2)],
        major_tasks=[major],
        warehouse_stock={Item.WHEAT: 5},
    )
    result = solve_cpsat(instance, CpSatConfig(time_limit_seconds=10, worker_pool_cap=11))
    assert result.status == "OPTIMAL"
    assert result.pool_capped_at is None


def test_infeasible_instance_reports_infeasible_status():
    # No wheat anywhere -- FEED can never happen. The static source
    # check rejects this before the model is built.
    feed = MinorTask(id="feed", cell=NW, action=Action.FEED, item=Item.WHEAT, qty=1)
    instance = Instance.compile(workers=[Worker(index=0, earliest_start=0)], standalone_minor_tasks=[feed])
    with pytest.raises(InfeasibleInputError):
        solve_cpsat(instance, FAST)


def test_global_stock_overrun_raises_before_solving():
    pickup = MinorTask(id="pickup", cell=None, action=Action.PICKUP, item=Item.WHEAT, qty=10)
    instance = Instance.compile(
        workers=[Worker(index=0, earliest_start=0)],
        standalone_minor_tasks=[pickup],
        warehouse_stock={Item.WHEAT: 3},
    )
    with pytest.raises(InfeasibleInputError):
        solve_cpsat(instance, FAST)


def test_time_limit_is_honored_and_still_finds_the_optimum_on_a_tiny_instance():
    task = MinorTask(id="t1", cell=NW, action=Action.PASS)
    instance = Instance.compile(workers=[Worker(index=0, earliest_start=0)], standalone_minor_tasks=[task])
    result = solve_cpsat(instance, CpSatConfig(time_limit_seconds=1))
    assert result.status == "OPTIMAL"
    assert result.wall_time_seconds < 1.5


def test_regression_drop_actually_resets_inventory_across_unrelated_chains():
    # instances too large for tests/brute_force.py: two concurrent
    # wet_harvst_plnt cycles (each ending in a DROP) plus two concurrent
    # single-worker feed chains, with an explicit precedence edge forcing
    # m3's FEED to happen only after m1's DROP -- nothing ties that DROP
    # to "its own" chain's worker, so CP-SAT was free to route worker 0
    # through worker 1's DROP mid-route. Since DROP wipes a worker's *entire*
    # inventory (see verify.py), that silently zeroed out wheat worker 0
    # had already picked up for its own FEED, and the old
    # AddReservoirConstraintWithActive-based model (which didn't know
    # DROP resets anything) had no way to notice: it returned an
    # "OPTIMAL" solution that verify_solution correctly rejected as
    # invalid (wheat inventory going negative). This is the regression
    # test for the fix (constraint 7's segmented, DROP-aware reservoir).
    majors = [
        MajorTask(id="m1", type="wet_harvst_plnt", cell=(1, 1), crop=Item.TOMATO, harvested_item=Item.TOMATO),
        MajorTask(id="m2", type="wet_harvst_plnt", cell=(8, 8), crop=Item.MELON, harvested_item=Item.MELON),
        MajorTask(id="m3", type="feed", cell=(2, 8)),
        MajorTask(id="m4", type="feed", cell=(8, 2)),
    ]
    drops = [
        MinorTask(id="m1_drop", cell=None, action=Action.DROP),
        MinorTask(id="m2_drop", cell=None, action=Action.DROP),
    ]
    workers = [Worker(index=i, earliest_start=0) for i in range(5)]
    instance = Instance.compile(
        workers=workers,
        major_tasks=majors,
        standalone_minor_tasks=drops,
        explicit_precedence=[("m1_harvest", "m1_drop"), ("m2_harvest", "m2_drop"), ("m1_drop", "m3_feed")],
        warehouse_stock={Item.WHEAT: 10},
        horizon=24,
    )
    result = solve_cpsat(instance, CpSatConfig(time_limit_seconds=30))
    assert result.status == "OPTIMAL", result.status

    # m3's and m4's acquire steps are both flexible-shed-location wheat
    # PICKUPs, so CP-SAT may legitimately route one worker through both
    # FEEDs and cover them with a single aggregated PICKUP -- see
    # test_pickup_aggregation.py for tests dedicated to that behavior.
    verification = verify_solution(
        instance, result.solution, entry_assign=None)
    assert verification.is_valid, verification.violations


def test_regression_active_workers_always_form_a_contiguous_prefix():
    # Constraint 11 (docs/problem-formulation.*.md): a real domain rule
    # missing from the formulation until a user's manual walkthrough of a
    # real instance caught it. Before the fix, u_w was unconstrained
    # outside of an equal-(cost,start) symmetry-break, so nothing stopped
    # CP-SAT from returning a non-optimal-but-valid-looking incumbent with
    # gaps in worker activation. Five independent tasks, each reachable
    # only by a specific one of five same-tier candidates within a tight
    # horizon, forces exactly 5 workers active -- if the fix regressed,
    # this is the shape of instance where a solver under time pressure
    # could plausibly skip one.
    cells = [(0, 0), (9, 9), (0, 9), (9, 0), (5, 5)]
    tasks = [MinorTask(id=f"t{i}", cell=c, action=Action.PASS) for i, c in enumerate(cells)]
    workers = [Worker(index=i, earliest_start=0) for i in range(5)]
    instance = Instance.compile(workers=workers, standalone_minor_tasks=tasks, horizon=9)

    result = solve_cpsat(instance, CpSatConfig(time_limit_seconds=30))
    assert result.status == "OPTIMAL", result.status

    verification = verify_solution(instance, result.solution)
    assert verification.is_valid, verification.violations

    active = sorted(route.worker_index for route in result.solution.routes)
    assert active == list(range(len(active)))


def test_regression_first_task_needs_its_own_turn_not_just_travel_time():
    # Constraint 3 (docs/problem-formulation.*.md): used to let a worker's
    # first task execute for free at sigma[w] + dist, with no +1 for the
    # action's own turn -- one turn cheaper than every later task. With
    # horizon=1, a worker starting at 0 can really fit only one action
    # (exec_time=1); two independent same-instant tasks can't both land on
    # it, so the true optimum needs workers 0 and 1 (cost 0+1=1). Under
    # the old off-by-one, one worker alone looked sufficient (t1 at
    # "t=0", t2 at "t=1" both <= horizon=1), reporting a wrong OPTIMAL
    # cost of 0 -- exactly what this regression test would catch.
    task1 = MinorTask(id="t1", cell=NW, action=Action.PASS)
    task2 = MinorTask(id="t2", cell=WAREHOUSE_ENTRY_CELLS["NE"], action=Action.PASS)
    workers = [Worker(index=i, earliest_start=0) for i in range(2)]
    instance = Instance.compile(workers=workers, standalone_minor_tasks=[task1, task2], horizon=1)

    result = solve_cpsat(instance, FAST)
    assert result.status == "OPTIMAL", result.status
    assert result.solution.reported_cost == 1


def test_pickup_aggregation_lets_two_feeds_share_one_pickup_when_needed():
    # Two feeds at the shed's own entry cell (distance 0 to everything
    # else at the shed): unaggregated, worker 0 needs 4 actions
    # (acquire1, feed1, acquire2, feed2), each its own turn -- tau 1..4 --
    # so horizon=3 makes that impossible. Aggregated (one PICKUP for both,
    # qty=2), only 3 actions are needed -- tau 1..3 -- which fits exactly.
    # A single worker candidate is offered, so this is only feasible at
    # all if CP-SAT actually exploits aggregation, not just prefers it.
    majors = [
        MajorTask(id="feed1", type="feed", cell=NW),
        MajorTask(id="feed2", type="feed", cell=NW),
    ]
    instance = Instance.compile(
        workers=[Worker(index=0, earliest_start=0)],
        major_tasks=majors,
        warehouse_stock={Item.WHEAT: 2},
        horizon=3,
    )
    result = solve_cpsat(instance, FAST)
    assert result.status == "OPTIMAL", result.status
    assert result.solution.reported_cost == 0

    (route,) = result.solution.routes
    scheduled_ids = {t.task_id for t in route.tasks}
    acquire_ids = {"feed1_acquire", "feed2_acquire"}
    scheduled_acquires = scheduled_ids & acquire_ids
    assert len(scheduled_acquires) == 1, scheduled_ids  # exactly one PICKUP covers both feeds
    (kept,) = scheduled_acquires
    (kept_task,) = [t for t in route.tasks if t.task_id == kept]
    assert kept_task.resolved_qty == 2

    verification = verify_solution(instance, result.solution, entry_assign=None)
    assert verification.is_valid, verification.violations


def test_pickup_aggregation_is_optional_when_the_schedule_has_room_anyway():
    # Same two feeds, but a horizon generous enough that 4 separate
    # actions fit without any aggregation. Nothing in the objective
    # rewards aggregating (it only counts active workers), so CP-SAT is
    # free to return either shape -- this only confirms both remain valid
    # (and self-consistent) once aggregation is a genuine possibility,
    # not that one is preferred over the other.
    majors = [
        MajorTask(id="feed1", type="feed", cell=NW),
        MajorTask(id="feed2", type="feed", cell=NW),
    ]
    instance = Instance.compile(
        workers=[Worker(index=0, earliest_start=0)],
        major_tasks=majors,
        warehouse_stock={Item.WHEAT: 2},
        horizon=24,
    )
    result = solve_cpsat(instance, FAST)
    assert result.status == "OPTIMAL", result.status
    assert result.solution.reported_cost == 0

    verification = verify_solution(
        instance, result.solution, entry_assign=None)
    assert verification.is_valid, verification.violations
