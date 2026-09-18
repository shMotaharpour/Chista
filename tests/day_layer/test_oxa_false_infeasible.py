"""R007 guard: OXA may never call a feasible day INFEASIBLE.

The sweep is the one from issue #14: ten workers (the first starting at
hour 0, the rest at hour 1), horizon 24, one major task per chain laid out
on a 5x5 grid, n = 6..25 per major-task kind. Every kind's chains are
independent, so the day's only binding resource is worker turns -- and
with 10 x 24 of those against at most 4n actions the answer is a
capacity question, not a construction one.

Measured before the fix (docs/F057) with the oracle from
tests/day_layer/test_cpsat_solver.py's entry point, `cpsat_binarySearch`. The
pre-fix `_build_worker_route` iterated the `remaining_targets` set, so its
verdicts moved with the interpreter's hash order -- the sets below are the
`PYTHONHASHSEED=0` run, and a reader comparing them must pin the same seed
(`test_oxa_reproducibility.py` holds the fixed solver to be seed-free):

| kind | `solve_oxa` INFEASIBLE at n = | falsified by the oracle |
|---|---|---|
| plnt | 15..19, 23..25 | 8 (all of them) |
| wet_harvst | 13, 15..19, 24 | 7 (all of them) |
| wet_harvst_plnt | 8..25 | 10: n = 8..17; n = 18..25 unproven in the 10 s slice |
| frtz_water | 7, 8, 9, 20..25 | 6: n = 7, 8, 9, 20, 21, 23; 22, 24, 25 unproven |
| feed, frtz, place_animal | (never) | — |

R007 failing direction: with `day/solvers/oxa_solver.py` reverted to
its pre-fix state this module reddens. Measured at `PYTHONHASHSEED=0`:
`pytest -x -q` stops on the sweep test's `plnt` case (`1 failed, 2 passed`),
and the eight pinned pairs all fail (`8 failed, 13 deselected`) with the
guard's own message naming the kind, the n and the oracle's answer.

Two more findings came out of the pre-PR review (both measured, docs/F057):
the tail gap must charge travel only for pairs that ONE worker has to serve
(`single_worker_group`), because a plain precedence pair can be split across
two workers -- charging travel there reddens
`test_a_cross_cell_chain_can_be_served_by_two_workers`; and the day's
`hired` is the engine's append-only payroll `fib(0..max_active)`,
not the sum over the routes that carry tasks (which reddens
`test_a_day_that_skips_lower_index_hands_pays_for_them`).
"""
import pytest

from agent.world.rules import hire_cost
from agent.wsr.models import (Cell, Instance, Item, MajorTask, Action,
                              MinorTask, ScheduledTask, Solution, Worker, WorkerRoute)
from agent.wsr.solvers.cpsat_solver import CpSatConfig, cpsat_binarySearch
from agent.wsr.solvers.oxa_solver import OxaConfig, solve_oxa
from offline_lab.verify import verify_solution

STOCK = {Item.WHEAT: 200, Item.FERTILIZER: 200, Item.COW: 50, Item.SHEEP: 50, Item.GOOSE: 50}

KINDS = ("feed", "frtz", "plnt", "wet_harvst", "wet_harvst_plnt", "frtz_water", "place_animal")
N_RANGE = range(6, 26)
ORACLE = CpSatConfig(time_limit_seconds=10)


def build(n: int, kind: str, workers: int = 10) -> Instance:
    """The sweep's instance: n chains of `kind`, one per (n % 5, n // 5) cell."""
    return Instance.compile(
        workers=[Worker(index=k, earliest_start=0 if k == 0 else 1) for k in range(workers)],
        major_tasks=[
            MajorTask(id=f"m{k}", type=kind, cell=Cell(x=k % 5, y=(k // 5) % 5),
                      item=(Item.COW if kind == "place_animal" else None))
            for k in range(n)
        ],
        warehouse_stock=STOCK,
        horizon=24,
    )


@pytest.mark.parametrize("kind", KINDS)
def test_the_sweep_never_calls_a_feasible_day_infeasible(kind):
    for n in N_RANGE:
        instance = build(n, kind)
        result = solve_oxa(instance, OxaConfig(min_workers=1))

        if result.status == "INFEASIBLE":
            # an INFEASIBLE verdict is only allowed when the exact oracle
            # cannot schedule the day either -- otherwise it is a
            # construction failure with idle workers (docs/F057)
            oracle = cpsat_binarySearch(instance, ORACLE)
            assert oracle.solution is None, (
                f"{kind} n={n}: OXA said INFEASIBLE but the oracle admits a "
                f"schedule ({oracle.status}) -- the greedy orphaned a chain "
                f"instead of leaving it to another worker"
            )
            continue

        assert result.status in ("OPTIMAL", "FEASIBLE"), (kind, n, result.status)
        verification = verify_solution(instance, result.solution)
        assert verification.is_valid, (kind, n, verification.violations)


@pytest.mark.parametrize("kind,n", [
    # the exact instances of the finding, each measured false-INFEASIBLE
    # before the fix with the oracle proving them feasible, at
    # PYTHONHASHSEED=0 (the pre-fix blob's verdicts follow the hash order;
    # see the table above and docs/F057's reproducibility section)
    ("plnt", 15),
    ("plnt", 25),
    ("wet_harvst", 15),
    ("wet_harvst", 24),
    ("wet_harvst_plnt", 8),
    ("wet_harvst_plnt", 17),
    ("frtz_water", 7),
    ("frtz_water", 23),
])
def test_the_measured_false_infeasible_instances_schedule_and_verify(kind, n):
    instance = build(n, kind)
    result = solve_oxa(instance, OxaConfig(min_workers=1))

    assert result.status in ("OPTIMAL", "FEASIBLE"), (kind, n, result.status)
    verification = verify_solution(instance, result.solution)
    assert verification.is_valid, (kind, n, verification.violations)
    used = len([route for route in result.solution.routes if route.tasks])
    assert used < 10, (kind, n, used)  # the workers that sat idle were the tell


@pytest.mark.parametrize("kind,n", [("plnt", 15), ("wet_harvst_plnt", 8)])
def test_a_bigger_worker_pool_does_not_change_the_verdict(kind, n):
    # issue #14 read "the greedy construction fails for a reason unrelated
    # to r -- the prefix-size search walks r upward and reads it as 'no
    # prefix works'". There is no r-search left in this module, and the
    # verdict cannot be bought with workers either: the committed exec_time
    # is a hard global lower bound for the successors, so extra workers
    # cannot take the orphaned chain either. Measured, not argued.
    statuses = {}
    for workers in (4, 8, 12, 16):
        instance = build(n, kind, workers=workers)
        result = solve_oxa(instance, OxaConfig(min_workers=1))
        statuses[workers] = result.status
        assert result.status in ("OPTIMAL", "FEASIBLE"), (kind, n, workers, result.status)
        verification = verify_solution(instance, result.solution)
        assert verification.is_valid, (kind, n, workers, verification.violations)
    assert len(set(statuses.values())) == 1, statuses


# ---------------------------------------------------------------- review findings


def test_a_cross_cell_chain_can_be_served_by_two_workers():
    """A plain precedence pair may be split across two hands.

    `p` at (0,0) then `s` at (9,9), horizon 10, two workers: w0 runs p at
    t=9 and w1 runs s at t=10, and the oracle calls that day OPTIMAL. A tail
    that charges the successor the travel from p's cell reads the chain as
    needing 28 turns and answers INFEASIBLE for a schedulable day -- the
    first version of the guard did exactly that (docs/F057, review pass).
    """
    instance = Instance.compile(
        workers=[Worker(index=0, earliest_start=0), Worker(index=1, earliest_start=0)],
        standalone_minor_tasks=[
            MinorTask(id="p", cell=Cell(0, 0), action=Action.PASS),
            MinorTask(id="s", cell=Cell(9, 9), action=Action.PASS),
        ],
        explicit_precedence=[("p", "s")],
        horizon=10,
    )
    result = solve_oxa(instance, OxaConfig(min_workers=1))

    assert result.status in ("OPTIMAL", "FEASIBLE"), result.status
    verification = verify_solution(instance, result.solution)
    assert verification.is_valid, verification.violations
    assert cpsat_binarySearch(instance, ORACLE).solution is not None  # provably schedulable


def test_a_day_that_skips_lower_index_hands_pays_for_them():
    """Engine accounting: hiring is append-only (verify.py's COST ACCOUNTING).

    Four hands are offered and the greedy may well give the day to hand 3 with
    hand 2 idle (or to hand 2 and skip nothing -- the route set depends on
    iteration order, both are valid). Whatever it picks, `hired` must
    be the engine's payroll `fib(0..max_active)`: reporting the sum over the
    routes that carry tasks under-reported a day with a gap and made the
    verifier answer INVALID_SOLUTION for an otherwise legal schedule
    (docs/F057, review pass).
    """
    instance = Instance.compile(
        workers=[Worker(index=i, earliest_start=start)
                 for i, start in ((0, 0), (1, 1), (2, 1), (3, 0))],
        standalone_minor_tasks=[
            MinorTask(id="c0_a", cell=Cell(4, 4), action=Action.PASS),
            MinorTask(id="c1_a", cell=Cell(4, 4), action=Action.PASS),
            MinorTask(id="c2_acq", cell=None, action=Action.PICKUP,
                      item=Item.FERTILIZER, qty=1),
            MinorTask(id="c2_cons", cell=Cell(2, 3), action=Action.FERTILIZE,
                      item=Item.FERTILIZER, qty=1),
            MinorTask(id="c2_tail", cell=Cell(9, 4), action=Action.PASS),
            MinorTask(id="extra", cell=Cell(0, 3), action=Action.PASS),
        ],
        explicit_precedence=[("c2_acq", "c2_cons"), ("c2_cons", "c2_tail")],
        explicit_single_worker_groups=[["c2_acq", "c2_cons"]],
        warehouse_stock={Item.FERTILIZER: 1},
        horizon=8,
    )
    result = solve_oxa(instance, OxaConfig(min_workers=1))

    assert result.status in ("OPTIMAL", "FEASIBLE"), result.status
    verification = verify_solution(instance, result.solution)
    assert verification.is_valid, verification.violations
    max_active = max(route.worker_index for route in result.solution.routes)
    payroll = sum(hire_cost(i) for i in range(max_active + 1))
    assert result.solution.hired == payroll
    assert verification.total_cost == payroll


def test_the_engine_charges_the_idle_hands_below_the_top_of_the_day():
    """The payroll rule, pinned deterministically through the verifier.

    One route on hand 3 with hands 0..2 idle is a legal day, and it pays for
    all four (fib(0..3) = 4). Reporting hand 3's fib alone (2) is the defect
    the test above caught in the solver.
    """
    instance = Instance.compile(
        workers=[Worker(index=i, earliest_start=0) for i in range(4)],
        standalone_minor_tasks=[MinorTask(id="t", cell=Cell(4, 4),
                                          action=Action.PASS)],
        horizon=8,
    )
    routes = [WorkerRoute(worker_index=3, start_time=0, start_cell=Cell(4, 4),
                          tasks=[ScheduledTask(task_id="t", exec_time=1,
                                               resolved_cell=Cell(4, 4))])]
    paid = verify_solution(instance, Solution(routes=routes, hired=4))
    assert paid.is_valid, paid.violations
    assert paid.total_cost == 4
    under = verify_solution(instance, Solution(routes=routes, hired=2))
    assert not under.is_valid and "hired" in under.violations[0]


def test_the_oracle_confirmation_agrees_on_a_genuinely_infeasible_day():
    """The sweep's confirmation branch is not a rubber stamp.

    horizon=1 with a chain that needs two turns: OXA answers INFEASIBLE and
    the oracle finds no schedule either, so the branch that *authorises* an
    INFEASIBLE verdict runs on green runs too (R007's warning about guards
    that only exercise the path production never takes).

    The oracle *decides* this day -- it comes back `INFEASIBLE`, not `UNKNOWN`
    -- so the control cannot be satisfied by a timeout. That distinction is
    the guard's honest limit: on a day the oracle cannot decide inside its
    slice (`UNKNOWN`, `solution is None`) the sweep's assertion abstains rather
    than confirming, and the pre-fix sweep has eight such rows
    (`wet_harvst_plnt` 18..25).
    """
    instance = Instance.compile(
        workers=[Worker(index=0, earliest_start=0)],
        standalone_minor_tasks=[
            MinorTask(id="a", cell=Cell(4, 4), action=Action.PASS),
            MinorTask(id="b", cell=Cell(4, 4), action=Action.PASS),
        ],
        explicit_precedence=[("a", "b")],
        horizon=1,
    )
    assert solve_oxa(instance, OxaConfig(min_workers=1)).status == "INFEASIBLE"
    assert cpsat_binarySearch(instance, ORACLE).solution is None

