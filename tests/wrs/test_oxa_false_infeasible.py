"""R007 guard: OXA may never call a feasible day INFEASIBLE.

The sweep is the one from issue #14: ten workers (the first starting at
hour 0, the rest at hour 1), horizon 24, one major task per chain laid out
on a 5x5 grid, n = 6..25 per major-task kind. Every kind's chains are
independent, so the day's only binding resource is worker turns -- and
with 10 x 24 of those against at most 4n actions the answer is a
capacity question, not a construction one.

Measured before the fix (docs/F054) with the oracle from
tests/wrs/test_cpsat_solver.py's entry point, `cpsat_binarySearch`:

| kind | false INFEASIBLE at n = |
|---|---|
| plnt | 15..25 |
| wet_harvst | 15..19, 22 |
| wet_harvst_plnt | 8..17 |
| frtz_water | 7, 8, 9, 18..25 |
| feed, frtz, place_animal | (never) |

R007 failing direction: with `secretary/solvers/oxa_solver.py` reverted to
its pre-fix state this module fails with the false verdicts listed above;
the guard's own message names the kind, the n and the oracle's answer.
"""
import pytest

from secretary.models import Cell, Instance, Item, MajorTask, Worker
from secretary.solvers.cpsat_solver import CpSatConfig, cpsat_binarySearch
from secretary.solvers.oxa_solver import OxaConfig, solve_oxa
from secretary.verify import verify_solution

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
            # construction failure with idle workers (docs/F054)
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
    # before the fix with the oracle proving them feasible
    ("plnt", 15),
    ("plnt", 25),
    ("wet_harvst", 15),
    ("wet_harvst", 22),
    ("wet_harvst_plnt", 8),
    ("wet_harvst_plnt", 17),
    ("frtz_water", 7),
    ("frtz_water", 25),
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
