"""docs/F057's numbers must stay true and repeatable (R005), and OXA's verdict
must be a function of the instance, not of the interpreter's hash order.

Three guards, each written against a defect that was measured:

1. `test_the_audit_reproduces_the_statuses_docs_F057_reports` runs the
   committed instrument (`bench/bench_oxa_fuzz.py`) over its 400 seeds and
   pins its answer. The number in the document and the number this test
   asserts are the same measurement; if the solver or the generator moves,
   the document's number moves with it, loudly instead of silently.

2. `test_the_verdicts_do_not_depend_on_the_interpreter_hash_order` runs the
   same audit under two different `PYTHONHASHSEED` values and requires
   byte-identical dumps. R007 failing direction: revert the candidate key in
   `_build_worker_route` to `(cost, tie_breaker)` -- one line -- and this test
   reddens, because `available` is built from the `remaining_targets` *set*,
   so the greedy's tie-breaks follow the hash order. That is why the numbers
   in this PR's first version could not be reproduced: the same 400 instances
   read `OPTIMAL 97 / FEASIBLE 92 / INFEASIBLE 208` on one hash seed and
   `97 / 94 / 207` on another.

3. `test_a_side_task_cannot_strand_a_chain` pins fuzz seed 307, the instance
   the hash-order fix exposed: the greedy gave worker 0 the standalone tile
   and the last worker the chain's head, so `c1_b` had no worker left to run
   it and `solve_oxa` answered INFEASIBLE for a day the oracle schedules. The
   candidate order now prefers the more constrained task on a tie
   (`key = (cost, tie_breaker, -tail, tid)`).
"""
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

from day.models import Cell, Instance, Item, Action, MinorTask, Worker
from day.solvers.cpsat_solver import CpSatConfig, cpsat_binarySearch
from day.solvers.oxa_solver import OxaConfig, solve_oxa
from day.verify import verify_solution

ROOT = Path(__file__).resolve().parents[2]

# docs/F057's reviewed row. The audit is clean on this branch: the three
# entry-cell rejections (93, 182, 205) are gone (the placement fixed point),
# seed 179's stranded successor is now scheduled (the repeated pass) and no
# `INVALID_SOLUTION` is left, so the default run exits clean. The INFEASIBLE
# verdicts the oracle schedules are F057's fourth class — 14 remain — and only
# show up under `--oracle`.
AUDIT_STATUSES = {"OPTIMAL": 97, "FEASIBLE": 102, "INFEASIBLE": 201}
AUDIT_REJECTED = []


def run_audit(dump: Path, hash_seed):
    env = dict(os.environ)
    if hash_seed is None:
        env.pop("PYTHONHASHSEED", None)  # the interpreter picks a random seed
    else:
        env["PYTHONHASHSEED"] = hash_seed
    proc = subprocess.run(
        [sys.executable, "-m", "bench.bench_oxa_fuzz", "--dump", str(dump)],
        cwd=ROOT, env=env, capture_output=True, text=True,
    )
    assert "seeds 0..399" in proc.stdout, proc.stdout + proc.stderr
    return json.loads(dump.read_text()), proc


def test_the_audit_reproduces_the_statuses_docs_F057_reports(tmp_path):
    per_seed, proc = run_audit(tmp_path / "audit.json", "0")

    assert Counter(per_seed.values()) == AUDIT_STATUSES, Counter(per_seed.values())
    assert proc.returncode == 0, (
        "the audit exits non-zero when it has a defect to report; with the "
        "entry-cell class fixed and no --oracle run, it has none"
    )
    line = next(line for line in proc.stdout.splitlines() if "rejected by the verifier" in line)
    assert line.strip() == f"schedules rejected by the verifier: 0 {AUDIT_REJECTED}", line


def test_the_entry_cell_is_assigned_over_the_routed_workers():
    """docs/F057's third class: the placement rule follows the ROUTED hands.

    `verify_solution` recomputes the entry cells over the workers that carry
    tasks (constraint 10), so a hand the greedy leaves idle does not hold a
    cell. Here worker 0 cannot reach the task inside a 5-turn horizon, worker 1
    takes it from NE -- and with the placement recomputed for worker 1 alone
    that hand holds NW, from which the task is one turn out of reach. The
    solver used to emit worker 1's route from NE anyway and answer
    INVALID_SOLUTION ("worker 1: first task 't0' reachable too early from
    entry"); the day is genuinely impossible (the oracle says INFEASIBLE too),
    so the honest verdict is INFEASIBLE.
    R007 failing direction: run one dispatch instead of the fixed point and
    this reddens with that INVALID_SOLUTION.
    """
    instance = Instance.compile(
        workers=[Worker(index=0, earliest_start=0), Worker(index=1, earliest_start=0)],
        standalone_minor_tasks=[MinorTask(id="t0", cell=Cell(5, 0), action=Action.PASS)],
        horizon=5,
    )
    result = solve_oxa(instance, OxaConfig(min_workers=1))

    assert result.status == "INFEASIBLE", result.status
    assert cpsat_binarySearch(instance, CpSatConfig(time_limit_seconds=10)).solution is None


def test_a_stranded_successor_goes_to_an_earlier_hand_that_still_waits():
    """docs/F057's fourth class, the idle-hand half: the pass repeats.

    Worker 0 cannot reach the predecessor `t0` inside a 10-turn horizon and
    refuses it (`10 + 1 > 10`), worker 1 takes it at hour 9 — and worker 0's
    single pass was already over, so the successor `t1` was stranded and the
    solver answered INFEASIBLE for a day the oracle schedules as
    `w0 -> t1 at 10, w1 -> t0 at 9`. Repeating the dispatch over the hands that
    have no route yet lets worker 0 take `t1`, and the greedy finds that exact
    day. R007 failing direction: one pass only (as before that change) and this
    reddens with INFEASIBLE.
    """
    instance = Instance.compile(
        workers=[Worker(index=0, earliest_start=0), Worker(index=1, earliest_start=0)],
        standalone_minor_tasks=[
            MinorTask(id="t0", cell=Cell(9, 0), action=Action.PASS),
            MinorTask(id="t1", cell=Cell(0, 0), action=Action.PASS),
        ],
        explicit_precedence=[("t0", "t1")],
        horizon=10,
    )
    result = solve_oxa(instance, OxaConfig(min_workers=1))

    assert result.status in ("OPTIMAL", "FEASIBLE"), result.status
    assert verify_solution(instance, result.solution).is_valid
    times = {task.task_id: task.exec_time
             for route in result.solution.routes for task in route.tasks}
    assert times == {"t0": 9, "t1": 10}, times
    assert cpsat_binarySearch(instance, CpSatConfig(time_limit_seconds=10)).solution is not None


def test_the_verdicts_do_not_depend_on_the_interpreter_hash_order(tmp_path):
    pinned, _ = run_audit(tmp_path / "pinned.json", "0")
    randomised, _ = run_audit(tmp_path / "randomised.json", None)

    assert pinned == randomised, (
        "the same instances answered differently under another hash order: "
        f"{sorted(seed for seed in pinned if pinned[seed] != randomised[seed])[:10]}"
    )


def test_a_side_task_cannot_strand_a_chain():
    """Fuzz seed 307: two workers could serve the chain, the greedy spent them.

    Worker 0's cheapest commit is a tie between the standalone tile `c0_a`
    and the chain head `c1_a`; taking `c0_a` leaves `c1_a` to the last worker
    (index 2), whose successor `c1_b` is then out of reach -- INFEASIBLE with
    no worker left. The oracle schedules the day (w0 `c1_a` @ 11, w1 `c1_b`
    @ 12, w2 `c0_a` @ 12), so the INFEASIBLE was a construction failure.
    """
    instance = Instance.compile(
        workers=[Worker(index=0, earliest_start=0),
                 Worker(index=1, earliest_start=1),
                 Worker(index=2, earliest_start=1)],
        standalone_minor_tasks=[
            MinorTask(id="c0_a", cell=Cell(9, 0), action=Action.PASS),
            MinorTask(id="c1_a", cell=Cell(0, 9), action=Action.PASS),
            MinorTask(id="c1_b", cell=Cell(2, 4), action=Action.PASS),
        ],
        explicit_precedence=[("c1_a", "c1_b")],
        horizon=12,
    )
    result = solve_oxa(instance, OxaConfig(min_workers=1))

    assert result.status in ("OPTIMAL", "FEASIBLE"), result.status
    assert verify_solution(instance, result.solution).is_valid
    oracle = cpsat_binarySearch(instance, CpSatConfig(time_limit_seconds=10))
    assert oracle.solution is not None, "the day is schedulable; the greedy must find one"
