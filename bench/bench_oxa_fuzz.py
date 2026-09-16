"""Audit of OXA's verdicts on randomised instances (docs/F054, review pass).

Usage:
    .venv/bin/python -m bench.bench_oxa_fuzz                 # 400 seeded instances
    .venv/bin/python -m bench.bench_oxa_fuzz --oracle        # + CP-SAT confirms
    .venv/bin/python -m bench.bench_oxa_fuzz --seeds 0:100

The generator is seeded, so the same seed is the same instance on every run --
`--seeds a:b` narrows the range. Each instance mixes single tasks, same-cell
and cross-cell precedence pairs, and one item-consuming chain with its paired
acquire and single-worker group, over random horizons and worker pools.

The audit asserts the two things a verdict must satisfy:
  * every answer that is not INFEASIBLE carries a schedule `verify_solution`
    accepts (a rejected schedule is as unusable as no schedule);
  * with --oracle, every INFEASIBLE answer is confirmed by `cpsat_binarySearch`
    -- the F054 defect was exactly an INFEASIBLE the oracle could schedule.

`docs/F054`'s differential numbers come from running this script with the
pre-fix blob in place: `git show <rev>:secretary/solvers/oxa_solver.py >
/tmp/old.py`, copy it over the module, run, restore.

A non-zero exit is the audit reporting a defect, not a harness error: today it
reports the `single_worker_group`/entry-cell placement class of `docs/F054`
(`first task ... reachable too early from entry`), which is open there.
"""
from __future__ import annotations

import argparse
import random
from collections import Counter

from secretary.models import (Cell, Instance, Item, MinorActionType, MinorTask, Worker)
from secretary.solvers.oxa_solver import OxaConfig, solve_oxa
from secretary.verify import verify_solution

CELLS = [Cell(x, y) for x in (0, 2, 4, 9) for y in (0, 3, 4, 9)]


def build(seed: int) -> Instance:
    rng = random.Random(seed)
    workers = [Worker(index=i, earliest_start=0 if i == 0 else rng.choice([0, 1]))
               for i in range(rng.randint(1, 4))]
    horizon = rng.choice([6, 8, 10, 12, 16, 24])
    minors: list[MinorTask] = []
    precedence: list[tuple[str, str]] = []
    groups: list[list[str]] = []
    stock: dict[Item, int] = {}

    for c in range(rng.randint(1, 3)):
        shape = rng.choice(['solo', 'pair_same', 'pair_cross', 'triple_mixed'])
        cell_a, cell_b = rng.choice(CELLS), rng.choice(CELLS)
        if shape == 'solo':
            minors.append(MinorTask(id=f'c{c}_a', cell=cell_a, action=MinorActionType.PASS))
        elif shape == 'pair_same':
            minors.append(MinorTask(id=f'c{c}_a', cell=cell_a, action=MinorActionType.PASS))
            minors.append(MinorTask(id=f'c{c}_b', cell=cell_a, action=MinorActionType.PASS))
            precedence.append((f'c{c}_a', f'c{c}_b'))
        elif shape == 'pair_cross':
            minors.append(MinorTask(id=f'c{c}_a', cell=cell_a, action=MinorActionType.PASS))
            minors.append(MinorTask(id=f'c{c}_b', cell=cell_b, action=MinorActionType.PASS))
            precedence.append((f'c{c}_a', f'c{c}_b'))
        else:
            item = rng.choice([Item.WHEAT, Item.FERTILIZER])
            action = MinorActionType.FEED if item is Item.WHEAT else MinorActionType.FERTILIZE
            acq = f'c{c}_acq'
            minors.append(MinorTask(id=acq, cell=None, action=MinorActionType.PICKUP,
                                    item=item, qty=1))
            minors.append(MinorTask(id=f'c{c}_cons', cell=cell_a, action=action,
                                    item=item, qty=1))
            minors.append(MinorTask(id=f'c{c}_tail', cell=cell_b, action=MinorActionType.PASS))
            precedence += [(acq, f'c{c}_cons'), (f'c{c}_cons', f'c{c}_tail')]
            groups.append([acq, f'c{c}_cons'])
            stock[item] = stock.get(item, 0) + rng.randint(1, 3)
    if rng.random() < 0.3:
        minors.append(MinorTask(id='extra', cell=rng.choice(CELLS), action=MinorActionType.PASS))

    return Instance.compile(workers=workers, standalone_minor_tasks=minors,
                            explicit_precedence=precedence,
                            explicit_single_worker_groups=groups,
                            warehouse_stock=stock, horizon=horizon)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds', default='0:400', help='seed range "a:b" (default 0:400)')
    parser.add_argument('--oracle', action='store_true',
                        help='confirm every INFEASIBLE verdict with CP-SAT (slow, needs ortools)')
    parser.add_argument('--oracle-seconds', type=float, default=5.0)
    parser.add_argument('--dump', help='write per-seed statuses to this JSON file '
                                       '(for a differential run against another solver revision)')
    args = parser.parse_args()
    first, last = (int(part) for part in args.seeds.split(':'))

    oracle = None
    if args.oracle:
        from secretary.solvers.cpsat_solver import CpSatConfig, cpsat_binarySearch
        oracle = CpSatConfig(time_limit_seconds=args.oracle_seconds)

    statuses: Counter[str] = Counter()
    rejected: list[int] = []
    unconfirmed: list[tuple[int, str]] = []
    per_seed: dict[int, str] = {}
    for seed in range(first, last):
        instance = build(seed)
        result = solve_oxa(instance, OxaConfig(min_workers=1))
        statuses[result.status] += 1
        per_seed[seed] = result.status
        if result.solution is None:
            if oracle is not None and cpsat_binarySearch(instance, oracle).solution is not None:
                unconfirmed.append((seed, result.status))
            continue
        verification = verify_solution(instance, result.solution)
        if not verification.is_valid:
            rejected.append(seed)
            print(f'  seed {seed}: {result.status} rejected by verify_solution '
                  f'{verification.violations[:2]}')
    print(f'seeds {first}..{last - 1}: {dict(statuses)}')
    print(f'schedules rejected by the verifier: {len(rejected)} {rejected}')
    if oracle is not None:
        print(f'INFEASIBLE verdicts the oracle can schedule: {len(unconfirmed)} {unconfirmed}')
    if args.dump:
        import json
        with open(args.dump, 'w') as handle:
            json.dump({str(seed): status for seed, status in per_seed.items()}, handle)
        print(f'per-seed statuses written to {args.dump}')
    return 1 if (rejected or unconfirmed) else 0


if __name__ == '__main__':
    raise SystemExit(main())
