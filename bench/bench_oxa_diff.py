"""Differential of two `bench_oxa_fuzz --dump` runs (docs/F057's tables).

Usage:
    .venv/bin/python -m bench.bench_oxa_diff OLD.json NEW.json

The two files are per-seed status maps written by
`bench/bench_oxa_fuzz.py --dump`, from two different solver revisions. This
script is the *source* of the "N regressions / M improvements" lines in
docs/F057: the counts are a measurement, and a measurement needs the
procedure that produced it committed next to the number (R005).

Definition (fixed here so two readers cannot disagree):

  * regression  -- the seed's OLD status is a usable day (OPTIMAL or
                   FEASIBLE) and its NEW status is not (INFEASIBLE or
                   INVALID_SOLUTION). A schedule the agent could have
                   worked with stopped being returned.
  * improvement -- the reverse: unusable became usable.
  * both-unusable -- OLD and NEW are both unusable but for a different
                   reason (e.g. INFEASIBLE -> INVALID_SOLUTION). Reported
                   separately, because it is neither: the day was lost
                   before and is lost now.

Cost is deliberately not part of the comparison: the bench dumps statuses.
`bench_oxa_cost_audit`-style cost diffs need the solutions back, which is
what a later revision of this script would take from the solver directly.

Exit code is 1 when the new revision regresses any seed, so the differential
can be wired into a check later.
"""
from __future__ import annotations

import argparse
import json
import sys

USABLE = ("OPTIMAL", "FEASIBLE")


def load(path: str) -> dict[int, str]:
    with open(path) as handle:
        return {int(seed): status for seed, status in json.load(handle).items()}


def diff(old: dict[int, str], new: dict[int, str]) -> tuple[list, list, list]:
    regressions, improvements, both_unusable = [], [], []
    for seed in sorted(old):
        before, after = old[seed], new[seed]
        if before in USABLE and after not in USABLE:
            regressions.append((seed, before, after))
        elif before not in USABLE and after in USABLE:
            improvements.append((seed, before, after))
        elif before not in USABLE and after not in USABLE and before != after:
            both_unusable.append((seed, before, after))
    return regressions, improvements, both_unusable


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("old", help="per-seed statuses of the earlier revision")
    parser.add_argument("new", help="per-seed statuses of the later revision")
    args = parser.parse_args()

    old, new = load(args.old), load(args.new)
    if set(old) != set(new):
        print(f"seed sets differ: {len(set(old) ^ set(new))} seeds not in both files")
        return 2
    regressions, improvements, both_unusable = diff(old, new)

    print(f"{args.old} -> {args.new} over {len(old)} seeds")
    print(f"regressions:  {len(regressions)}")
    for seed, before, after in regressions:
        print(f"   seed {seed}: {before} -> {after}")
    print(f"improvements: {len(improvements)}")
    for seed, before, after in improvements:
        print(f"   seed {seed}: {before} -> {after}")
    print(f"both-unusable, reason changed: {len(both_unusable)}")
    for seed, before, after in both_unusable:
        print(f"   seed {seed}: {before} -> {after}")
    return 1 if regressions else 0


if __name__ == "__main__":
    sys.exit(main())
