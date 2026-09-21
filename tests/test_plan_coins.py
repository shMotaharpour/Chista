"""Guards for the ladder's multi-day tools: `split_days` (the exact best
split) and `plan_coins` (the manager's what-if on ANY plan) — #96.

The defect this pins: the pre-fix DP added the already-sold units back
into the state's inventory and reused the day-0 cumsum window for every
state, so it deferred everything to the last day — 2,646 where 2,701 was
available (MILK, 9,950, 12 lots, 5 days, drain 1/day). No test priced
the DP against enumeration, so it shipped green.

Run:  .venv/bin/python -m tests.test_plan_coins   (also under pytest)
"""

from __future__ import annotations

import numpy as np

from agent.belief.ladder import plan_coins, sell_coins, split_days
from itertools import product


def _brute(good: str, inv: int, lot: int, drains: np.ndarray) -> int:
    return max(plan_coins(good, inv, combo, drains)
               for combo in product(range(lot + 1), repeat=len(drains))
               if sum(combo) == lot)


def test_split_days_equals_brute_force_enumeration() -> None:
    """The DP's optimum equals enumerating EVERY split, to the coin.

    The guard the old test lacked: test_market_ladder compared the DP
    against greedy, and greedy is wrong in the DP's favour — the defect
    survived because nothing enumerated.
    """
    for good, inv, lot, days in (("MILK", 9950, 12, 5),
                                 ("WHEAT", 9900, 10, 4),
                                 ("WOOL", 9950, 7, 6)):
        drains = np.ones(days, dtype=np.int64)
        xs, coins = split_days(good, inv, lot, drains)
        assert coins == _brute(good, inv, lot, drains), (
            f"{good}: DP {coins} != enumeration {_brute(good, inv, lot, drains)}")
        assert sum(xs.tolist()) == lot
        assert plan_coins(good, inv, xs, drains) == coins, (
            "plan_coins disagrees with split_days on split_days' own plan")


def test_plan_coins_distinguishes_3_per_day_from_1_per_day() -> None:
    """The property the manager prices on: the SHAPE of a sell plan
    changes its total — 3/day for 4 days is NOT 1/day for 12, on a
    board the town drains."""
    inv, lot = 9950, 12
    a = plan_coins("MILK", inv, [3, 3, 3, 3, 0], np.ones(5, dtype=np.int64))
    b = plan_coins("MILK", inv, [1] * 12, np.ones(12, dtype=np.int64))
    assert a != b, (a, b)
    # and both are real coin totals, priced by the same ladder:
    assert a > 0 and b > 0


def test_plan_coins_never_oversells_the_shed() -> None:
    """A plan that sells more than the inventory holds clamps per day and
    prices the clamped plan — it never goes negative or raises."""
    coins = plan_coins("MILK", 9950, [5000, 5000], np.ones(2, dtype=np.int64))
    assert coins > 0


def test_plan_coins_matches_per_day_sell_coins() -> None:
    """One day of the plan prices exactly as `sell_coins` for that day."""
    drains = np.array([2, 3], dtype=np.int64)
    coins = plan_coins("WHEAT", 9900, [4, 6], drains)
    hand = (sell_coins("WHEAT", 9900, 4)
            + sell_coins("WHEAT", 9900 - 4 - 2, 6))
    assert coins == hand


def main() -> int:
    tests = [(k, v) for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    failures = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except Exception as exc:                       # noqa: BLE001
            failures += 1
            print(f"FAIL {name}: {type(exc).__name__}: {exc}")
    print(f"{len(tests) - failures}/{len(tests)} plan-coins checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
