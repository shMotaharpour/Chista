"""The #110 own-supply finding, re-measured against the NEW master.

The master's LP grew a shed stock, balance rows, two-tier sells and an
appetite cap since #110 was filed. The question this module answers:
does the plan's own supply still mis-price the objective?

Two effects, separated:

1. THE APPETITE CAP (#126/#141's work, already on main): `_sell_cap`
   subtracts the rival's calendar supply from the town's appetite, so a
   rival's 20-melon dump shrinks the cap our sells compete for. Wired.

2. THE INTRA-DAY LADDER MOVE (still open): the LP prices every shallow
   sell at ONE flat quote (the path's price at day d), but the engine
   walks the ladder down unit by unit as the shared inventory fills.
   Measured on MELON at inv 9,990: 21 sells at the moving ladder fetch
   5,400 coins; the LP's flat-price model pays 5,691 — a 291-coin (5.4%)
   overstatement on exactly the plan sizes #110 is about.

The two-tier structure (`SELL_DEEP_FACTOR = 0.5`) models the cliff past
the appetite, not the gentle ladder move inside it. This guard pins the
5.4% number so the fix (a ladder-aware shallow price or an
inventory-marginal objective term) has a baseline to beat.

Run:  .venv/bin/python -m tests.test_own_supply_gap   (also pytest)
"""

from __future__ import annotations

from kaggle_environments.envs.kaggriculture import kaggriculture as K


def _moving_ladder_coins(good: str, inv: int, units: int) -> int:
    """The engine's own answer: sell `units` one at a time, each quoted at
    the shared inventory AFTER the previous sell added to it."""
    coins = 0
    for _ in range(units):
        coins += K.market_price(good, float(inv))
        inv += 1                     # our sell adds to the shared board
    return coins


def test_the_flat_price_overstates_the_moving_ladder() -> None:
    """21 melons at the flat path price vs the engine's moving ladder:
    the gap is the #110 own-supply overstatement, measured."""
    inv = 9990
    units = 21
    flat = units * K.market_price("MELON", float(inv))
    true_coins = _moving_ladder_coins("MELON", inv, units)
    over = flat - true_coins
    assert over > 0, (
        "the moving ladder paid MORE than the flat price: the effect "
        "#110 describes is not present on this board")
    assert over / true_coins > 0.03, (
        f"the overstatement fell to {over / true_coins * 100:.1f}% — "
        "re-measure and re-file the number")


def main() -> int:
    inv = 9990
    units = 21
    flat = units * K.market_price("MELON", float(inv))
    true_coins = _moving_ladder_coins("MELON", inv, units)
    print(f"flat price model : {flat} coins")
    print(f"moving ladder    : {true_coins} coins")
    print(f"overstatement    : {flat - true_coins} coins "
          f"({(flat - true_coins) / true_coins * 100:.1f}%)")
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
    print(f"{len(tests) - failures}/{len(tests)} own-supply checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
