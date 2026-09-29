"""Guards for `agent/belief/ladder.py` — the array identity claims, each seen
to fail once (R007 applies to the DP: the naive greedy IS the red control).

Run:  .venv/bin/python -m tests.test_market_ladder   (also under pytest)
"""

from __future__ import annotations

import numpy as np

from kaggle_environments.envs.kaggriculture import kaggriculture as K
from agent.world.model import PRODUCTS
from agent.belief.ladder import (FLOOR, G_HI, G_LO, P, S, buy_coins,
                                 good_index, marginal_coins_vec, sell_coins,
                                 sell_coins_vec, split_days,
                                 supply_after_sell)


def _walk_sell(good: str, inventory: int, units: int) -> tuple[int, int]:
    """The engine's own walk: quote, then +1 supply iff price > floor."""
    coins, supply, inv = 0, 0, inventory
    for _ in range(units):
        price = K.market_price(good, float(inv))
        coins += price
        if price > 1:
            inv += 1
            supply += 1
    return coins, supply


def test_the_quote_table_is_the_engine_bit_for_bit() -> None:
    """P is `market_price` over the whole grid, every good, exactly."""
    grid = np.arange(G_LO, G_HI, dtype=np.float64)
    for i, g in enumerate(PRODUCTS):
        theirs = np.array([K.market_price(g, float(v)) for v in grid])
        assert np.array_equal(P[i], theirs), f"{g}: P differs from the engine"


def test_the_sell_ladder_matches_the_unit_walk() -> None:
    """Coins AND landed supply equal the engine walk, floor stall included."""
    rng = np.random.default_rng(11)
    for _ in range(1200):
        g = PRODUCTS[int(rng.integers(len(PRODUCTS)))]
        i = good_index(g)
        inv = int(rng.integers(8000, 11000))
        units = int(rng.integers(0, 400))
        coins, supply = _walk_sell(g, inv, units)
        assert sell_coins(g, inv, units) == coins, (g, inv, units)
        assert supply_after_sell(g, inv, units) == supply, (g, inv, units)


def test_the_buy_ladder_matches_the_unit_walk() -> None:
    """A buy is quoted at price(I - 1); the cumsum window says the same."""
    rng = np.random.default_rng(5)
    for _ in range(600):
        g = PRODUCTS[int(rng.integers(len(PRODUCTS)))]
        inv = int(rng.integers(9200, 14000))
        units = int(rng.integers(0, 300))
        coins = 0
        for k in range(units):
            coins += K.market_price(g, float(inv - 1 - k))
        assert buy_coins(g, inv, units) == coins, (g, inv, units)


def test_the_floor_stall_is_real() -> None:
    """WOOL's floor exists, and selling past it lands no extra supply."""
    g = "WOOL"
    floor = FLOOR[good_index(g)]
    assert floor is not None
    assert K.market_price(g, float(floor)) == 1
    assert supply_after_sell(g, floor - 5, 50) == 5
    assert sell_coins(g, floor - 5, 50) == _walk_sell(g, floor - 5, 50)[0]


def _brute(good: str, inventory: int, lot: int, drains: list[int]) -> int:
    """Exhaustive enumeration of the day split — the DP's reference.

    Engine order per day (F037): the sale quotes the inventory BEFORE
    that day's drain, so both the units sold and the past drains
    subtract from the state the day prices against (`inventory - sold -
    W[d]`). The pre-#96 version ADDED `sold` here, which agreed with the
    DP's own sign bug — two wrongs agreeing, the reason the split defect
    survived this enumeration.
    """
    g = good
    W = [0]
    for d in drains[:-1]:
        W.append(W[-1] + d)
    best = [-1]
    def rec(d: int, sold: int, coins: int) -> None:
        if d == len(drains):
            best[0] = max(best[0], coins)
            return
        st = inventory - sold - W[d]
        for k in range(lot - sold + 1):
            rec(d + 1, sold + k,
                coins + sell_coins(g, st, k))
    rec(0, 0, 0)
    return best[0]


def test_the_vectorized_ladder_agrees_with_the_scalar_cell_by_cell() -> None:
    """`sell_coins_vec`/`marginal_coins_vec` are the scalar's numbers, arrayed.

    The depth surface calls them for whole (good x day x block) cells at once,
    so the two spellings must agree everywhere it can be asked — including at
    and past the floor point, where the `$1`-a-unit branch and its head term are
    the parts a dropped branch gets wrong.
    """
    rng = np.random.default_rng(29)
    for _ in range(40):
        g = PRODUCTS[int(rng.integers(len(PRODUCTS)))]
        floor = FLOOR[good_index(g)]
        # half the draws sit near the good's own floor when it has one
        base = int(rng.integers(G_LO + 1, 12000)) if floor is None \
            else int(floor + rng.integers(-40, 60))
        invs = np.array([base + k for k in range(12)], dtype=np.int64)
        units = np.array([0, 1, 2, 7, 25, 120], dtype=np.int64)
        grid_inv = np.repeat(invs, len(units))
        grid_units = np.tile(units, len(invs))
        theirs = np.array([sell_coins(g, int(i), int(u))
                           for i, u in zip(grid_inv, grid_units)])
        assert np.array_equal(sell_coins_vec(g, grid_inv, grid_units), theirs), \
            f"{g}: sell_coins_vec differs from sell_coins"
        # the marginal: sell_coins(inv, m + 1) - sell_coins(inv, m)
        sold = np.repeat(np.array([0, 1, 5], dtype=np.int64), len(invs))
        rep_inv = np.tile(invs, 3)
        theirs_m = np.array([
            sell_coins(g, int(i), int(m) + 1) - sell_coins(g, int(i), int(m))
            for i, m in zip(rep_inv, sold)])
        assert np.array_equal(marginal_coins_vec(g, rep_inv, sold), theirs_m), \
            f"{g}: marginal_coins_vec differs from the ladder's own difference"


def test_the_day_split_is_the_exact_optimum() -> None:
    """DP == enumeration on randomized instances; the greedy is the control."""
    rng = np.random.default_rng(3)
    greedy_loses = 0
    for _ in range(200):
        g = PRODUCTS[int(rng.integers(len(PRODUCTS)))]
        inv = int(rng.integers(9500, 9900))
        lot = int(rng.integers(0, 15))
        drains = rng.integers(0, 3, int(rng.integers(2, 6))).tolist()
        xs, coins = split_days(g, inv, lot, drains)
        assert xs.sum() == lot, (g, inv, lot, drains, xs)
        assert coins == _brute(g, inv, lot, drains), (g, inv, lot, drains, xs)
        # the naive control: each unit takes the day with the best marginal price
        W = np.concatenate([[0], np.cumsum(drains)[:-1]])
        xg = np.zeros(len(drains), dtype=int)
        for _ in range(lot):
            marg = [K.market_price(g, float(inv - xg.sum() - W[d]))
                    for d in range(len(drains))]
            xg[int(np.argmax(marg))] += 1
        gc = sum(sell_coins(g, inv - int(xg[:d].sum()) - W[d], int(xg[d]))
                 for d in range(len(drains)))
        greedy_loses += gc < coins
    assert greedy_loses > 0, (
        "the greedy tied the DP everywhere: the coupling the DP owns never "
        "showed up, so this guard cannot fail on a broken split")


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
    print(f"{len(tests) - failures}/{len(tests)} market ladder checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
