"""Tests for market mechanics — quoted/commit behavior measured in-engine.

Covers the user's game-rules.md §3 findings:
- SELL commits unit-by-unit: price walks DOWN as your own supply hits the market
- price floor $1 sales do NOT add supply (L669: `if price > 1`)
- both players are quoted from the same pre-commit inventory per queue index
- BUY_PRODUCT round-trip nets zero against an unchanged market
- order cap 10 per turn

Note: wheat starts at I0-ish inventory; buying 90 costs ~2820 (leaves ~180),
selling 90 walks the price 26→... The tests use observable invariants, not
exact amounts, wherever the starting price varies with seed.
"""
from __future__ import annotations

from kaggle_environments import make

P = {"farmer": ["PASS"], "hands": [], "market": []}


def make_env(seed: int = 70):
    env = make("kaggriculture", configuration={"episodeSteps": 720, "seed": seed},
               debug=False)
    env.reset(2)
    return env


def buy_wheat(env, n: int) -> int:
    """Buy n wheat via BUY_PRODUCT; returns how many actually landed in shed
    (money-bound)."""
    env.step([{"farmer": ["PASS"], "hands": [], "market":
               [["BUY_PRODUCT", "WHEAT", n]]}, P])
    return env.state[0].observation.private["shed"].get("WHEAT", 0)


def test_sell_walks_price_down_unit_by_unit():
    """MEASURED: SELL 90 wheat from I0-ish market pushes inventory well above
    I0 → price must DROP below the starting 26... wait, starting price is 26
    ABOVE I0? Wheat starts AT I0=10000, price = base 25 (26 after rounding at
    slightly-above). Selling 90 → inventory 10090 → price drops to ~24."""
    env = make_env()
    n = buy_wheat(env, 90)
    assert n == 90
    p_before = env.state[0].observation.market["prices"]["WHEAT"]
    inv_before = env.state[0].observation.market["inventory"]["WHEAT"]
    env.step([{"farmer": ["PASS"], "hands": [], "market": [["SELL", "WHEAT", 90]]}, P])
    o = env.state[0].observation
    assert o.market["inventory"]["WHEAT"] == inv_before + 90
    assert o.market["prices"]["WHEAT"] < p_before, (p_before, o.market["prices"]["WHEAT"])


def test_floor_sales_add_no_supply():
    """MEASURED (engine L669-671): a sale at price $1 does not increase market
    inventory. WHEAT never floors in practice (log shape asymptotes ~13);
    MELON (sq shape) floors just past I0 — verified via the engine's own
    _commit_unit, the exact code path a real SELL takes."""
    from kaggle_environments.envs.kaggriculture import kaggriculture as K
    assert K.market_price("MELON", K.MARKET_I0 + 1000) == 1   # glut → floor
    farm = {"money": 0}
    private = {"shed": {"MELON": 5}}
    market = {"inventory": {"MELON": K.MARKET_I0 + 1000}}
    ok = K._commit_unit("SELL", "MELON", 1, farm, private, market)
    assert ok is True
    assert farm["money"] == 1
    assert market["inventory"]["MELON"] == K.MARKET_I0 + 1000   # unchanged!
    # sanity: a sale ABOVE the floor DOES add supply
    market2 = {"inventory": {"MELON": K.MARKET_I0}}
    farm2 = {"money": 0}
    private2 = {"shed": {"MELON": 1}}
    K._commit_unit("SELL", "MELON", 250, farm2, private2, market2)
    assert market2["inventory"]["MELON"] == K.MARKET_I0 + 1


def test_buy_product_roundtrip_nets_zero():
    """MEASURED (engine comment L599-600): BUY_PRODUCT is quoted at post-buy
    inventory, so buy-then-sell against an otherwise unchanged market nets zero."""
    env = make_env()
    m0 = env.state[0].observation.farms[0]["money"]
    env.step([{"farmer": ["PASS"], "hands": [], "market":
               [["BUY_PRODUCT", "WHEAT", 10], ["SELL", "WHEAT", 10]]}, P])
    m1 = env.state[0].observation.farms[0]["money"]
    assert m1 == m0, (m0, m1)


def test_two_players_same_index_same_precommit_quote():
    """MEASURED: both players' SELLs at the same queue index are quoted from
    the same pre-commit inventory — two sellers of X push the price lower
    than one seller of X."""
    def build():
        e = make_env()
        # both players buy 90 wheat (they hold their own sheds)
        e.step([{"farmer": ["PASS"], "hands": [], "market":
                  [["BUY_PRODUCT", "WHEAT", 90]]},
                {"farmer": ["PASS"], "hands": [], "market":
                  [["BUY_PRODUCT", "WHEAT", 90]]}])
        return e

    # both sell 100? they hold 90 each: both sell 90 in the same turn
    e_both = build()
    inv0 = e_both.state[0].observation.market["inventory"]["WHEAT"]
    e_both.step([{"farmer": ["PASS"], "hands": [], "market": [["SELL", "WHEAT", 90]]},
                 {"farmer": ["PASS"], "hands": [], "market": [["SELL", "WHEAT", 90]]}])
    p_both = e_both.state[0].observation.market["prices"]["WHEAT"]

    # baseline: ONE player sells 90 from the same starting inventory
    e_one = build()
    e_one.step([{"farmer": ["PASS"], "hands": [], "market": [["SELL", "WHEAT", 90]]}, P])
    p_one = e_one.state[0].observation.market["prices"]["WHEAT"]

    assert p_both < p_one, (p_both, p_one)


def test_order_cap_ten_per_turn():
    """MEASURED: maxMarketOrdersPerTurn = 10 — the 11th order in a turn is
    dropped silently (12 sells of 1 → only 10 commit, 2 wheat remain)."""
    env = make_env()
    n = buy_wheat(env, 12)
    assert n == 12
    orders = [["SELL", "WHEAT", 1]] * 12
    env.step([{"farmer": ["PASS"], "hands": [], "market": orders}, P])
    shed_after = env.state[0].observation.private["shed"].get("WHEAT", 0)
    assert shed_after == 2   # only 10 of the 12 sells committed


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {fn.__name__}: {e}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"ERROR {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    raise SystemExit(1 if failed else 0)
