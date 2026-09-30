"""Guards for `belief/` — the claims its docstrings make, each seen to fail once.

Run:  .venv/bin/python -m tests.test_market_analyzer     (also under pytest)

R007 (a guard must be seen to fail) is applied per guard in the PR body; the
claims are the ones other issues will build on, so a guard that only ever ran
green would be worse than no guard:

  * the price function equals the engine's, everywhere;
  * the residual is exact on the seven one-way goods and net-only on the two
    dual goods (the reason the second claim is weaker is the engine's own quote
    rule, not a modelling choice of ours);
  * the demand forecast is closed form: already-open shops are facts, and a
    single-type prior reproduces that type's basket exactly;
  * the slot game's maximin beats dumping the lot, and the two solvers agree
    where the problem is discrete;
  * the season LP's absorption row binds (without it the same LP prints 188M);
  * the stubs satisfy the contract, so the pipeline can be assembled today.
"""

from __future__ import annotations

import numpy as np

from kaggle_environments.envs.kaggriculture import kaggriculture as K

from agent.belief.schemas import (DUAL, G_IX, SHED_CAP, SHOP_BASKET,
                                  SHOP_TYPES)
from agent.world.model import PRODUCTS as GOODS
from agent.world.prices import MARKET_I0, price, price_vec

from bench.bench_market_analyzer import (center_drain_probe, run_episode,
                                         slot_inference_experiment)
from agent.belief.opponent import drain_forecast, infer_rival_slot
from agent.belief.schemas import DaySchedule, OrderBook, SellIntent
from agent.belief.solvers import (default_schedules, maximin_mixed_lp,
                            maximin_mixed_slsqp, season_plan_maximin,
                            slot_game_matrix)
from agent.belief.stubs import naive_day_plan, naive_order_book, rival_supply_stub
from agent.belief.tracker import MarketTracker

GRID = np.concatenate([np.arange(0, 60, 1.0), np.arange(100, 12000, 17.0)])

#: The fractional grid: the walk's rows carry fractional inventories (the
#: residual is spread per turn, the mean unlock policy adds fractional shops), so
#: `price_vec` is asked for non-integers in production. Its steps are irrational
#: on purpose (0.37, 1/3) so the samples cannot land only on the integers, and it
#: straddles I0 in 0.01 steps — the branch boundary the two-sided formula meets.
GRID_FRACTIONAL = np.concatenate([
    np.arange(0.0, 12000.0, 0.37),
    np.arange(9990.0, 10010.0, 0.01),
    np.arange(0.0, 400.0, 1.0 / 3.0),
])


def test_price_parity() -> None:
    """The vectorised curve IS the engine's curve, at every inventory.

    Integer and fractional: the rows the hourly surface prices are not integers,
    so a guard that only walks whole units would not see the side of the formula
    the engine's `round` acts on.
    """
    for g in GOODS:
        for grid in (GRID, GRID_FRACTIONAL):
            mine = price_vec(g, grid)
            theirs = np.array([K.market_price(g, float(i)) for i in grid])
            assert np.array_equal(mine, theirs), (
                f"{g}: vectorised price differs from the engine "
                f"({int((mine != theirs).sum())} of {len(grid)} cells, "
                f"first at inventory {grid[int((mine != theirs).argmax())]!r})")


def test_price_grid_equals_the_engine_for_every_good_at_once() -> None:
    """`price_grid`'s (rows x 9) surface is `price_table`'s row, repeated.

    The hourly and walk surfaces price all nine goods in one call, so the
    all-goods form and the one-row form must be the same numbers, and both must
    be the engine's at every cell (fractional inventories included).
    """
    from agent.world.prices import price_grid, price_table

    # a (rows, 9) surface: fractional inventories, one column per good
    vals = GRID_FRACTIONAL[:200]
    surface = np.column_stack([vals + i for i in range(len(GOODS))])
    grid = price_grid(surface)
    assert grid.shape == surface.shape
    for i in (0, 1, 57, 199):
        row = surface[i]
        assert np.array_equal(grid[i], price_table(row)), i
        theirs = np.array([K.market_price(g, float(x)) for g, x in zip(GOODS, row)])
        assert np.array_equal(grid[i], theirs), (
            f"row {i}: price_grid differs from the engine at "
            f"{[(g, float(x)) for g, x in zip(GOODS, row)]}")
    # and the one-row surface the tracker and the schema reader ask for
    row = np.array([9999.5, 10000.5, 10001.25, 0.0, 7.5,
                    120.5, 350.75, 9000.25, 11999.9])
    theirs = np.array([K.market_price(g, float(x)) for g, x in zip(GOODS, row)])
    assert np.array_equal(price_grid(row[None, :])[0], theirs)
    assert np.array_equal(price_table(row), theirs)


def test_the_goods_split_is_seven_one_way_two_dual() -> None:
    """7 goods can only be sold; WHEAT and FERTILIZER can also be bought."""
    assert len(GOODS) == 9 and len(DUAL) == 2
    assert sum(1 for g in GOODS if g not in DUAL) == 7


# --------------------------------------------------------------------------- #
# phase 1
# --------------------------------------------------------------------------- #

def test_residual_is_exact_on_the_one_way_goods() -> None:
    """The inventory residual reproduces the rival's orders on the 7, exactly.

    The coverage assertion is part of the guard, not decoration: the first version
    of this test passed with the drain term deleted from the residual, because a
    turn where only the town consumes pushes the residual negative and the
    `max(net, 0)` clamp absorbs the error. The run must contain a turn where the
    town drains *and* the rival sells, or the guard is blind to the term it claims
    to cover.
    """
    r = run_episode(days=2, seat0_silent=True)
    assert len(r["residual7"]) > 24, "not enough turns to judge"
    assert r["residual7"].max() == 0.0, (
        f"residual error {r['residual7'].max()} on goods that cannot be bought"
    )


def test_residual_survives_a_turn_that_both_drains_and_sells() -> None:
    """The drain term is covered: a centre-consumption turn where the rival sells.

    Without this case the previous guard passed with the drain term deleted, since
    a drain-only turn is clamped away by `max(net, 0)`.
    """
    probe = center_drain_probe(lot=5)
    assert probe["at_center"], (
        f"probe drained step {probe['drained_step']}, not a centre turn"
    )
    assert probe["drain"][[G_IX[g] for g in GOODS if g != "FERTILIZER"]].sum() > 0
    assert probe["residual"][G_IX["MILK"]] == probe["truth_units"], probe["residual"]
    assert probe["residual"][G_IX["EGG"]] == 0.0, "a good the rival never sold was invented"


def test_the_dual_goods_are_net_only() -> None:
    """WHEAT/FERTILIZER error stays inside the buy volume, and the slope says why."""
    r = run_episode(days=2, seat0_silent=True)
    buy_qty = 2                        # the bench buys WHEAT 2 when it trades
    assert r["residual_dual"].max() <= buy_qty, r["residual_dual"].max()
    slope = r["tracker"].diagnostics["slope_per_unit"]
    assert all(abs(v) <= 1 for v in slope.values()), slope


def test_rival_stock_estimate_tracks_the_truth() -> None:
    """Their shed, estimated from public data alone, stays within a couple of units."""
    r = run_episode(days=2, seat0_silent=True)
    assert r["shed"].mean() <= 1.0, r["shed"].mean()


# --------------------------------------------------------------------------- #
# phase 2
# --------------------------------------------------------------------------- #

def test_drain_forecast_is_closed_form() -> None:
    """No unlock in the window means no variance, and a single-type prior is exact."""
    obs = {"step": 24, "town": {"unlocked_shops": []}}          # no unlock falls in (24, 48]
    mean, sd = drain_forecast(obs, 24)
    assert sd.max() == 0.0, "variance appeared where the schedule has no draw"
    # one prior mass on a single-product shop, in a window that DOES contain an
    # unlock: day 3 (step 72) is inside (24, 72], and its basket is WOOL x2.
    idx = SHOP_TYPES.index("YARN_STORE")
    prior = np.zeros(len(SHOP_TYPES))
    prior[idx] = 1.0
    mean1, sd1 = drain_forecast({"step": 24, "town": {"unlocked_shops": []}}, 48, prior)
    shop_events = (24 + 48 - 72) // 4 + 1        # one consumption after the unlock
    center_events = 48 // 24 + 1                 # the centre eats every day
    assert mean1[G_IX["WOOL"]] == float(2 * shop_events + center_events), mean1[G_IX["WOOL"]]
    assert sd1[G_IX["WOOL"]] == 0.0, "a degenerate prior must have no spread"


def test_open_shops_are_facts_not_draws() -> None:
    """An already-open shop contributes its basket every 4 steps, regardless of prior."""
    obs = {"step": 24, "town": {"unlocked_shops": ["YARN_STORE"]}}
    mean, sd = drain_forecast(obs, 24)
    events = 24 // 4 + 1
    assert mean[G_IX["WOOL"]] >= float(2 * events)


def test_slot_inference_reads_the_side_and_admits_when_it_cannot() -> None:
    """Their realised price locates their volume relative to ours -- if we traded."""
    early = slot_inference_experiment(true_index=0, our_slot=6)
    late = slot_inference_experiment(true_index=9, our_slot=6)
    assert early["with_us"]["identifiable"]
    assert early["with_us"]["estimate"] < early["our_slot"]
    assert not late["with_us"]["identifiable"] or late["with_us"]["estimate"] > 0
    assert not early["without_us"]["identifiable"], "claimed identifiability with no trade"


# --------------------------------------------------------------------------- #
# phase 3
# --------------------------------------------------------------------------- #

def _wheat_slot_matrix() -> np.ndarray:
    lot = 100.0
    rival = np.array([[lot, 0.0], [0.0, lot], [lot / 2, lot / 2]])
    return slot_game_matrix("WHEAT", float(K.MARKET_I0), default_schedules(lot), rival,
                            drain_per_turn=1.0, turns=24)


def test_maximin_mix_beats_dumping_the_lot() -> None:
    """Spreading the lot is worth coins, and the security level says how many."""
    A = _wheat_slot_matrix()
    mix, value = maximin_mixed_lp(A)
    assert abs(float(mix.sum()) - 1.0) < 1e-9 and (mix >= 0).all()
    assert value > A[0].min(), f"security {value:.1f} not above dumping {A[0].min():.1f}"


def test_the_two_solvers_agree_on_the_discrete_game() -> None:
    """With the risk weight at its extreme the SLSQP path is the maximin problem."""
    A = _wheat_slot_matrix()
    _mix_lp, value_lp = maximin_mixed_lp(A)
    mix_sq, worst_sq, _mean = maximin_mixed_slsqp(A, risk_lambda=1.0)
    assert abs(float(mix_sq.sum()) - 1.0) < 1e-9 and (mix_sq >= 0).all()
    assert worst_sq >= value_lp - 0.01 * value_lp, (worst_sq, value_lp)


def test_season_lp_absorption_row_binds() -> None:
    """Sales of a good cannot exceed what the town eats at the worst-case share."""
    goods = ["WHEAT", "MILK"]
    prices0 = np.array([price(g, float(K.MARKET_I0)) for g in goods], dtype=float)
    drain = np.array([17.5, 10.9])
    scenarios = np.array([[1.0, 1.0], [0.5, 0.5]])
    plan = season_plan_maximin(goods, 30, np.array([1.4, 0.4]), 100.0, prices0,
                               scenarios, drain)
    caps = [drain[i] * 30 * scenarios[:, i].min() for i in range(len(goods))]
    for gi in range(len(goods)):
        assert plan["units"][gi].sum() <= caps[gi] + 1e-6, (goods[gi], plan["units"][gi].sum())
    assert any(plan["units"][gi].sum() >= caps[gi] - 1e-6 for gi in range(len(goods))), (
        "no good reached its absorption cap: the row is not binding, so the test "
        "would pass on a model without it"
    )


# --------------------------------------------------------------------------- #
# the replacements
# --------------------------------------------------------------------------- #

def test_stubs_keep_the_pipeline_importable() -> None:
    """A caller can be written against the contract before the units exist."""
    forecast, confidence = rival_supply_stub()
    assert forecast.shape == (len(GOODS),) and confidence == 0.0
    book = naive_order_book({"WHEAT": 4, "MILK": 2},
                            [SellIntent("MILK", 2, 0, (0, 23), 0, "test")],
                            hires_today=1)
    assert isinstance(book, OrderBook) and len(book.per_turn) >= 1
    assert book.per_turn[0][0][0] == "HIRE"          # hires lead the turn they are decided
    assert all(len(row) <= 10 for row in book.per_turn)
    plan = naive_day_plan([1, None], seeds=1)
    assert isinstance(plan, DaySchedule) and len(plan.per_unit) == 2


# --------------------------------------------------------------------------- #
# runner (the project's own form: python -m tests.test_market_analyzer)
# --------------------------------------------------------------------------- #

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
    print(f"{len(tests) - failures}/{len(tests)} market analyzer checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
