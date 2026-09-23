"""Guards for the master's sell model on the market's DEPTH curve (slice 2a).

The claim: when belief's curve is handed in (`belief.depth.sell_blocks`), the
master prices a day's sells on the ladder's own blocks instead of the flat
`0.5·p` tier — so the internal price of a good in the shed is the marginal the
ladder actually pays, and the Lagrangian bound still bounds.

  * `σ` falls below the horizon's peak quote (the shipped model credits the peak
    on every day: measured 792 carrot / 1729 tomato on the seeded day-0 board);
  * the bound stays a bound — with finite block sizes the sells' own inner
    problem must be carried in it, and leaving that term out puts the bound
    BELOW the objective (measured 29,161.7 against 32,083.5);
  * the town's appetite row is vacuous with a curve (the ladder replaces it) and
    stays live without one (the feature-absent path is the shipped model).

Run:  .venv/bin/python -m tests.test_master_depth   (also under pytest)
"""

from __future__ import annotations

import numpy as np

from agent.config import Config
from agent.belief.market import PRODUCTS, forecast
import agent.planner.master as M
from agent.planner import colgen
from agent.planner.colgen import classes_of


def _board(seed: int = 33, days: int | None = None):
    """The day-0 board: obs, supply, classes and the season's own horizon."""
    from offline_lab.fast_sim import FastSim
    sim = FastSim(configuration={"seed": seed, "episodeSteps": 720},
                  validate="dev")
    obs = sim.observations()[0]
    supply = M.supply_from_obs(obs)
    owned = M._owned_states(object(), obs)
    from agent.planner import columns as C
    reps, counts, _of_tile = classes_of(
        owned, M._owned_distances(obs, C.shed_distance()))
    horizon = M.season_horizon(obs) if days is None else days
    return obs, supply, reps, counts, horizon


def _solve(obs, supply, reps, counts, horizon, *, depth: bool,
           iter_cap: int | None = None):
    """One master solve. Five rounds is the least that computes a BOUND at all
    (round 1 prices at the σ seed and computes none), and it is what makes the
    bound guard able to fail."""
    from agent.tile_dp.contractor import TileContractor
    from agent.planner.inputs import GRAPH_PATH
    from agent.tile_dp.graph import TileGraph
    graph = TileGraph.load(GRAPH_PATH)
    contractor = TileContractor(graph, days=horizon)
    fc = forecast(obs, days=horizon) if depth else None
    res = M.equilibrate(object(), obs, contractor, supply,
                        iter_cap=iter_cap if iter_cap is not None
                        else 5,
                        forecast_obj=fc)
    return res, fc


def test_the_curve_lowers_the_internal_price_below_the_peak() -> None:
    """σ is the ladder's marginal, not the best price on the horizon."""
    obs, supply, reps, counts, horizon = _board()
    flat, _ = _solve(obs, supply, reps, counts, horizon, depth=False)
    deep, fc = _solve(obs, supply, reps, counts, horizon, depth=True)
    assert fc is not None
    peak = {g: max(fc.price_of(g, d) for d in range(horizon)) for g in PRODUCTS}
    sig_flat = np.asarray(flat.sigma, dtype=np.float64)
    sig_deep = np.asarray(deep.sigma, dtype=np.float64)
    # the shipped model credits the PEAK on every day, for the goods it sells
    for good in ("CARROT", "TOMATO"):
        i = PRODUCTS.index(good)
        assert np.isclose(sig_flat[:, i].max(), peak[good], rtol=1e-9), (
            f"{good}: the flat model should credit the peak {peak[good]}, "
            f"got {sig_flat[:, i].max():.2f}")
        assert sig_deep[:, i].max() < peak[good], (
            f"{good}: the curve must price below the peak {peak[good]}, "
            f"got {sig_deep[:, i].max():.2f}")
        assert sig_deep[:, i].max() > 0.0, f"{good}: the curve priced it at zero"


def _loaded_supply(obs, good: str = "MELON", units: int = 60):
    """The board's supply with a shed full of one good.

    A curve only DISCOUNTS the internal price when the plan sells in volume, and
    an empty shed with one tile does not: this is the board on which the sells'
    inner term in the bound is worth more than a rounding error.
    """
    import dataclasses
    supply = M.supply_from_obs(obs)
    stock = np.asarray(supply.shed_stock, dtype=np.int64).copy()
    stock[M.SHED_ITEMS.index(good)] = int(units)
    return dataclasses.replace(supply, shed_stock=stock)


def test_the_bound_is_still_a_bound_with_the_curve() -> None:
    """The sells' inner term is in the bound, or the bound is not a bound."""
    obs, supply, reps, counts, horizon = _board()
    supply = _loaded_supply(obs)
    for depth in (False, True):
        res, _ = _solve(obs, supply, reps, counts, horizon, depth=depth)
        assert res.bound >= res.objective - 1e-6, (
            f"depth={depth}: bound {res.bound:.1f} is BELOW the objective "
            f"{res.objective:.1f} — not a bound (the sells' inner term is "
            "missing from `lagrangian_bound`)")
        assert res.gap >= -1e-9, f"depth={depth}: negative gap {res.gap}"


def test_the_appetite_row_is_vacuous_with_a_curve() -> None:
    """The ladder replaces the appetite cap; the flat path keeps it."""
    obs, supply, reps, counts, horizon = _board()
    from agent.tile_dp.contractor import TileContractor
    from agent.planner.inputs import GRAPH_PATH
    from agent.tile_dp.graph import TileGraph
    graph = TileGraph.load(GRAPH_PATH)
    contractor = TileContractor(graph, days=horizon)
    fc = forecast(obs, days=horizon)
    goods = [M.SHED_ITEMS[ii] for ii in M.SELLABLE]
    depth = None
    from agent.belief.depth import sell_blocks
    depth = sell_blocks(fc, goods, int(obs.get("day", 0)), horizon,
                        int(supply.shed_capacity), blocks=M.SELL_BLOCKS)
    units, prices = depth
    assert units.shape[0] == len(goods) and units.shape[1] == horizon
    assert units.sum(axis=2).min() >= 1, "every good/day needs at least one block"
    # the first block's price IS the day's quote (the curve continues the path)
    for gi, good in enumerate(goods):
        assert np.isclose(prices[gi, 0, 0], fc.price_of(good, 0), rtol=1e-9), (
            f"{good}: the first block's price must be the day's quote")
    # The appetite row's own vacuity, on a good whose appetite is SMALLER than a
    # day's block capacity: MELON eats 31 units over the season while one day
    # can sell 100, so a live appetite row binds and its dual is positive. With
    # the curve the row carries no coefficient at all and that dual is zero.
    melon = PRODUCTS.index("MELON")
    stock = np.asarray(supply.shed_stock, dtype=np.int64).copy()
    stock[M.SHED_ITEMS.index("MELON")] = 60
    import dataclasses
    loaded = dataclasses.replace(supply, shed_stock=stock)
    res, _ = _solve(obs, loaded, reps, counts, horizon, depth=False)
    pool = list(res.pool)
    from agent.belief.market import price_paths
    path = np.asarray([price_paths(fc, days=horizon)[g] for g in PRODUCTS],
                      dtype=np.float64).T
    kwargs = dict(shed_stock=loaded.shed_stock,
                  shed_capacity=float(loaded.shed_capacity),
                  prices=path, market=M.SELLABLE,
                  sell_cap=M._sell_cap(obs, horizon))
    solved = colgen.MasterLP().solve(pool, counts, loaded.hours, loaded.money,
                                     horizon, M.N_COUPLING,
                                     depth=depth, **kwargs)
    assert np.allclose(np.asarray(solved.rho), 0.0), (
        "with a curve the appetite row is vacuous: rho must be 0, got "
        f"{np.asarray(solved.rho)}")
    solved_flat = colgen.MasterLP().solve(pool, counts, loaded.hours,
                                          loaded.money, horizon, M.N_COUPLING,
                                          depth=None, **kwargs)
    assert np.asarray(solved_flat.appetite).sum() > 0, (
        "without a curve the appetite row must stay live (the shipped model)")
    assert float(np.asarray(solved_flat.rho)[melon]) > 0.0, (
        "the flat model's appetite row must BIND on a 60-melon shed, or this "
        "guard proves nothing")


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
    print(f"{len(tests) - failures}/{len(tests)} master depth checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
