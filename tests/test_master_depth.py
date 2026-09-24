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



# --- the reduced process, for the guard that needs a CERTIFICATE ----------
# The one-tile board is a PROBE-level patch (price and run only the tile the
# farmer stands on, no hands), never a `Config` switch: the guards that need a
# certified LP use it, everything else runs the real board.
_REAL = {}


def _one_tile(runtime, obs):
    from agent.planner import columns as C
    states = _REAL["states"](runtime, obs)
    distances = _REAL["distances"](obs, C.shed_distance())
    here = [i for i, d in enumerate(distances) if int(d) == 0]
    return [states[here[0]]] if here else states[:1]


def _only_here(obs, steps):
    return [0]


def _board_class_map(self, obs, of_tile):
    from agent.obs import decode_world
    view = decode_world(obs, at_day_start=True, graph_keys=self.keys)
    keys = np.asarray(view.me.keys)
    farm = (obs.get("farms") or [{}])[int(obs.get("player", 0))]
    fx, fy = farm.get("farmer", (0, 0))
    idx = int(fy) * keys.shape[1] + int(fx)
    out = [None] * keys.size
    if keys.reshape(-1)[idx] >= 0:
        out[idx] = 0
    return out


def _solve_one_tile(obs, supply, horizon, *, depth: bool):
    """`equilibrate` on the reduced process, at the default round cap."""
    import agent.manager.core as MC
    from agent.tile_dp.contractor import TileContractor
    from agent.planner.inputs import GRAPH_PATH
    from agent.tile_dp.graph import TileGraph
    _REAL.setdefault("states", M._owned_states)
    _REAL.setdefault("distances", M._owned_distances)
    _REAL.setdefault("class_of_tile", MC.Manager._class_of_tile)
    contractor = TileContractor(TileGraph.load(GRAPH_PATH), days=horizon)
    fc = forecast(obs, days=horizon) if depth else None
    M._owned_states, M._owned_distances = _one_tile, _only_here
    MC.Manager._class_of_tile = _board_class_map
    try:
        res = M.equilibrate(object(), obs, contractor, supply,
                            iter_cap=Config().master_rounds, forecast_obj=fc)
    finally:
        M._owned_states = _REAL["states"]
        M._owned_distances = _REAL["distances"]
        MC.Manager._class_of_tile = _REAL["class_of_tile"]
    return res, fc

def test_the_curve_lowers_the_internal_price_below_the_peak() -> None:
    """σ is the ladder's marginal, not the best price on the horizon."""
    obs, supply, reps, counts, horizon = _board()
    flat, _ = _solve(obs, supply, reps, counts, horizon, depth=False)
    deep, fc = _solve(obs, supply, reps, counts, horizon, depth=True)
    assert fc is not None
    # The ceiling is the market's own peak, and the day-start rows are no longer
    # it: the model prices a day at the best hour that day can reach, so the
    # reference has to be the hourly surface's maximum, not `price_of`.
    from agent.belief.market import hourly_prices
    H = hourly_prices(fc, days=horizon)
    # Two references, because the two arms price on different surfaces: the
    # shipped flat model on the day-start rows, the depth model on each day's
    # best hour (so its ceiling is the hourly maximum).
    peak_rows = {g: max(fc.price_of(g, d) for d in range(horizon))
                 for g in PRODUCTS}
    peak = {g: int(H[:, PRODUCTS.index(g)].max()) for g in PRODUCTS}
    sig_flat = np.asarray(flat.sigma, dtype=np.float64)
    sig_deep = np.asarray(deep.sigma, dtype=np.float64)
    # the shipped model credits the PEAK on every day, for the goods it sells
    for good in ("CARROT", "TOMATO"):
        i = PRODUCTS.index(good)
        assert np.isclose(sig_flat[:, i].max(), peak_rows[good], rtol=1e-9), (
            f"{good}: the flat model should credit the day-start peak "
            f"{peak_rows[good]}, got {sig_flat[:, i].max():.2f}")
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
    """The sells' inner term is in the bound, or the bound is not a bound.

    The inequality is checked directly, at whatever cap the reduced process runs. The
    original form certified first and then compared, on the theory that the reduced board
    lands on the objective and a dropped term shows up as a bound BELOW it at once
    (measured 29,161.7 against 32,083.5). It no longer certifies: measured, the gap walks
    down 0.85 at five rounds, 0.77 at eight, 0.66 at twelve, 0.45 at twenty - tens of rounds
    to close - so the precondition had stopped being reachable and the guard proved nothing.

    Dropping the precondition costs nothing here, because the defect it catches is a bound
    BELOW the objective, and that is visible at any cap: the bound stays above the objective
    at every cap measured (914,247 against 133,480 at five rounds; 366,181 against 199,911
    at twenty). Certification was never what gave this guard teeth.
    """
    obs, supply, _reps, _counts, horizon = _board()
    for depth in (True, False):
        res, _fc = _solve_one_tile(obs, supply, horizon, depth=depth)
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


def test_the_blocks_term_is_in_the_bound() -> None:
    """The sells' inner problem is a term of the bound, and it is exactly this term.

    `lagrangian_bound`'s own comment: with finite block sizes the term is
    `sum_b u_b * max(0, p_b - sigma)` - block b is worth selling whenever the ladder pays
    more than the unit in the shed is carried at - and "a term left out does not make this
    conservative, it makes it a different number".

    This checks the number. The bound is computed twice on the SAME multipliers, once with
    the blocks and once without; everything else is identical, so the difference is the
    term and nothing else. The numbers are chosen to make the expectation a literal: one
    block, ten units, a price of 7 against a carry of 1, so the term is 10 * (7 - 1) = 60.
    """
    from agent.planner.colgen import MasterSolve, lagrangian_bound

    days, n_coupling = 2, 1
    solve = MasterSolve(lam=np.zeros(0), y=np.zeros((days, n_coupling)),
                        cash=np.zeros(days), mu=np.zeros(0), objective=0.0)
    values, counts = np.zeros(0), np.zeros(0)
    hours, money = np.zeros(days), 1.0
    sigma = np.array([[1.0], [1.0]])                     # the carry of the priced item
    shed = (np.zeros(1), 0.0, sigma, np.zeros(days))     # opening, capacity, sigma, tau
    market = (0,)
    units = np.full((days, days, 1), 10.0)               # u_b = 10 on every block-day
    prices = np.zeros((days, days, 1))
    prices[0, 0, 0] = 7.0                                # p_b = 7 on day 0, hour 0

    def bound(blocks):
        return lagrangian_bound(solve, values, counts, hours, money, days, n_coupling,
                                shed=shed, blocks=blocks, market=market)

    without = bound(None)
    with_blocks = bound((units, prices))
    assert with_blocks > without, "the blocks term is not in the bound at all"
    term = with_blocks - without
    # 10 units * max(0, 7 - 1) on the one block-day that has a price.
    assert term == 60.0, f"the sells' term reads {term}, not 10 * (7 - 1) = 60"

    # And a block whose price is below the carry contributes nothing: it is not sold.
    flat = np.zeros((days, days, 1))
    assert bound((units, flat)) == without, (
        "a block priced below the carry was counted anyway")
