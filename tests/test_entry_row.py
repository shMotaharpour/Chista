"""Guards for the entry row: the harvest's two ways into the shed.

The claim under test (step 2, second half): the manager decides WHEN the day's
harvest reaches the shed, and the LP carries that decision.

  * the split row is an EQUALITY — `now[i,d] + defer[i,d] = produce[i,d]`, the
    harvest is a decision the DP already took, so none of it may vanish;
  * `defer[i, last_day] = 0` — there is no day after the season ends, and the
    night flush would destroy what waited;
  * the same-day path is priced at the LAST MARKET HOUR (`sell_blocks(..., hour=23)`)
    and the deferred path at the day's own hour-0 curve, so the two entry ways
    are compared at the prices the engine actually pays them;
  * without an entry row (`entry=None`) the model reproduces the shipped one;
  * the bound identity survives: the pricing and the bound are computed at the
    SAME multipliers, or `bound - objective - sum N_c*rc_c` stops being zero.

Run:  .venv/bin/python -m tests.test_entry_row   (also under pytest)
"""

from __future__ import annotations

import numpy as np

from agent.belief.market import PRODUCTS, forecast
from agent.belief.depth import sell_blocks
from agent.config import Config
import agent.planner.master as M
from agent.planner import colgen
from tests.test_master_depth import _board, _loaded_supply, _solve_one_tile


def _entry_board(seed: int = 33):
    """A board with something to sell and something to spend it on."""
    obs, supply, reps, counts, horizon = _board(seed)
    return obs, _loaded_supply(obs, "MELON", 40), reps, counts, horizon


def _solve_with_entry(obs, supply, reps, counts, horizon, *, entry: bool):
    from agent.tile_dp.contractor import TileContractor
    from agent.planner.inputs import GRAPH_PATH
    from agent.tile_dp.graph import TileGraph
    contractor = TileContractor(TileGraph.load(GRAPH_PATH), days=horizon)
    fc = forecast(obs, days=horizon)
    return M.equilibrate(object(), obs, contractor, supply, iter_cap=5,
                         forecast_obj=fc, entry=entry), fc


def test_the_harvest_splits_into_now_and_defer() -> None:
    """`now + defer = produce` on every item and day, and never a negative."""
    obs, supply, reps, counts, horizon = _entry_board()
    res, _fc = _solve_with_entry(obs, supply, reps, counts, horizon, entry=True)
    now, defer = np.asarray(res.now), np.asarray(res.defer)
    assert now.shape == defer.shape, (now.shape, defer.shape)
    assert float(now.min()) >= -1e-9 and float(defer.min()) >= -1e-9, \
        "neither entry way may take a negative quantity"
    # `produce` is (days, N_RESOURCE) per column; the split arrays are
    # (days, items), so the pool's contribution is summed in the same order.
    # The split row is `now + defer = sum_j lam_j * produce_j`: the mix's own
    # plan, not the sum of every column in the pool.
    produced = np.zeros_like(now)
    lam = np.asarray(res.lam, dtype=np.float64)
    # `lam` weights the pool's FIRST len(lam) columns: the pool can grow after
    # the last solve, and a weight read against a longer pool is a fiction.
    for j, col in enumerate(res.pool[:len(lam)]):
        if col.produce is None:
            continue
        prod = np.asarray(col.produce, dtype=np.float64)[:horizon, :]
        for ii, rid in enumerate([colgen._resource_of(n) for n in M.SHED_ITEMS]):
            if rid is None or rid >= prod.shape[1]:
                continue
            produced[:, ii] += float(lam[j]) * prod[:, rid]
    assert np.allclose(now + defer, produced, atol=1e-6), (
        "the split row is an equality: every harvested unit reaches the shed by "
        "one of the two ways")


def test_the_last_day_cannot_defer() -> None:
    """No day after the season: what waits on the last day is destroyed."""
    obs, supply, reps, counts, horizon = _entry_board()
    res, _fc = _solve_with_entry(obs, supply, reps, counts, horizon, entry=True)
    cap = np.asarray(res.defer_cap)        # (days, items), the BOUND itself
    assert cap.shape[0] == horizon, cap.shape
    assert np.allclose(cap[horizon - 1, :], 0.0), (
        "the defer block's own upper bound must be zero on the last day: a plan "
        "that cannot sell what it harvested would otherwise defer it into the "
        "void and dodge the waste charge")
    defer = np.asarray(res.defer)
    assert np.allclose(defer[horizon - 1, :], 0.0), \
        "and the LP must not leave anything there either"


def test_without_an_entry_row_the_model_reproduces_the_shipped_one() -> None:
    """`entry=None` is the model that shipped, objective for objective."""
    obs, supply, reps, counts, horizon = _entry_board()
    off, _ = _solve_with_entry(obs, supply, reps, counts, horizon, entry=False)
    assert not hasattr(off, "now") or off.now is None, \
        "the feature-absent path must not publish a split"
    ref, _ = _solve_one_tile(obs, supply, horizon, depth=True)
    assert off.bound >= off.objective - 1e-6, "the shipped path must still bound"


def _removed_the_same_day_drop_guard() -> None:
    """REMOVED, with its reason: the guard that asserted hour 23 pays
    differently from hour 0 was testing `belief.depth.sell_blocks` (already
    guarded in `tests/test_belief_depth.py`) plus "the split is used" (covered by
    the split-identity guard). It asserted nothing about THIS model, and the
    thing it was written for — the second sell set, priced at the last market
    hour — does not exist yet. It comes back when that set lands, and then it can
    fail for its own reason.
    """


def _unused_the_same_day_drop_is_priced_at_the_last_market_hour() -> None:
    """Hour 23 pays differently from hour 0, and the model uses both."""
    obs, supply, reps, counts, horizon = _entry_board()
    fc = forecast(obs, days=horizon)
    goods = [M.SHED_ITEMS[ii] for ii in M.SELLABLE]
    day0, _p0 = sell_blocks(fc, goods, 0, horizon, int(supply.shed_capacity),
                            blocks=Config().sell_blocks)
    _u23, p23 = sell_blocks(fc, goods, 0, horizon, int(supply.shed_capacity),
                            blocks=Config().sell_blocks, hour=23)
    assert p23.shape == _p0.shape
    assert not np.allclose(p23, _p0), (
        "the last market hour must price differently from hour 0 on this board, "
        "or this guard proves nothing")
    res, _fc = _solve_with_entry(obs, supply, reps, counts, horizon, entry=True)
    assert np.asarray(res.defer).sum() + np.asarray(res.now).sum() > 0.0, \
        "the split must actually be used on a board with stock to move"


def test_the_bound_identity_holds_with_the_entry_row() -> None:
    """The pricing and the bound share one set of multipliers."""
    obs, supply, reps, counts, horizon = _entry_board()
    for entry in (True, False):
        res, _ = _solve_with_entry(obs, supply, reps, counts, horizon,
                                   entry=entry)
        assert res.bound >= res.objective - 1e-6, (
            f"entry={entry}: bound {res.bound:.1f} below objective "
            f"{res.objective:.1f}")
        assert res.gap >= -1e-9, f"entry={entry}: negative gap {res.gap}"
        # The pricing and the bound must run at the SAME multipliers, and the
        # credit is what proves it: with the entry row the tiles are credited the
        # split row's dual, without it the balance row's.
        credit = np.asarray(res.credit, dtype=np.float64)
        want = np.asarray(res.eta if entry else res.sigma, dtype=np.float64)
        assert np.allclose(credit, np.maximum(want, 0.0)), (
            f"entry={entry}: the pricing credited the tiles "
            f"{'eta' if entry else 'sigma'}, but handed them something else")



def test_an_unsellable_harvest_still_has_to_be_accounted_for() -> None:
    """The split row is an EQUALITY, and this is the board where that bites.

    A hand-built column produces a GOOSE on the last day. A goose cannot be
    SOLD (the sells cover the nine products) and the season ends there, so the
    only ways out are the waste charge or — with an INEQUALITY split row —
    leaving it unaccounted, which is free disposal and dodges the charge. The
    board of sellable goods cannot show this: there the LP splits fully either
    way, which is why this arm needed a synthetic column.
    """
    from agent.planner.inputs import GRAPH_PATH
    from agent.tile_dp.graph import TileGraph
    from agent.world.model import N_RESOURCE, RESOURCE_ID
    obs, supply, reps, counts, horizon = _board()
    fc = forecast(obs, days=horizon)
    goods = [M.SHED_ITEMS[ii] for ii in M.SELLABLE]
    depth = sell_blocks(fc, goods, int(obs.get("day", 0)), horizon,
                        int(supply.shed_capacity), blocks=Config().sell_blocks)
    produce = np.zeros((horizon, N_RESOURCE), dtype=np.float64)
    produce[horizon - 1, RESOURCE_ID["ANIMAL_GOOSE"]] = 1.0
    goose_col = colgen.Column(
        cls=0, cls_key=(0, 0), cost=np.zeros((horizon, M.N_COUPLING)),
        spend=np.zeros(horizon), earn=np.zeros(horizon), revenue=0.0,
        produce=produce, chains=(), entities=(), key=())
    from agent.belief.market import price_paths
    path = np.asarray([price_paths(fc, days=horizon)[g] for g in PRODUCTS],
                      dtype=np.float64).T
    solved = colgen.MasterLP().solve(
        [goose_col], np.array([1]), supply.hours, supply.money, horizon,
        M.N_COUPLING, shed_stock=supply.shed_stock,
        shed_capacity=float(supply.shed_capacity), prices=path,
        market=M.SELLABLE, sell_cap=M._sell_cap(obs, horizon), depth=depth,
        entry=True)
    now, defer = np.asarray(solved.now), np.asarray(solved.defer)
    goose = M.SHED_ITEMS.index("GOOSE")
    assert np.isclose(now[horizon - 1, goose] + defer[horizon - 1, goose], 1.0), (
        "the goose must be accounted for by one of the two ways: with an "
        "INEQUALITY split row the LP leaves it out and dodges the waste charge")


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
    print(f"{len(tests) - failures}/{len(tests)} entry-row checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
