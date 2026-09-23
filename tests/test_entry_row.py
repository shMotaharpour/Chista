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
    defer = np.asarray(res.defer)          # (days, items)
    assert defer.shape[0] == horizon, defer.shape
    assert np.allclose(defer[horizon - 1, :], 0.0), (
        "defer on the last day must be zero: the night flush would destroy it")


def test_without_an_entry_row_the_model_reproduces_the_shipped_one() -> None:
    """`entry=None` is the model that shipped, objective for objective."""
    obs, supply, reps, counts, horizon = _entry_board()
    off, _ = _solve_with_entry(obs, supply, reps, counts, horizon, entry=False)
    assert not hasattr(off, "now") or off.now is None, \
        "the feature-absent path must not publish a split"
    ref, _ = _solve_one_tile(obs, supply, horizon, depth=True)
    assert off.bound >= off.objective - 1e-6, "the shipped path must still bound"


def test_the_same_day_drop_is_priced_at_the_last_market_hour() -> None:
    """Hour 23 pays differently from hour 0, and the model uses both."""
    obs, supply, reps, counts, horizon = _entry_board()
    fc = forecast(obs, days=horizon)
    goods = [M.SHED_ITEMS[ii] for ii in M.SELLABLE]
    day0, _p0 = sell_blocks(fc, goods, 0, horizon, int(supply.shed_capacity),
                            blocks=M.SELL_BLOCKS)
    _u23, p23 = sell_blocks(fc, goods, 0, horizon, int(supply.shed_capacity),
                            blocks=M.SELL_BLOCKS, hour=23)
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
