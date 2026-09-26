"""Guards for the season horizon (F029) and the pool carried across days.

There is no horizon after the season ends. A plan made on day 0 starts at 30
days — the DP needs the whole season to value what it plants — and every day the
farm moves forward one, the horizon shrinks by one. Two things follow, and both
are guarded here:

- the master prices `DAYS - day` days, so no row of its product price
  path is a day the season does not have. The path used to be padded
  (`path[min(day, len(path) - 1)]`), which repeated the last modelled quote over
  days the forecast never walked — measured on a real day-26 board, `p` carried
  20 rows, i.e. days 26..45, sixteen of which do not exist;
- a column is a plan for the days of the board it was BUILT on, so carrying it
  into the next day moves its day-indexed arrays up one (`colgen.advance_pool`).
  A column that did not move would be dropped by `generate`'s shape check (its
  `cost` has one row too many) and the warm start would be gone every day — and
  the `chains`/`entities` the day layer commits are day 0 of the plan, so a
  stale column would hand the board yesterday's chain.

The same pass drops the columns the LP gave no weight to, which is what stops
the pool growing without bound (measured by a reviewer: 126 columns on day 0 to
1,054 on day 29).

Run:  .venv/bin/python -m tests.test_season_horizon   (also under pytest)
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from agent.planner import colgen as CG
from agent.planner import master as M
from agent.planner.inputs import load_contractor
from agent.world.model import N_RESOURCE

PASS = {"farmer": ["PASS"], "hands": [], "market": []}

#: The season's last day: 30 days, day 0..29. Here so a guard can say where a
#: row would have to sit to be past the season.
LAST_DAY = 29


def _obs_at(day: int) -> dict:
    """A real board `day` days into a season nothing has played on.

    FastSim, not a hand-built dict: the master reads the board's tiles, the
    market's inventory and the town through the same decode the runtime does.
    The seat is silent (PASS every turn), so what the day carries is the
    season's own weather and market and no plan of ours.
    """
    from offline_lab.fast_sim import FastSim
    sim = FastSim({"episodeSteps": 24 * 30, "seed": 0})
    sim.reset()
    for _ in range(day * 24):
        sim.step([dict(PASS), dict(PASS)])
    obs = dict(sim.observations()[0])
    assert int(obs["day"]) == day, f"the fixture landed on day {obs['day']}"
    return obs


class _Path:
    """A forecast stub: `price_paths` reads `price_of` and `first_day`.

    Rising by one coin a day from `first_day`, so a padded row is a DIFFERENT
    number from the row that belongs there and the padding cannot pass unnoticed.
    """

    def __init__(self, days: int, first_day: int = 26, base: int = 100) -> None:
        self.days = int(days)
        self.first_day = int(first_day)
        self.unlock_policy = "stub"
        self._base = int(base)

    def price_of(self, item, day) -> int:
        return self._base + (int(day) - self.first_day)


# --- the horizon ----------------------------------------------------------

def test_the_horizon_on_a_late_board_is_the_season_that_is_left():
    """A day-26 board prices 4 days, and every row of `p` is one of them.

    Both halves are the guard. The horizon is `DAYS - day`, and the rows
    of the product price path are the quotes for days 26, 27, 28 and 29 — not
    four copies of a padded quote, and not the twenty rows a fixed look-ahead
    gives (days 26..45, sixteen past the season).
    """
    from agent.belief.market import forecast, price_paths
    from agent.planner.master import MARKET_IDS
    from agent.world.model import RESOURCE_NAMES

    obs = _obs_at(26)
    assert M.season_horizon(obs) == 4, "the horizon is the season that is left"

    result = M.equilibrate(object(), obs, load_contractor(days=20),
                           M.supply_from_obs(obs), iter_cap=6)
    assert not result.used_fallback, result.fallback_reason
    p = np.asarray(result.p)
    assert p.shape[0] == 4, (
        f"a day-26 board priced {p.shape[0]} days: the horizon is not the "
        f"season that is left ({M.season_horizon(obs)})")
    assert 26 + p.shape[0] - 1 <= LAST_DAY, "a row of `p` is past the season"

    paths = price_paths(forecast(obs, days=4), days=4)
    for k in range(p.shape[0]):
        for rid in MARKET_IDS:
            name = RESOURCE_NAMES[rid]
            if name not in paths:
                continue
            assert float(p[k, rid]) == float(paths[name][k]), (
                f"row {k} of `p` is not the quote for day {26 + k}: "
                f"{float(p[k, rid])} against {float(paths[name][k])}")


def test_the_horizon_is_the_season_and_the_two_season_constants_agree():
    """Day 0 prices the whole season, day 29 one day, and the 30s agree.

    The horizon rule reads the tile DP's own season length (the horizon its
    sweep can run over) while the world spells the season `DAYS` (F029). Two
    constants for one fact drift silently, so the guard pins them equal — and
    pins the ends of the shrink, which is the owner's rule: the plan
    STARTS at 30 days on day 0 and each day the farm moves forward one, the
    horizon shrinks by one.
    """
    from agent.world.rules import DAYS
    from agent.tile_dp.contractor import HORIZON_DAYS

    assert HORIZON_DAYS == DAYS, (
        f"the DP's season ({HORIZON_DAYS}) and the world's ({DAYS}) have "
        f"drifted apart")
    assert M.season_horizon({"day": 0}) == 30
    assert M.season_horizon({"day": 1}) == 29
    assert M.season_horizon({"day": 26}) == 4
    assert M.season_horizon({"day": 29}) == 1
    assert M.season_horizon({"day": 30}) == 1, "a day past the season still has one"


def test_the_contractor_never_prices_a_day_the_season_does_not_have():
    """It shrinks to the season that is left, and honours a shorter look-ahead.

    The contractor's `days` is the horizon of the sweep, and the master's duals
    are handed to it shaped `(days, N_RESOURCE)` — so the two must be the same
    number, and a contractor that overshoots the season has to be rebuilt rather
    than sliced (`price_many` refuses a shorter matrix: measured
    `ValueError: prices: expected shape (20, 18), got (4, 18)`). A contractor
    built for FEWER days than the season has left is the caller's own
    look-ahead and is priced as it stands — every fixture in the suite hands one.
    """
    late, early = _obs_at(26), _obs_at(0)
    assert M.priced_contractor(load_contractor(days=20), late).days == 4
    assert M.priced_contractor(load_contractor(days=30), late).days == 4
    assert M.priced_contractor(load_contractor(days=4), late).days == 4
    assert M.priced_contractor(load_contractor(days=4), early).days == 4
    assert M.priced_contractor(load_contractor(days=20), early).days == 20
    assert M.priced_contractor(load_contractor(days=30), early).days == 30


def test_a_forecast_shorter_than_the_horizon_is_not_padded():
    """A forecast that does not cover the horizon degrades — it is not padded.

    `MarketForecast.price_of` clamps onto its last modelled day, so asking a
    two-day forecast for day 28 returns day 27's quote: the pad
    `path[min(day, len(path) - 1)]` used to write by hand, one layer down, and a
    price for a day the forecast never walked. The flat stand-in is the
    documented fallback and names the shortfall; the guard reads both arms, so a
    forecast that covers the horizon still prices row by row.
    """
    from agent.planner.master import MARKET_IDS

    obs = _obs_at(26)
    flat = np.full((4, N_RESOURCE), 7.0)

    out, source = M._product_price_path(obs, 4, flat, forecast_obj=_Path(2))
    assert "covers 2 of 4 days" in source, source
    assert np.array_equal(out, flat), (
        "a two-day forecast priced days 28 and 29 instead of degrading")

    full, source = M._product_price_path(obs, 4, flat, forecast_obj=_Path(4))
    assert "market forecast" in source, source
    for rid in MARKET_IDS:
        assert [float(v) for v in full[:, rid]] == [100.0, 101.0, 102.0, 103.0], (
            f"row by row, `p`'s market rows are days 26..29 at the forecast's "
            f"own quotes; got {[float(v) for v in full[:, rid]]}")


# --- the carried pool -----------------------------------------------------

def _column(days: int = 4, cls: int = 0) -> CG.Column:
    """A column shaped like the ones the pricing builds, with countable values."""
    cost = np.arange(days * 1, dtype=np.float64).reshape(days, 1) + 1.0
    spend = np.arange(days, dtype=np.float64) + 10.0
    earn = np.arange(days, dtype=np.float64) + 100.0
    produce = np.zeros((days, N_RESOURCE), dtype=np.float64)
    produce[:, 0] = np.arange(days, dtype=np.float64) + 1.0
    return CG.Column(
        cls=cls, cost=cost, spend=spend, earn=earn,
        revenue=float(earn.sum()), produce=produce,
        chains=tuple((d, 7, 100 + d) for d in range(days)),
        entities=tuple(f"E{d}" for d in range(days)),
        cls_key=(11, 2), key=tuple((d, 100 + d, d) for d in range(days)))


def test_a_carried_column_moves_up_a_day():
    """Every day-indexed field of a carried column is one day further on.

    `cost`, `spend`, `earn` and `produce` are `(days, ...)`, `chains` and
    `entities` are the per-day plan the day layer reads at index 0, and `key` is
    the plan's own signature — four arrays and three day-indexed tuples, all of
    them wrong if they do not move.
    """
    col = _column(4)
    out = CG.advance_pool([col])
    assert len(out) == 1
    moved = out[0]
    assert np.array_equal(moved.cost, col.cost[1:])
    assert np.array_equal(moved.spend, col.spend[1:])
    assert np.array_equal(moved.earn, col.earn[1:])
    assert np.array_equal(moved.produce, col.produce[1:])
    assert moved.revenue == float(col.earn[1:].sum())
    assert moved.cls_key == col.cls_key
    assert moved.cls == col.cls
    assert moved.chains == col.chains[1:], (
        "the day layer commits `chains[0]`: a column that did not move hands it "
        "yesterday's chain")
    assert moved.entities == col.entities[1:]
    assert moved.key == tuple((d - 1, ch, ent) for d, ch, ent in col.key[1:]), (
        "a stale key makes the loop refuse a plan the pool no longer holds")


def test_the_carry_drops_the_columns_the_lp_gave_no_weight():
    """λ = 0 is dropped, an unpriced column is kept, and an idle one goes.

    The weight is the last solve's mix, in pool order: a column the master does
    not use is a plan the next day's pricing can rediscover, and carrying it is
    what made the pool grow to 1,054 columns by day 29. A column BEYOND the mix
    was never priced (added after the final solve), so there is no evidence
    against it and it is kept. An idle column carries no `produce`, so it cannot
    be re-priced — and `generate` re-seeds one per class anyway.
    """
    pool = [_column(4, cls=0), _column(4, cls=1), _column(4, cls=2),
            replace(_column(4, cls=3), produce=None)]
    lam = np.array([0.0, 3.0, 0.0])            # the idle column has no weight
    out = CG.advance_pool(pool, lam)
    assert [c.cls for c in out] == [1], (
        f"kept {[c.cls for c in out]}: only the weighted column may survive")
    out = CG.advance_pool(pool, np.array([0.0, 0.0, 0.0]))
    assert out == [], "every column had zero weight and none was dropped"
    out = CG.advance_pool(pool, np.array([0.0]))
    assert [c.cls for c in out] == [1, 2], (
        "a column beyond the mix was never priced and must be kept")


def test_the_manager_moves_the_pool_and_the_horizon_with_the_day():
    """`observe` on day 1: the horizon shrinks, and yesterday's plan comes along.

    Driven through the manager's own entry point, because that is where the day
    boundary is. The carried column is one the day-1 pricing cannot return — its
    cost arrays are a pattern no board produces — so finding it in the pool after
    the second observe is evidence it was CARRIED, not re-derived.
    """
    from agent.config import Config
    from agent.manager import core as MC

    manager = MC.Manager(Config())
    manager.observe(_obs_at(0), {"farmHandCostMult": 1})
    assert manager.contractor.days == 30, "day 0 prices the whole season"
    pool0 = list(manager.pool)
    assert pool0, "the day-0 solve found no columns at all"

    # The pool opens with one idle column per class (`produce` is None), and an
    # idle column is not carried — `generate` re-seeds it every call. The marker
    # rides a column that has a plan to move.
    source = next(c for c in pool0 if c.produce is not None)
    marker = replace(source,
                     cost=np.full_like(np.asarray(source.cost, dtype=float), 1e3))
    manager.pool = pool0 + [marker]

    manager.observe(_obs_at(1), {"farmHandCostMult": 1})
    assert manager.contractor.days == 29, (
        f"day 1 priced {manager.contractor.days} days, not the 29 the season has "
        f"left")
    carried = np.asarray(marker.cost, dtype=float)[1:]
    assert any(np.array_equal(np.asarray(c.cost, dtype=float), carried)
               for c in manager.pool), (
        "the column the manager was carrying is not in the day-1 pool: it was "
        "dropped instead of moved up a day")
    for column in manager.pool:
        assert np.asarray(column.cost).shape[0] == 29, (
            f"a column in the day-1 pool is {np.asarray(column.cost).shape[0]} "
            f"days long, not 29")


def test_the_manager_prunes_on_the_mix_of_the_solve_that_made_the_pool(monkeypatch):
    """The weight that travels with the pool is the LAST solve's, not hour 0's.

    `generate` rebuilds the pool as [idle, warm, new] on every call, so the pool
    `Manager.step` leaves behind is in a different order from the one `observe`
    solved — and a weight indexed against the wrong order drops the columns the
    last solve was using. The guard reads the last solve's own `(pool, lam)`
    pair, then what the day roll hands the carry: same pool objects, same order,
    same weights.
    """
    from agent.config import Config
    from agent.manager import core as MC
    from agent.planner import master as M

    solves: list = []
    real = M.equilibrate

    def spy(*args, **kwargs):
        result = real(*args, **kwargs)
        solves.append(result)
        return result

    handed: dict = {}
    real_advance = MC.advance_pool

    def spy_advance(pool, lam=None, **kwargs):
        handed["pool"] = list(pool)
        handed["lam"] = None if lam is None else np.asarray(lam, dtype=float)
        return real_advance(pool, lam, **kwargs)

    monkeypatch.setattr(M, "equilibrate", spy)
    monkeypatch.setattr(MC, "advance_pool", spy_advance)

    manager = MC.Manager(Config())
    manager.observe(_obs_at(0), {"farmHandCostMult": 1})
    assert handed == {}, "the day-0 observe rolled a day it had not seen yet"
    day0_pool = list(manager.pool)
    assert day0_pool, "the day-0 solve found no columns at all"

    # Hours 1..23 of the SAME day: a fresh `generate`, so the pool comes back
    # reordered ([idle, warm, new]) and its mix belongs to the new order.
    manager.step(_obs_at(0))
    assert len(solves) >= 2, (
        "the day-0 observe certified in ONE round, so no step solve reordered "
        "the pool: this guard needs a day that is still solving")
    last = solves[-1]
    assert last.pool and np.asarray(last.lam).size, "the step left no mix behind"
    assert np.array_equal(np.asarray(manager.lam, dtype=float),
                          np.asarray(last.lam, dtype=float)), (
        "the manager is holding a mix from another solve than the pool it holds")

    manager.observe(_obs_at(1), {"farmHandCostMult": 1})
    assert handed.get("lam") is not None, "the carry ran without the solve's mix"
    assert len(handed["pool"]) == len(last.pool), (
        f"the carry was handed {len(handed['pool'])} columns while the last "
        f"solve's pool holds {len(last.pool)}")
    assert [id(c) for c in handed["pool"]] == [id(c) for c in last.pool], (
        "the carry was handed a different pool than the last solve solved")
    assert np.array_equal(handed["lam"], np.asarray(last.lam, dtype=float)), (
        "the weights handed to the carry are not the last solve's mix: a weight "
        "indexed against another order prunes the columns the master is using")
    assert (np.asarray(handed["lam"]) == 0.0).any(), (
        "the mix has no zero weight, so nothing can be pruned on it")


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
    print(f"{len(tests) - failures}/{len(tests)} season-horizon checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
