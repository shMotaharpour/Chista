r"""The Walrasian master: duals for the contractor, by tâtonnement (#12).

The auctioneer. It publishes a wage vector `w` (the farm-internal input
prices); the contractor answers with each tile's best plan at those
prices (`TileContractor.price`, #32); the master looks at what those
plans would consume of the farm's shared resources, re-prices what
demand exceeds supply, and publishes again. Product prices `p` stay at
the market's own quotes throughout — the farm is a price taker (F033:
the ladder reads the shared board inventory, not our plans; F035:
prices rise through the season regardless), so the only prices the
master sets are the INTERNAL ones.

## What ties the tiles together (the coupling rows)

A tile's DP is independent *given prices*. The rows below are the only
things that make tiles compete, and each names its finding (issue #12
brief §2 — do not invent a row):

| row (resource column) | source |
|---|---|
| `LABOR_HOURS` | F039 fib hire costs, F040 a hand's first hour is lost |
| `FERTILIZER` | F006/F023: one stock, many tiles want it |
| `WHEAT` | F017/F018: animal feed eats the shared wheat stock |
| `SEED_*` ×5 | F001: seeds bypass the shed, bought from one purse |
| `ANIMAL_*` ×3 | F016: placed from one purse |

Two rows the #12 body lists are deliberately NOT here in M3, both named:

- **cash** (`M_d`) needs the revenue-enters-at-d+1 model (contract 2 of
  #8), which the body itself says to confirm before wiring.
  TODO(#12 M4): the cash row, once that contract is probed.
- **shed capacity** (F043) binds unsold produce; the columns carry no
  unsold-stock state, so the row has nothing to couple yet.
  TODO(#14): the day owns storage; the row lands with it.

## The LP (one solve per round)

```
max   Σ_k  λ_k · revenue_k                 revenue_k = produce_k · p_mkt
s.t.  Σ_k  λ_k · cost_k[r,d]  ≤  supply[r,d]   ∀ r, d   → y[r,d] ≥ 0
      Σ_k  λ_k  =  n_tiles                              → σ
```

`revenue_k` is the column's market value at the observed quotes; the
internal resources are priced ONLY by the rows' duals. That is the
classical DW master: the dual y IS the internal shadow price, and the
tâtonnement below feeds it back to the tiles.

`λ` is FRACTIONAL on purpose — Dantzig-Wolfe prices come from the LP
relaxation; making one tile one plan is #13's rounding problem.

`scipy.optimize.linprog(method="highs")` — scipy is imported at module
load, never inside the turn (probe, #12 brief §3: `scipy.optimize`
imports in 0.30 s — a first-turn cost paid once). HiGHS's ≤-row
marginals come back NEGATIVE in min-form (measured: a row worth 3
reports `[-3]`); the duals published here are their negation, clamped
with `np.maximum(y, 0)` before the contractor ever sees them (R006:
HiGHS returns tiny negatives at degenerate optima).

## Tâtonnement, damped

```
w_next = (1 − α)·w_lag + α·max(y, 0)         α = 0.5 (measured — tests)
w_next = np.maximum(w_next, 0)               # R006, projected every round
```

- Warm start: yesterday's published `w` (`(days, N_RESOURCE)` form, as
  the runtime caches it) — F035 says the market moves slowly; starting
  from scratch every day wastes the budget.
- **The publish is `max(engine quote, dual)` on every purchasable
  input.** A coupling dual is the INTERNAL shadow price of sharing a
  stock the farm already owns; the farm can also buy the item at the
  engine's quote (seeds/animals are fixed engine costs, wheat and
  fertilizer are quoted on the market, hours cost `_hire_cost`, F039).
  A dual BELOW the quote would tell the tiles the input is cheaper than
  it is, so the quote is the floor and the dual can only raise it.
  (Measured consequence: with slack stocks the duals sit at zero and
  the publish correctly falls back to the engine's own prices.)
- **Stop conditions** (in this order):
  1. every coupling dual moved less than `TOL_DUAL` this round
     (dual-stationarity — one coin is the smallest change that can move
     an engine decision: quotes are integer coins),
  2. `ITER_CAP` rounds spent — the budget decides this, never the
     convergence test (#12 brief §4: never let the loop decide).
- Interruptible at every round: `poll()` between rounds; on
  `TimeoutError` the caller still holds the incumbent `(p, w_lag)` —
  always a usable answer (#9's deadline contract).

## Fallback (required, not optional — #12 brief §3)

If linprog is not importable (the module-level import is GUARDED so a
bare import cannot take the whole submission down at load), or the
contractor or a solve fails, the master publishes its LAGGED prices with
no update — degraded, not dead. `MasterResult.used_fallback` +
`fallback_reason` carry which path fired, so a run never silently mixes
master-priced and stand-in-priced days. Both paths are tested:
`test_fallback_fires_on_solver_error` (a raising solver) and
`test_scipy_absent_is_survivable` (a subprocess whose scipy import is
genuinely blocked, R007-verified in its failing direction).

Every number has a source (R005): the row set names findings, α carries
its measured sweep (`tests/test_master.py`), the day-0 supply model
names its sources in `CouplingSupply`, and the market quotes come from
the observation through `dual_stand_in` (#32).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
try:
    from scipy.optimize import linprog
    HAS_SCIPY = True
except ImportError:                     # scipy is optional at import time
    # The guard is import hygiene, not a forecast. A bare module-level
    # import takes the WHOLE SUBMISSION down at load if scipy is missing
    # (measured: an import hook that blocks scipy raises straight out of
    # `import agent.planner.master`, so the fallback below could never run —
    # there is nothing left to fall back FROM). Three lines buy that
    # back, and they route into the solve-failure path that has to exist
    # anyway. No claim is made here about what any image carries: the
    # probe runs that would have measured the grading image returned
    # nothing, and R005 forbids reasoning from a measurement we do not
    # have.
    linprog = None
    HAS_SCIPY = False

from agent.planner.inputs import dual_stand_in
from agent.world.model import (ANIMALS, CROPS, N_RESOURCE, RESOURCE_ID,
                               RES_LABOR)
from agent.tile_dp.contractor import PricedBoard

LABOR_ID = RESOURCE_ID[RES_LABOR]
FERT_ID = RESOURCE_ID["FERTILIZER"]
WHEAT_ID = RESOURCE_ID["WHEAT"]

# The coupling rows: farm-internal resources one tile's plan competes
# for. Row r of the LP is resource column COUPLING_IDS[r] of the
# per-day cost vectors — no second vocabulary to drift (R005).
COUPLING_IDS: tuple[int, ...] = (
    LABOR_ID,                                    # F039/F040
    FERT_ID,                                     # F006/F023
    WHEAT_ID,                                    # F017/F018
    RESOURCE_ID["SEED_WHEAT"],                   # F001: one purse
    RESOURCE_ID["SEED_CARROT"],
    RESOURCE_ID["SEED_TOMATO"],
    RESOURCE_ID["SEED_STRAWBERRY"],
    RESOURCE_ID["SEED_MELON"],
    RESOURCE_ID["ANIMAL_GOOSE"],                 # F016: one purse
    RESOURCE_ID["ANIMAL_COW"],
    RESOURCE_ID["ANIMAL_SHEEP"],
)
N_COUPLING = len(COUPLING_IDS)

# Market-priced produce: what the farm sells (priced INTO the objective
# at the observation's quotes — exogenous to the tiles, F033/F035).
MARKET_IDS: tuple[int, ...] = (
    RESOURCE_ID["WHEAT"], RESOURCE_ID["CARROT"], RESOURCE_ID["TOMATO"],
    RESOURCE_ID["STRAWBERRY"], RESOURCE_ID["MELON"],
    RESOURCE_ID["EGG"], RESOURCE_ID["MILK"], RESOURCE_ID["WOOL"],
)
SEED_IDS = tuple(RESOURCE_ID[f"SEED_{c}"] for c in
                 ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON"))
ANIMAL_IDS = tuple(RESOURCE_ID[f"ANIMAL_{a}"] for a in
                   ("GOOSE", "COW", "SHEEP"))

# Tâtonnement damping. α = 0.5 is the MEASURED choice: the sweep over
# (0.2, 0.35, 0.5, 0.7) on a fixed two-class board is pinned in
# tests/test_master.py::test_alpha_sweep_is_the_evidence (the checked-in
# price trajectory the #12 brief asks for); 0.5 reached dual-stationarity
# fewest rounds on it. R005: the sweep lives in the test, this constant
# names it.
ALPHA = 0.5

# Dual-stationarity tolerance: one coin. Engine quotes are integer
# coins; a smaller dual move cannot change any hire / buy decision.
TOL_DUAL = 1.0

# Iteration cap. The full loop's cost is dominated by the contractor's
# re-pricing sweep (~9.3 ms measured, tests/test_master.py::test_budget
# prints the breakdown), not the LP: 8 rounds measured ~80 ms total on
# this box. The #12 acceptance ceiling (full CG round ≤ 45 ms) is met
# by a SINGLE round; the cap of 8 is the OSCILLATION budget and the
# loop must be left earlier by the deadline. TODO(#12): re-derive the
# cap from timing_contended on the grader (A4) when the rung is wired;
# the cap must come off the contended number, not this floor.
ITER_CAP_DEFAULT = 8

# A single master round (contractor sweep + LP solve), measured:
# ~10 ms on the dev box (test_budget prints the live number). The
# brief's 45 ms full-round ceiling leaves ~4x headroom for the
# grader's 1.17-1.41x slowdown (agent/runtime.py P-series probes).
ROUND_BUDGET_MS = 45.0

# The M3 overhead the #12 brief names for H_d ("start at 35% and
# measure"): hours the day's routing/carry will eat.
# TODO(#14): replace with the realised fraction the day reports.
HOURS_OVERHEAD = 0.35


@dataclass(frozen=True)
class CouplingSupply:
    """The farm-level resources the LP's rows share out (per day).

    Sources (R005), day-0 flat model:
    - `hours`: 24·(1 + hands) − hands (F040: a hand hired at hour 0
      first acts at hour 1), times (1 − HOURS_OVERHEAD).
    - `seed_stock`: the private purse counts (F001: seeds bypass the
      shed), shared evenly across the horizon — TODO(#12 M4): the
      per-day cash model replaces the flat split.
    - `fert_stock` / `wheat_feed_stock`: shed counts (F023: animals
      refill fertilizer nightly; the flat model under-counts, safely).
    """

    hours: np.ndarray             # (days,) float
    seed_stock: np.ndarray        # (5,) int, SEED_IDS order
    animal_stock: np.ndarray      # (3,) int, ANIMAL_IDS order
    fert_stock: float
    wheat_feed_stock: float


@dataclass
class MasterResult:
    """What one `equilibrate` call hands back — with its evidence."""

    p: np.ndarray                 # (days, N_RESOURCE) published product prices
    w: np.ndarray                 # (days, N_RESOURCE) published input prices
    duals: np.ndarray             # (days, N_COUPLING) clamped coupling duals
    lam: np.ndarray               # (n_cols,) the final LP mix (fractional)
    objective: float              # LP objective at the final round
    rounds: int
    converged: bool               # dual movement fell under TOL_DUAL
    used_fallback: bool = False
    fallback_reason: str = ""
    p_source: str = ""            # where the product price path came from
    history: list = field(default_factory=list)   # per-round max dual move


def published_duals(w_coupling: np.ndarray, days: int) -> np.ndarray:
    """`(days, N_COUPLING)` duals -> the `(days, N_RESOURCE)` wage matrix
    the contractor consumes: duals on their resource columns, zeros
    elsewhere (market-priced products are NOT input-priced to tiles —
    their value sits in the objective, not in w)."""
    out = np.zeros((days, N_RESOURCE), dtype=np.float64)
    for r, rid in enumerate(COUPLING_IDS):
        out[:, rid] = w_coupling[:, r]
    return out


def supply_from_obs(obs) -> CouplingSupply:
    """The farm's day-0 coupling supply from the observation (R005).

    Hours: 24·(1 + hands) − hands (F040), times (1 − 35 %) overhead
    (the #12 brief's M3 constant). Purse/shed counts via the same
    `private` fields the decode reads; animals per species from the
    shed (BUY_ANIMAL deposits there, `_commit_unit`).
    """
    farms = obs.get("farms", []) if isinstance(obs, dict) else []
    player = int(obs.get("player", 0)) if isinstance(obs, dict) else 0
    farm = farms[player] if len(farms) > player else {}
    private = obs.get("private", {}) if isinstance(obs, dict) else {}
    seeds = dict(private.get("seeds", {}) or {})
    shed = dict(private.get("shed", {}) or {})

    hands = len(farm.get("hands", []) or [])
    gross = 24.0 * (1 + hands) - hands          # F040
    hours = np.full(30, gross * (1.0 - HOURS_OVERHEAD))
    seed_stock = np.array([int(seeds.get(c, 0)) for c in CROPS],
                          dtype=np.int64)
    animal_stock = np.array([int(shed.get(a, 0)) for a in ANIMALS],
                            dtype=np.int64)
    return CouplingSupply(hours=hours, seed_stock=seed_stock,
                          animal_stock=animal_stock,
                          fert_stock=float(shed.get("FERTILIZER", 0)),
                          wheat_feed_stock=float(shed.get("WHEAT", 0)))


def _solve_lp(cost: np.ndarray, revenue: np.ndarray, supply: CouplingSupply,
              days: int, n_tiles: int) -> tuple[np.ndarray, np.ndarray, float]:
    """One restricted-master solve -> (lam, duals (days, N_C), objective).

    `cost` (n_cols, days, N_COUPLING) INCLUDING the trailing idle column;
    `revenue` (n_cols,); `n_tiles` is the convexity right-hand side — the
    number of REAL tiles, one plan-weight each (the idle column is a
    column, not a tile). Raises RuntimeError on solver failure OR on an
    absent scipy (the guarded import, module top) — the caller decides
    fallback.
    """
    if not HAS_SCIPY:
        raise RuntimeError("master LP failed: scipy is not available")
    n_cols = cost.shape[0]
    # rows: N_COUPLING·days ≤-constraints (r outer, d inner), one convexity.
    A_ub = cost.transpose(0, 2, 1).reshape(n_cols, -1).T          # (rows, cols)
    b_ub = np.empty(A_ub.shape[0])
    for r, rid in enumerate(COUPLING_IDS):
        lo, hi = r * days, (r + 1) * days
        if rid == LABOR_ID:
            b_ub[lo:hi] = supply.hours[:days]
        elif rid == FERT_ID:
            b_ub[lo:hi] = supply.fert_stock / days
        elif rid == WHEAT_ID:
            b_ub[lo:hi] = supply.wheat_feed_stock / days
        elif rid in SEED_IDS:
            b_ub[lo:hi] = supply.seed_stock[SEED_IDS.index(rid)] / days
        else:
            b_ub[lo:hi] = (supply.animal_stock[ANIMAL_IDS.index(rid)]
                           / days)
    res = linprog(-revenue, A_ub=A_ub, b_ub=b_ub,
                  A_eq=np.ones((1, n_cols)), b_eq=np.array([float(n_tiles)]),
                  bounds=[(0.0, None)] * n_cols, method="highs")
    if not res.success:
        raise RuntimeError(f"master LP failed: {res.message}")
    # HiGHS ≤-marginals are negative in min form; duals = −marginals,
    # clamped onto R006's orthant before anything sees them.
    y = np.maximum(-np.asarray(res.ineqlin.marginals), 0.0)
    return np.asarray(res.x), y.reshape(N_COUPLING, days).T, -float(res.fun)


def _validate_cost(cost: np.ndarray) -> None:
    """Reject columns whose coupling cost is negative anywhere.

    The graph's edge_cost is a consumption vector (chain_requirements:
    LABOR_HOURS + inputs, never netted against produce), so a negative
    component here means a caller built the matrix wrong — say so
    instead of letting the LP price a phantom supply."""
    if cost.size and float(cost.min()) < 0.0:
        raise RuntimeError(
            f"master LP failed: negative coupling cost {float(cost.min())} "
            "(chain costs are consumption vectors; the matrix is built wrong)")


def _idle_column(cost: np.ndarray) -> np.ndarray:
    """A zero-cost, zero-revenue column appended to the LP's column set.

    Every real column commits its tile to a chain for all 30 days, and
    the convexity row forces `Σλ = n_tiles` — so a board whose priced
    chains cannot fit under the supply (a goose place with no goose in
    the shed) is INFEASIBLE, not just unprofitable. The idle column is
    what the tiles fall back to: `λ_idle` absorbs any tile whose plans
    are unaffordable, which is exactly the DW semantics of "leave it
    idle". Its cost row is zeros, so it never binds any coupling row.
    """
    return np.zeros_like(cost)


def _product_price_path(obs, days: int, p_flat: np.ndarray,
                        config=None) -> tuple[np.ndarray, str]:
    """The product rows of `p`: the market forecast (#15), or flat quotes.

    F035: prices rise through the season, so the flat stand-in under-prices
    every later day of the horizon. `day/market.py` walks the town's
    own consumption forward and re-prices through the engine's price
    function (R002 — imported, never transcribed), so a day-20 harvest is
    priced on the day-20 curve. Any failure keeps the flat path and SAYS
    so: the master must never fail for a forecast.
    """
    import os
    if os.environ.get("CHISTA_MARKET_FORECAST", "1") != "1":
        return p_flat, "flat stand-in (CHISTA_MARKET_FORECAST=0)"
    try:
        from agent.belief.market import forecast as _forecast
        from agent.belief.market import price_paths
        fc = _forecast(obs, days=days, config=config)
        paths = price_paths(fc, days=days)
    except Exception as exc:                     # noqa: BLE001 - degrade
        return p_flat, f"flat stand-in (forecast failed: {type(exc).__name__})"
    out = p_flat.copy()
    for item, path in paths.items():
        rid = RESOURCE_ID.get(item)
        if rid is None or rid not in MARKET_IDS:
            continue
        out[:, rid] = [float(path[min(day, len(path) - 1)])
                       for day in range(days)]
    return out, f"market forecast (#15, unlock policy {fc.unlock_policy})"


def equilibrate(runtime, obs, contractor, supply: CouplingSupply,
                w_warm: np.ndarray | None = None,
                iter_cap: int = ITER_CAP_DEFAULT,
                poll=None, owned: list[int] | None = None) -> MasterResult:
    """Tâtonnement to (approximate) equilibrium; always publishable.

    The loop is a true Dantzig-Wolfe round: at the current duals the
    contractor RE-PRICES the board (a price change changes which chain
    is each tile's best response — the columns must follow), the LP
    re-solves over the fresh columns, the duals damp onto the new
    prices. Stopping early still leaves a usable incumbent: the last
    `(p, published_w)` pair is on the result at every point.

    Never raises for solver trouble — the fallback publishes the warm
    prices and says so. `poll()` (the rung's deadline bail) may raise:
    the incumbent `(p, w_lag)` is on the result object either way.
    """
    days = contractor.days
    p_mkt_full, w_stand_full = dual_stand_in(obs)
    p = p_mkt_full[:days]
    # #15: the product rows of `p` come from the market forecast (F035's
    # rising path); the flat stand-in is the documented fallback.
    p, p_source = _product_price_path(obs, days, p)
    p_mkt = p[:, list(MARKET_IDS)]
    # The engine-quote floor (see the publish rule in the docstring):
    # the stand-in wages ARE the engine's own prices for the inputs.
    w_floor = published_duals(w_stand_full[:days], days)[:, COUPLING_IDS]
    # Warm start: yesterday's published w (resource form) -> coupling form.
    if w_warm is not None:
        w_lag = np.asarray(w_warm, dtype=np.float64)[:days][:, COUPLING_IDS].copy()
    else:
        w_lag = w_floor.copy()
    w_lag = np.maximum(w_lag, 0.0)              # R006 on the warm start too

    result = MasterResult(p=p, w=published_duals(w_lag, days), duals=w_lag,
                          lam=np.zeros(0), objective=0.0, rounds=0,
                          converged=False, p_source=p_source)

    def _fallback(reason: str) -> MasterResult:
        result.used_fallback = True
        result.fallback_reason = reason
        return result

    if owned is None:
        owned = _owned_states(runtime, obs)
    # Fallback trigger 1: linprog was not importable. The module still
    # imports (guarded import above); the publish degrades to the warm
    # prices and SAYS so.
    if not HAS_SCIPY:
        return _fallback("scipy unavailable: linprog not importable")
    deadline = getattr(runtime, "_deadline", None)
    t_end = (time.perf_counter() + deadline.remaining_ms() / 1000.0 - 0.020
             if deadline is not None else None)

    w_cur = w_lag
    y_prev: np.ndarray | None = None
    converged = False
    priced_any = False
    for _ in range(max(1, iter_cap)):
        if poll is not None:
            poll()
        # (re)price the board at the current duals — columns follow prices
        try:
            board = contractor.price(p, published_duals(w_cur, days), owned)
        except Exception as exc:                # noqa: BLE001 - degraded, not dead
            if not priced_any:
                return _fallback(f"contractor failed: "
                                 f"{type(exc).__name__}: {exc}")
            break                                # keep the incumbent
        if board.columns.shape[0] == 0:
            # No owned tile: nothing to couple; the warm publish IS the answer.
            result.converged = True
            return result
        priced_any = True
        cost = board.per_day_cost[:, :, COUPLING_IDS].astype(np.float64)
        _validate_cost(cost)
        produce_mkt = board.per_day_produce[:, :, list(MARKET_IDS)] \
            .astype(np.float64)
        revenue = (produce_mkt * p_mkt[None, :, :]).sum(axis=(1, 2))
        # the idle column (see _idle_column): tiles fall back to it
        n_tiles = cost.shape[0]
        # ONE column, not a copy of the tensor: `np.zeros_like(cost)` doubles the
        # column count, and with a single tile (1 + 1) that accidentally matched
        # the revenue vector, so the bug only appeared once the board carried
        # more than one priced tile (measured: 100 tiles -> A_ub 200 columns
        # against a 101-long objective, linprog refused the problem).
        idle = np.zeros((1,) + cost.shape[1:], dtype=cost.dtype)
        cost = np.vstack([cost, idle])
        revenue = np.append(revenue, 0.0)

        try:
            lam, y, obj = _solve_lp(cost, revenue, supply, days, n_tiles)
        except RuntimeError as exc:
            if result.rounds == 0:
                return _fallback(str(exc)[:200])
            break                                # keep the incumbent
        result.rounds += 1
        # Dual-stationarity compares the LP's duals to the previous
        # ROUND's duals, not to the damped publish: after a slack row
        # stops binding its dual drops to 0 while the damped w_lag is
        # still decaying — that decay is the publish converging, not
        # the auctioneer oscillating.
        move = (float(np.max(np.abs(y - y_prev))) if y_prev is not None
                else float(np.max(np.abs(y - w_cur))))
        y_prev = y
        result.history.append(move)
        # Damp, project (R006), then raise onto the engine-quote floor:
        # the farm can BUY any coupling input at the quote, so the dual
        # may raise the input's price above it but never undercut it.
        w_cur = np.maximum((1.0 - ALPHA) * w_cur + ALPHA * y, 0.0)
        w_cur = np.maximum(w_cur, w_floor)
        result.duals = w_cur
        result.lam = lam
        result.objective = obj
        if move < TOL_DUAL:
            converged = True
            break
        if t_end is not None and time.perf_counter() >= t_end:
            break                               # the budget decides

    result.converged = converged
    result.w = published_duals(w_cur, days)
    return result


_GRAPH = None


def _shipped_graph():
    """The shipped tile graph, loaded once per process and never rebuilt."""
    global _GRAPH
    if _GRAPH is None:
        from agent.planner.inputs import load_contractor
        _GRAPH = load_contractor().graph
    return _GRAPH


def _owned_states(runtime, obs) -> list[int]:
    """The graph state ids of the tiles we own, via #32's decode."""
    from agent.obs import decode_world
    from agent.planner.inputs import load_contractor, unit_state_ids
    # The graph is cast once per PROCESS, here. It used to be cached on the
    # runtime object (`runtime._replan_resources`), which is why `equilibrate`
    # needed a rung to be handed one at all - a coordinator that writes to the
    # thing that calls it is not a coordinator. `runtime` stays in the
    # signature for the deadline `poll` and nothing else.
    graph = _shipped_graph()
    view = decode_world(obs, at_day_start=True,
                        graph_keys=frozenset(graph.key_index))
    ids = unit_state_ids(view, graph)
    return [s for s in ids if s is not None]
