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

from agent.planner import colgen
from agent.planner.inputs import dual_stand_in
from agent.world.model import (ANIMALS, CROPS, N_RESOURCE, RESOURCE_ID,
                               RES_LABOR)
from agent.tile_dp.contractor import PricedBoard
from agent.world.rules import ANIMAL_RULES, CROP_RULES

LABOR_ID = RESOURCE_ID[RES_LABOR]
FERT_ID = RESOURCE_ID["FERTILIZER"]
WHEAT_ID = RESOURCE_ID["WHEAT"]

# The coupling rows: farm-internal resources one tile's plan competes
# for. Row r of the LP is resource column COUPLING_IDS[r] of the
# per-day cost vectors — no second vocabulary to drift (R005).
# ONE quantity row, because hours are the only thing the farm cannot buy.
# FERTILIZER (F006/F023) and WHEAT (F017/F018) were rows here and are not any
# more, for the same reason the seeds and animals went: they are purchasable,
# their stock on a day-0 board is zero, and a `≤ 0` row on a purchasable input
# is not scarcity — it is a wrong model that forbids the plan outright. Their
# price reaches the tiles through the quote, and their scarcity through the
# cash row. What a plan of theirs is worth is the objective's business.
COUPLING_IDS: tuple[int, ...] = (
    LABOR_ID,                                    # F039/F040
)
N_COUPLING = len(COUPLING_IDS)

# The five rows #12's brief names, and `columns.py:ROW_NAMES` expects:
#   labour · cash_out · wheat_net · fert_net · stored
# Three of them are the quantity rows above. `cash_out` is the row below.
# `stored` is still absent and still honest about it: a column carries no
# SELL decision, so what it leaves in the shed overnight is not derivable
# from it. TODO(#79): the day layer's arrivals are the state it was
# waiting for, but the column has to carry the sell side first.
#
# Eight rows were here that the brief never listed: one per seed and one
# per animal, each bounded by the stock in the purse. They are the reason
# the master could not price a board. Seeds and animals are PURCHASABLE
# (F001, F016) and on day 0 the purse holds none of them, so every such
# row read `≤ 0` and every column that plants or places violated it — the
# LP was left with the idle column, objective 0, and the duals on those
# rows diverged (measured: 2.44e3 per hour after 8 rounds). A purchasable
# input is not scarce in QUANTITY, it is scarce in MONEY, and the row it
# belongs in is the cash row. Its quote already reaches the tiles through
# `w_floor` below.
PURCHASE_IDS: tuple[int, ...] = (
    RESOURCE_ID["SEED_WHEAT"], RESOURCE_ID["SEED_CARROT"],
    RESOURCE_ID["SEED_TOMATO"], RESOURCE_ID["SEED_STRAWBERRY"],
    RESOURCE_ID["SEED_MELON"],
    RESOURCE_ID["ANIMAL_GOOSE"], RESOURCE_ID["ANIMAL_COW"],
    RESOURCE_ID["ANIMAL_SHEEP"],
    WHEAT_ID, FERT_ID,
)

# Market-priced produce: what the farm sells (priced INTO the objective
# at the observation's quotes — exogenous to the tiles, F033/F035).
MARKET_IDS: tuple[int, ...] = (
    RESOURCE_ID["WHEAT"], RESOURCE_ID["CARROT"], RESOURCE_ID["TOMATO"],
    RESOURCE_ID["STRAWBERRY"], RESOURCE_ID["MELON"],
    RESOURCE_ID["EGG"], RESOURCE_ID["MILK"], RESOURCE_ID["WOOL"],
    # FERTILIZER is a product and the market quotes it, so a column that
    # produces it earns. Leaving it out made the DP and the master disagree
    # about the same plan: a tile collecting 19 fertilizer was paid for them
    # in `tile_values` and credited nothing in the column's revenue, so the
    # reduced cost never reached zero and the loop stalled 60 short of a proof
    # on a 52,279 objective. It is also in PURCHASE_IDS, which is not a
    # contradiction: produced and bought at the same quote, internal use is a
    # wash and only a net producer is paid.
    RESOURCE_ID["FERTILIZER"],
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
    #: Coins on hand at the start of the horizon. The cash row's day-0
    #: right-hand side; later days add the revenue the plan banks.
    money: float = 0.0
    #: What one unit of each PURCHASE_IDS input costs, same order. Seeds
    #: and animals are engine tables; WHEAT and FERTILIZER are the
    #: observation's own quotes, because those two the market sells back.
    #: Never None once constructed: a supply built by hand (every test that
    #: probes one row in isolation does) would otherwise take the whole master
    #: down inside `column_cash`, and the fallback would report it as a
    #: pricing failure rather than as the missing field it is.
    quotes: np.ndarray = None     # (len(PURCHASE_IDS),) float

    def __post_init__(self) -> None:
        if self.quotes is None:
            object.__setattr__(self, "quotes",
                               np.zeros(len(PURCHASE_IDS), dtype=np.float64))


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
    cash_duals: np.ndarray = None  # (days,) shadow price of a coin per day
    #: The columns `lam` weights, and the classes they belong to. A mix is
    #: useless without them: `columns.assign_tiles` has to know WHICH plan each
    #: weight is for, and re-pricing at the published duals gives a different
    #: board and a different answer.
    pool: list = field(default_factory=list)
    classes: tuple = ()            # (reps, counts, class of each owned tile)
    mu: np.ndarray = None          # (n_classes,) convexity duals, signed
    #: True only when a pricing round found no class with a positive reduced
    #: cost. `converged` is kept as its alias for the callers that read it.
    certified: bool = False
    stopped: str = ""              # why the loop ended, when it was not certified


def published_duals(w_coupling: np.ndarray, days: int,
                    cash_dual: np.ndarray | None = None,
                    quotes: np.ndarray | None = None) -> np.ndarray:
    """`(days, N_COUPLING)` duals -> the `(days, N_RESOURCE)` wage matrix
    the contractor consumes: duals on their resource columns, zeros
    elsewhere (market-priced products are NOT input-priced to tiles —
    their value sits in the objective, not in w).

    A purchasable input has no quantity row, so without `quotes` it would
    reach the tiles at zero and every chain would look free. Its price is
    its quote, raised by the cash row's dual:

        w[d, r] = quote_r · (1 + y_cash[d])

    `y_cash[d]` is the shadow price of a coin on day `d` — what the plan's
    objective would gain from one more. When the purse is slack it is 0 and
    the input costs its quote; when the purse binds, every seed and every
    animal gets dearer in proportion, and the tiles re-plan onto chains that
    need fewer of them. This is the term that tells identical tiles apart:
    a quantity row is one number for all of them and can only move WHAT the
    identical answer is, never that it is identical.
    """
    out = np.zeros((days, N_RESOURCE), dtype=np.float64)
    for r, rid in enumerate(COUPLING_IDS):
        out[:, rid] = w_coupling[:, r]
    if quotes is not None:
        scale = 1.0 + (np.zeros(days) if cash_dual is None
                       else np.asarray(cash_dual, dtype=np.float64)[:days])
        for i, rid in enumerate(PURCHASE_IDS):
            # WHEAT and FERTILIZER are both a quantity row and purchasable:
            # the dearer of the two prices is the one a tile faces.
            out[:, rid] = np.maximum(out[:, rid], float(quotes[i]) * scale)
    return np.maximum(out, 0.0)


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
    prices = (obs.get("market", {}) or {}).get("prices", {}) or {}
    quotes = np.array(
        [float(CROP_RULES[c]["seed"]) for c in CROPS]
        + [float(ANIMAL_RULES[a]["cost"]) for a in ANIMALS]
        + [float(prices.get("WHEAT", 0.0)), float(prices.get("FERTILIZER", 0.0))],
        dtype=np.float64)
    return CouplingSupply(hours=hours, seed_stock=seed_stock,
                          animal_stock=animal_stock,
                          fert_stock=float(shed.get("FERTILIZER", 0)),
                          wheat_feed_stock=float(shed.get("WHEAT", 0)),
                          money=float(farm.get("money", 0.0)),
                          quotes=quotes)


def column_cash(board: PricedBoard, supply: CouplingSupply, days: int
                ) -> tuple[np.ndarray, np.ndarray]:
    """What each priced column spends and earns, per day.

    `spend` is the column's purchasable inputs at their quotes. The stock
    already in the purse is NOT netted off here: it is one endowment shared
    by every tile, and a per-column subtraction would hand the same seed to
    all of them. Ignoring it prices the plan slightly high, which is the
    safe direction — a plan that fits with the stock ignored fits with it.

    `earn` comes from the caller, which holds the product price path; the
    zeros here are a shape, not an answer.
    """
    cost = board.per_day_cost[:, :days, PURCHASE_IDS].astype(np.float64)
    spend = (cost * np.asarray(supply.quotes)[None, None, :]).sum(axis=2)
    return spend, np.zeros_like(spend)


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

    # ---- the column-generation loop -------------------------------------
    # One subproblem per CLASS, not per tile: tiles in the same graph state
    # have the same answer, so pricing both prices twice. `colgen` accumulates
    # the columns across rounds and stops on the reduced-cost certificate.
    reps, counts, of_tile = colgen.classes_of(owned)
    result.classes = (reps, counts, of_tile)
    idle = [colgen.Column(cls=c, cost=np.zeros((days, N_COUPLING)),
                          spend=np.zeros(days), earn=np.zeros(days),
                          revenue=0.0, key=("idle",))
            for c in range(len(reps))]

    w_cur = w_lag
    state = {"w": w_lag, "cash": np.zeros(days), "failed": None}

    def price(y, cash):
        """The subproblem: price each class at the master's OWN duals.

        Exactly those duals, not damped ones and not raised onto a floor. The
        reduced-cost test `value + mu` is only a reduced cost of the LP the
        duals came from; price the tiles at anything else and the test stops
        being about that LP. That is not a theoretical worry - the stall
        detector caught it: with every row slack the contractor priced at the
        engine-quote floor while the master's duals were 0, the same plan came
        back round after round with rc 1597, and the loop could neither add it
        nor prove it was done.

        Damping still happens, on what the REST of the agent reads (`result.w`)
        - the duals oscillate, lesson 1.9 §4 records it, and a consumer chasing
        them is a consumer thrashing. It just may not touch the pricing step.

        The floor is gone from here and it is not missed: `published_duals`
        already puts each purchasable input at its quote, which IS the floor
        for everything that can be bought. Putting it in the dual as well
        charged the plan twice for the same seed.
        """
        # The product price the subproblem sees is NOT the market quote. A
        # unit produced on day d is sold, and the coins relieve the cash row on
        # every day after it — so the tile is worth the quote PLUS the cash
        # those coins unlock. Without this term the DP is charged for what it
        # spends and credited nothing for what it earns, the subproblem stops
        # being the master's reduced cost, and the loop stalls short of a
        # proof: measured at rc 60 on a 52,279 objective, with the same column
        # coming back round after round.
        cash_arr = np.asarray(cash, dtype=np.float64)
        later = np.concatenate([np.cumsum(cash_arr[::-1])[::-1][1:], [0.0]])
        p_eff = p.copy()
        p_eff[:days, list(MARKET_IDS)] *= (1.0 + later[:days])[:, None]
        exact = published_duals(y, days, cash, supply.quotes)
        state["w"] = np.maximum(
            np.maximum((1.0 - ALPHA) * state["w"] + ALPHA * y, 0.0), w_floor)
        state["cash"] = (1.0 - ALPHA) * state["cash"] + ALPHA * np.asarray(cash)
        board = contractor.price(p_eff, exact, reps)
        cost = board.per_day_cost[:, :days, COUPLING_IDS].astype(np.float64)
        _validate_cost(cost)
        produce = board.per_day_produce[:, :days, list(MARKET_IDS)] \
            .astype(np.float64)
        earn = (produce * p_mkt[:days][None, :, :]).sum(axis=2)
        spend, _ = column_cash(board, supply, days)
        columns = [
            colgen.Column(cls=c, cost=cost[c], spend=spend[c], earn=earn[c],
                          revenue=float(earn[c].sum()),
                          chains=tuple(board.plans[c]) if c < len(board.plans) else (),
                          key=colgen.column_key(board, c, days))
            for c in range(len(reps))]
        return np.asarray(board.tile_values, dtype=np.float64), columns

    try:
        cg = colgen.generate(price, supply.hours, supply.money, counts, days,
                             N_COUPLING, idle, rounds=max(1, iter_cap),
                             poll=poll, deadline=t_end)
    except RuntimeError as exc:
        return _fallback(str(exc)[:200])
    except Exception as exc:                    # noqa: BLE001 - degraded, not dead
        return _fallback(f"pricing failed: {type(exc).__name__}: {exc}")

    w_cur = state["w"]
    result.cash_duals = state["cash"]
    result.rounds = cg.rounds
    result.pool = cg.pool
    result.certified = cg.certified
    result.stopped = cg.stopped
    result.history = list(cg.rc_history)
    if cg.solve is not None:
        result.lam = cg.solve.lam
        result.objective = cg.solve.objective
        result.duals = state["w"]
        result.mu = cg.solve.mu
    converged = cg.certified

    result.converged = converged
    result.w = published_duals(w_cur, days, result.cash_duals, supply.quotes)
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
    from agent.obs import LOCKED_KEY
    # The graph is cast once per PROCESS, here. It used to be cached on the
    # runtime object (`runtime._replan_resources`), which is why `equilibrate`
    # needed a rung to be handed one at all - a coordinator that writes to the
    # thing that calls it is not a coordinator. `runtime` stays in the
    # signature for the deadline `poll` and nothing else.
    graph = _shipped_graph()
    view = decode_world(obs, at_day_start=True,
                        graph_keys=frozenset(graph.key_index))
    # Every tile we own, NOT the tile each worker stands on. This read
    # `unit_state_ids(view, graph)` while its docstring said "the tiles we
    # own": on a day-0 board that is one farmer, so the master priced ONE
    # column against 25 owned tiles, the convexity row read `Σλ = 1`, and
    # the idle column took all of it. A LOCKED tile is not ours (F042).
    keys = np.asarray(view.me.keys).reshape(-1)
    return [int(graph.key_index[int(k)]) for k in keys
            if int(k) != LOCKED_KEY and int(k) in graph.key_index]
