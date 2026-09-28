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
  the manager hands it back) — F035 says the market moves slowly; starting
  from scratch every day spends rounds rediscovering what is already priced.
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
  1. the column-generation CERTIFICATE: a pricing round found no class whose
     reduced cost clears `Config.rc_tol` on this board's objective, i.e. the
     pool already holds everything worth holding (`cg.certified`),
  2. the round cap (`cfg.iter_cap`, the manager asks with `cfg.master_rounds`)
     — the cap decides this, never a self-declared convergence (#12 brief §4:
     never let the loop decide). A dual-movement threshold was once claimed as
     a stop here and never was one; `converged` is the certificate's alias.
- The loop is not interruptible by a clock. It used to poll a deadline
  between rounds; a round count that follows the machine makes two runs of
  one seed disagree (18,312 against 28,890 with every RNG in our code seeded
  and the threads pinned to one), so the cap above is the only other stop.
  Whatever the loop has found when it stops is a usable incumbent.

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

from dataclasses import dataclass, field, replace

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

from agent.config import Config
from agent.planner import colgen
from agent.planner.inputs import dual_stand_in
from agent.world.terms import EngineTerms
from agent.world.model import (ANIMALS, CROPS, N_RESOURCE, PRODUCTS,
                               RESOURCE_ID, RES_LABOR, SHED_ITEMS)
from agent.tile_dp.contractor import HORIZON_DAYS, PricedBoard
from agent.planner.colgen import _resource_of

#: The shed items the market sells, as INDICES into `SHED_ITEMS` — the products
#: come first in that tuple, and only they can be sold (the species are placed,
#: never sold). The master's sell variables are laid out in this order.
SELLABLE: tuple[int, ...] = tuple(range(len(PRODUCTS)))
from agent.world.rules import (ANIMAL_RULES, CROP_RULES, SHED_CAPACITY,
                               TURNS_PER_DAY)

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

# The solve's own numbers are NOT here. Every cap the master runs under lives in
# `agent/config.Config` and arrives as `cfg=` (`Agent(config)` injects it, and
# `Config.load`/`dump` ship it beside the agent), so a measurement can run a
# different set without editing this module and two callers cannot disagree
# about one decision. What is left in this file is STRUCTURE: the row
# vocabulary, the column layout, the bound identity.
#
# The two numbers this block used to hold and no longer needs:
# - the iteration cap: `Config.iter_cap` (the library default) and
#   `Config.master_rounds` (what the manager asks for);
# - `ROUND_BUDGET_MS`, a wall-clock ceiling that only a test read: with the
#   clock out of the loop nothing in the agent may bound work by time.


@dataclass(frozen=True)
class CouplingSupply:
    """The farm-level resources the LP's rows share out (per day).

    Sources (R005), day-0 flat model:
    - `hours`: 24·(1 + hands) − hands (F040: a hand hired at hour 0
      first acts at hour 1), times `(1 - cfg.hours_overhead)`.
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
    #: The shed's own counts, in `PRODUCTS + ANIMALS` order (12 items). The
    #: engine's shed holds exactly those (`_new_private`: `PRODUCTS +
    #: list(ANIMALS)`), the cap counts ALL of them together (`sum(shed.values())`
    #: at the DROP, the buy and the night flush), and seeds are separate
    #: (`private["seeds"]`, they never pass through the shed). This is the
    #: opening balance of the inventory rows the master prices the sell timing
    #: against — a plan that cannot see the animals already in the shed
    #: over-estimates the free room and schedules drops and sells the engine
    #: refuses or discards.
    shed_stock: np.ndarray = None      # (len(PRODUCTS) + len(ANIMALS),) int
    #: The shed's capacity, from the world's own constant (the engine's
    #: `shedCapacity`, default 100, and the run's config can override it).
    shed_capacity: float = float(SHED_CAPACITY)

    def __post_init__(self) -> None:
        if self.quotes is None:
            object.__setattr__(self, "quotes",
                               np.zeros(len(PURCHASE_IDS), dtype=np.float64))
        if self.shed_stock is None:
            object.__setattr__(
                self, "shed_stock",
                np.zeros(len(PRODUCTS) + len(ANIMALS), dtype=np.int64))


@dataclass
class MasterResult:
    """What one `equilibrate` call hands back — with its evidence."""

    p: np.ndarray                 # (days, N_RESOURCE) published product prices
    w: np.ndarray                 # (days, N_RESOURCE) published input prices
    duals: np.ndarray             # (days, N_COUPLING) clamped coupling duals
    lam: np.ndarray               # (n_cols,) the final LP mix (fractional)
    objective: float              # LP objective at the final round
    rounds: int
    converged: bool               # `cg.certified` (the pricing certificate)
    #: True when the DECISION solve was the MIP: `lam` is then integral and the
    #: solve's own duals are zero placeholders, so every published price below
    #: came from `lp_final` — the LP solve of the SAME matrix and the same pool,
    #: which is the only side of a MIP that has marginals at all.
    integral: bool = False
    #: That LP solve, when it happened (`integral` only). The pricing certificate
    #: and the gate "the MIP cannot beat its own relaxation" are both read here.
    lp_final: object = None
    #: (days,) the hands the model BOUGHT in the decision solve -- the labour
    #: row's `delta` block -- or None when that block was off. The day layer
    #: asks wsr about `round()` of the first day and hires exactly that; the
    #: bill is already inside the objective, so no caller adds it again.
    hands_bought: object = None
    used_fallback: bool = False
    fallback_reason: str = ""
    p_source: str = ""            # where the product price path came from
    history: list = field(default_factory=list)   # per-round max dual move
    cash_duals: np.ndarray = None  # (days,) shadow price of a coin per day
    #: (days,) the LP's OWN cash-row duals at the final round — the raw marginals,
    #: not the damped publish. `to_mixes` prices a plan's spend with them, because
    #: they are what the pricing charged the tiles (`quote·(1+ahead)`), and the
    #: #13 gap has to be measured in the same accounting as the objective (#142).
    cash_lp: np.ndarray = None
    #: (days, len(SHED_ITEMS)) the BALANCE rows' duals the LP solved with — what
    #: one unit of a good sitting in the shed is worth. It is the master's own
    #: price for a plan's OUTPUT (the produce feeds the stock and the stock is
    #: sold later), so it is the price `to_mixes` values a plan at. Empty without
    #: a shed.
    sigma: np.ndarray = None
    #: (days, items) what the plan DROPS by the last market hour of the day, and
    #: what waits for the night flush — the entry row's own decision.
    now: np.ndarray = None
    defer: np.ndarray = None
    #: (days, items) the entry rows' duals: the internal price of a harvested
    #: unit, which the pricing credits the tiles with when the entry row is on.
    eta: np.ndarray = None
    #: (days, items) the produce credit the pricing handed the tiles, and the
    #: defer block's own upper bounds: both are read by guards that would be
    #: blind if they had to infer them from the answer.
    credit: np.ndarray = None
    defer_cap: np.ndarray = None
    #: The columns `lam` weights, and the classes they belong to. A mix is
    #: useless without them: `columns.assign_tiles` has to know WHICH plan each
    #: weight is for, and re-pricing at the published duals gives a different
    #: board and a different answer.
    pool: list = field(default_factory=list)
    classes: tuple = ()            # (reps, counts, class of each owned tile)
    mu: np.ndarray = None          # (n_classes,) convexity duals, signed
    sells: np.ndarray = None       # (n_goods, days) planned market sales from the LP
    #: True only when a pricing round found no class with a positive reduced
    #: cost. `converged` is kept as its alias for the callers that read it.
    certified: bool = False
    stopped: str = ""              # why the loop ended, when it was not certified
    bound: float = float("inf")    # the best Lagrangian bound seen
    gap: float = float("inf")      # (bound - objective) / bound
    #: (days, hours) cells the #87 dead-zone clamp pulled under the edge.
    #: A nonzero count says the raw tâtonnement wanted a wage the DP cannot
    #: answer — the season then runs on the busiest legal wage instead.
    labour_clamped_cells: int = 0


def published_duals(w_coupling: np.ndarray, days: int,
                    cash_dual: np.ndarray | None = None,
                    quotes: np.ndarray | None = None,
                    sigma: np.ndarray | None = None,
                    market: tuple[int, ...] = ()) -> np.ndarray:
    """`(days, N_COUPLING)` duals -> the `(days, N_RESOURCE)` wage matrix
    the contractor consumes: duals on their resource columns, zeros
    elsewhere (market-priced products are NOT input-priced to tiles —
    their value sits in the objective, not in w).

    A purchasable input has no quantity row, so without `quotes` it would
    reach the tiles at zero and every chain would look free. Its price is
    its quote, raised by the cash row's dual:

        w[d, r] = quote_r · (1 + y_cash[d])

    `cash_dual[d]` here is `Σ_{d' ≥ d}` of the cash rows' duals — a coin spent
    on day d sits in every cumulative row from d onward. It is the shadow price
    of a coin — what the plan's
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
    if sigma is not None and len(market):
        # A stored good is an INPUT to the plan that consumes it — feed, a dose,
        # an animal going out to its plot — and with a shed its worth is the
        # balance row's dual, not the market quote. This is the PUBLISH map: what
        # the rest of the agent reads a stored unit at.
        #
        # It is deliberately NOT what the pricing step charges, and the two are
        # not the same statement: a plan's consumption is bought from the market
        # and charged to the cash row at its quote, so a subproblem that pays σ
        # for it as well pays twice, comes back with a class value below the
        # LP's own value of a column the pool already holds, and takes the
        # Lagrangian bound below the objective it bounds (measured: rc −12.93 /
        # −62.10 / −162.38 on the three working classes of a real board).
        sig = np.asarray(sigma, dtype=np.float64)[:days]
        for gi, ii in enumerate(market):
            rid = _resource_of(SHED_ITEMS[ii])
            if rid is not None:
                out[:, rid] = np.maximum(out[:, rid], sig[:, ii])
    return np.maximum(out, 0.0)


def _sell_cap(obs, days: int, risk_z: float = 0.0) -> tuple[np.ndarray, dict]:
    """`(1, len(PRODUCTS))`: how much the town will buy over the horizon.

    ONE number per good, because the town eats that much over the whole horizon
    and the sales may land on any day of it — belief's `drain_forecast` is
    already that total, and dividing it by the days was a wrong scale (it turned
    16 units a day into 54). The rival's share comes off it: a unit the rival
    pours into the same town is a unit the shops do not buy from us, and the same
    `supply_curve` the price path carries is where that is known.

    The rival's side is read through a guard, and its absence is not an error.
    `supply_curve` decodes the opponent's farm out of the observation, and a
    board with one farm in it (every fixture that prices a single farm, and any
    harness that hands the master a hand-built observation) has no rival to
    decode — measured: `IndexError: list index out of range` out of
    `decode_world`, which reached `equilibrate`'s fallback and left the master
    reporting a pricing failure it did not have. Without a rival the town's own
    drain IS the appetite: the rival's supply is a subtraction, not the model.
    """
    from agent.belief.opponent import GOODS, drain_forecast

    horizon = max(1, int(days)) * TURNS_PER_DAY
    drain, sd = drain_forecast(obs, horizon)
    total = np.asarray(drain, dtype=np.float64)[None, :len(PRODUCTS)]
    # The SAME call's second moment, kept instead of discarded: `risk_z` sd of
    # the drain falling short is the sale ladder walked `risk_z·sd` units up
    # (belief's own `opponent.quantile_price_floor`). Zero is the mean model.
    risk_pad = ({str(g): float(risk_z) * max(0.0, float(v))
                 for g, v in zip(GOODS, np.asarray(sd, dtype=np.float64))}
                if float(risk_z) > 0.0 else {})
    try:
        from agent.belief.rival_calendar import supply_curve
        rival = np.asarray(supply_curve(obs, PRODUCTS, days), dtype=np.float64)
    except Exception:                              # noqa: BLE001 - no rival
        return np.maximum(total, 0.0), risk_pad
    if rival.ndim == 2 and rival.shape[1] == total.shape[1]:
        total = total - rival.sum(axis=0)[None, :]
    return np.maximum(total, 0.0), risk_pad


def _shed_capacity(obs) -> int:
    """The shed's capacity for this run, resolved in one place.

    `EngineTerms` is what decides (the observation's own configuration when the
    harness carried one, else the world's transcription of the engine's default,
    kaggriculture.py:553); this function is only the call site.
    """
    return int(EngineTerms.from_obs(obs).shed_capacity)


def supply_from_obs(obs, cfg: "Config | None" = None) -> CouplingSupply:
    """The farm's day-0 coupling supply from the observation (R005).

    Hours: 24·(1 + hands) − hands (F040), times `(1 - cfg.hours_overhead)`
    (the #12 brief's M3 constant — the flat stand-in for the walking a route
    does beyond what a column charges). Purse/shed counts via the same
    `private` fields the decode reads; animals per species from the
    shed (BUY_ANIMAL deposits there, `_commit_unit`).
    """
    cfg = Config() if cfg is None else cfg
    farms = obs.get("farms", []) if isinstance(obs, dict) else []
    player = int(obs.get("player", 0)) if isinstance(obs, dict) else 0
    farm = farms[player] if len(farms) > player else {}
    private = obs.get("private", {}) if isinstance(obs, dict) else {}
    seeds = dict(private.get("seeds", {}) or {})
    shed = dict(private.get("shed", {}) or {})

    hands = len(farm.get("hands", []) or [])
    gross = 24.0 * (1 + hands) - hands          # F040
    hours = np.full(30, gross * (1.0 - cfg.hours_overhead))
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
                          quotes=quotes,
                          shed_stock=np.array(
                              [int(shed.get(item, 0))
                               for item in PRODUCTS + ANIMALS],
                              dtype=np.int64),
                          shed_capacity=float(_shed_capacity(obs)))


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


def depth_coins_from(fc, item: str, day: int, quote: float, start_units: int,
                     units: int) -> float:
    """The ladder's coins for `units`, walked from an inventory that pays `quote`.

    `belief.depth.depth_coins` walks from the forecast's own (drained) inventory.
    A sale that OFFSETS the drain has to be walked from the un-drained one, and
    the only honest way to say that without a second price model is to ask the
    engine's own quote function for the inventory that pays `quote` today.
    """
    from agent.belief.ladder import sell_coins
    from agent.world.prices import price as _quote
    # the engine's price is monotone in inventory: search the inventory that
    # quotes `quote` today, then walk the ladder from there.
    lo, hi = 0.0, 20000.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if float(_quote(item, mid)) > float(quote):
            lo = mid
        else:
            hi = mid
    inv0 = int(hi)
    return float(sell_coins(item, inv0, max(1, int(units)))
                 - sell_coins(item, inv0, max(0, int(start_units))))


def _product_price_path(obs, days: int, p_flat: np.ndarray,
                        forecast_obj=None, supply=None,
                        rival_supply=None,
                        cfg=None) -> tuple[np.ndarray, str]:
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
        # A caller that already has this turn's forecast hands it in: the same
        # curve prices the master's objective and re-times the day's sells, and
        # building it twice was 1.1 + 1.4 ms of the turn.
        # `belief.market.forecast` already takes the rival's supply as a DATED
        # curve (`rival_calendar.supply_curve`, #205) -- and this call was not
        # passing it, so the path the master priced every far day with carried
        # the town's drain and NOBODY's supply: measured on the seed-33 board,
        # MILK rose 160 -> 454 by day 29. The rival pours goods in too; with
        # their curve (and ours) on top, the far days stop being free money.
        fc = (forecast_obj if forecast_obj is not None
              else _forecast(obs, days=days, rival_supply=rival_supply))
        # Every row of `p` must be a day the season HAS, and the path is indexed
        # from the observation's own day (`price_paths(from_day=first_day)`), so
        # a forecast that covers the horizon puts day `days - 1` on the season's
        # last day and nothing beyond it. A forecast built for FEWER days than
        # the horizon cannot: `MarketForecast.price_of` clamps onto its last
        # modelled day, so the rows past its own end repeat a quote for a day
        # nobody walked — the same pad `path[min(day, len(path) - 1)]` used to
        # write here by hand, one layer down. It degrades to the flat stand-in
        # and SAYS so, the way every other forecast failure here does.
        if int(getattr(fc, "days", days)) < days:
            return p_flat, (f"flat stand-in (forecast covers "
                            f"{int(getattr(fc, 'days', 0))} of {days} days)")
        paths = price_paths(fc, days=days)
        # The path above is the forecast's OWN walk: the town drains and nobody
        # sells. Our plan does sell, and a good we pour in is worth what the
        # ladder pays for it AFTER our own supply -- `belief.depth.inventory_at`
        # plus the engine's own price function, never a transcribed one (R002).
        # `supply` is `{item: (units, ...)}` per day; without it, nothing changes.
        # The gate alone opens the block: `supply` may be absent, in which case the
        # declared default lot (`Config.sell_lot_default`) is what every good is
        # priced as. Requiring a supply dict here made the default lot DEAD CODE.
        if int(getattr(cfg, "price_supply_rounds", 0)) > 0:
            supply = supply or {}
            from agent.belief.depth import depth_coins
            first = int(obs.get("day", 0)) if isinstance(obs, dict) else 0
            # The DP plans against a price; the LP sells through the LADDER. A
            # rising path (F035) makes holding the crop to the peak look best --
            # and the peak is a price a LOT can never realise. So the price the
            # contractor plans with is the ladder's OWN average for the lot the
            # plan would sell that day (`depth_coins(lot)/lot`), which is the same
            # curve the master prices the sale with. Measured on the seed-33 board,
            # day 3: MILK quotes 175 but a 40-unit lot averages 126 (-28%); wheat
            # 30 -> 26; far days move ~1% (a scarce market is nearly flat).
            priced = {}
            # No caller has told us the lot yet (the hand-off the day layer still
            # owes): the SAFE default is the shed's own ceiling, the largest lot a
            # day can physically put on the market. That is the conservative end of
            # the ladder, and it is what stops a rising path from paying the PEAK
            # price for a lot no peak can absorb. The measured own-supply lot
            # replaces this as soon as the manager hands one in.
            default_lot = float(getattr(cfg, "sell_lot_default", 0.0))
            for item, path in paths.items():
                units = np.asarray(supply.get(item, np.full(days, default_lot)),
                                   dtype=np.float64)[:days]
                # The drain is counted TWICE unless the sale line offsets it (#151
                # point 2): the forecast's path rises because the town eats the
                # stock -- but the units WE sell land in the same market, so on the
                # days we sell, the inventory does not fall and the price does not
                # rise. Our own sale is therefore priced on the UN-DRAINED quote,
                # and the ladder is walked from there (our lot still pays its own
                # depth). Measured: MILK's day-29 quote 471 -> the flat quote.
                base_day = float(depth_coins(fc, item, first, 1))
                priced[item] = tuple(
                    float(depth_coins_from(fc, item, first, base_day, 1,
                                           max(1, int(round(float(units[d]))))))
                    / max(1.0, float(round(float(units[d])))) if units[d] > 0
                    else float(path[d])
                    for d in range(len(path)))
            paths = priced
    except Exception as exc:                     # noqa: BLE001 - degrade
        return p_flat, f"flat stand-in (forecast failed: {type(exc).__name__})"
    out = p_flat.copy()
    for item, path in paths.items():
        rid = RESOURCE_ID.get(item)
        if rid is None or rid not in MARKET_IDS:
            continue
        out[:, rid] = [float(path[day]) for day in range(days)]
    return out, f"market forecast (#15, unlock policy {fc.unlock_policy})"


def _repriced_pool(pool, p_mkt: np.ndarray, days: int) -> list:
    """The warm columns, priced at THIS board's product prices.

    A column's `earn` and `revenue` are its produce at the prices of the board it
    was BUILT on, and those prices move (F035: the market path rises through the
    season). Carried as they stand they make the master's LP a hybrid — its
    objective is yesterday's revenue under today's rows — and its optimum can
    then beat the Lagrangian bound built from today's class values. Measured on a
    real mid-season board: 14 warm columns exceeded their own class's value by
    1,246.4 in total, and at the class maxima that was exactly the bound's
    shortfall, 203.366 — the bound sat below the objective by precisely the
    amount the stale prices inflated.

    `produce` is kept for this, so the adoption re-prices instead of discarding.
    A column without it (an older pool, a hand-built one) is DROPPED: a number
    nobody can recompute is a number nobody can trust, and a stale revenue is
    worse than a smaller pool.
    """
    if not pool:
        return pool
    market = list(MARKET_IDS)
    out = []
    for column in pool:
        if column.produce is None:
            continue
        produce = np.asarray(column.produce, dtype=np.float64)
        if produce.ndim != 2 or produce.shape[0] < days:
            continue
        earn = (produce[:days][:, market] * p_mkt[:days]).sum(axis=1)
        out.append(replace(column, earn=earn, revenue=float(earn.sum())))
    return out


def season_horizon(obs) -> int:
    """The days a plan may look ahead: the season that is LEFT (F029).

    There is no horizon after the season ends. A plan made on day 0 starts at
    30 days — the DP needs the whole season to value what it builds — and every
    day the farm moves forward one, the horizon shrinks by one: day 1 -> 29,
    ..., day 29 -> 1. A fixed look-ahead is wrong at both ends of the season:
    20 days on day 0 stops the DP short of the season's own end (a crop it
    plants on day 19 has nowhere to be harvested), and 20 days on day 20 prices
    days 20..39, ten of which the season does not have.

    The season's length is read from the tile DP's own constant, because that
    is the horizon the sweep can run over at all — belief spells the same 30
    `world.rules.DAYS` (F029) and `tests/test_season_horizon.py` pins the two
    together, so a drift is a red test rather than a horizon nobody can price.

    `max(1, ...)`: a day past the last one has no horizon at all, and a caller
    that asks for one gets the single day it is standing on rather than an
    empty LP.
    """
    day = int(obs.get("day", 0)) if isinstance(obs, dict) else 0
    return max(1, int(HORIZON_DAYS) - day)


def priced_contractor(contractor, obs):
    """`contractor`, shrunk to the season that is left when it overshoots it.

    The horizon is a fact of the day (F029), and the contractor's `days` is the
    number of days its sweep runs over — so the two must agree, or the DP prices
    days the season does not have and the columns come back with rows past the
    horizon. A contractor built for FEWER days is honoured as it stands: a
    shorter look-ahead is a legal plan, and every fixture that prices a board
    hands one (20 days). A contractor built for MORE is rebuilt over its own
    graph, which costs 0.4 ms measured against the artifact read
    `load_contractor` pays.

    `manager._roll_day` builds the day's contractor at the remaining season
    already, so in the shipped path this is a no-op; it is here because
    `equilibrate` may be called by anything, and pricing past the season is a
    silent defect — the terminal row of the DP stops being the season's end.
    """
    days = season_horizon(obs)
    if int(contractor.days) <= days:
        return contractor
    from agent.tile_dp.contractor import TileContractor
    return TileContractor(contractor.graph, days=days)


def equilibrate(runtime, obs, contractor, supply: CouplingSupply,
                w_warm: np.ndarray | None = None,
                iter_cap: int | None = None,
                integral: bool = False,
                owned: list[int] | None = None,
                pool: list | None = None,
                forecast_obj=None,
                smoothing: float = 0.0,
                entry: bool = False,
                cfg: "Config | None" = None,
                buy_hands: bool = False,
                hand_mult: int = 0) -> MasterResult:
    """Column generation over the tile classes; always publishable.

    One round is one Dantzig-Wolfe round (lesson 1.9): the LP solves over EVERY
    column found so far — the pool accumulates, it is never replaced — the
    duals price each class's subproblem, the subproblem's best plan is added if
    its reduced cost clears the tolerance, and the loop stops when no class
    offers one. That stop is a certificate, not a flat objective.

    **The pricing step is the exact Lagrangian subproblem**, and two things
    depend on it. `p_eff` prices produce at `p·(1+later)` and `exact` prices a
    purchasable input at `quote·ahead`: the cash row prices SPENDING at `ahead`,
    so the base quote belongs to the row, not to the subproblem. And the walk is
    charged inside the DP's own objective (`TileContractor._travel_edge_costs`),
    not added to the column afterwards. Charging either one in the wrong place
    leaves the DP optimising a different objective than the master prices, and
    then `values` is not the class's dual-priced value: the reduced-cost test is
    no longer about this LP, the certificate proves nothing, and the Lagrangian
    bound can come out BELOW the objective it bounds. Measured before the fix:
    bound 33,765.5 against an objective of 34,008.8.

    Stopping early still leaves a usable incumbent: the last `(p, published_w)`
    pair is on the result at every point, so a round cap or a non-certificate
    answer is usable rather than lost work.

    Never raises for solver trouble — the fallback publishes the warm
    prices and says so.

    `cfg` is the injected `agent/config.Config`: every cap this solve runs under
    (the round cap, the damping, the depth blocks, the reduced-cost tolerances,
    the labour clamp) comes from it, and a caller with no config gets the shipped
    defaults — one definition, one place.
    """
    # The horizon is the season that is LEFT (F029), never a fixed look-ahead:
    # a day-0 plan starts at 30 days and shrinks by one a day, so the DP's
    # terminal row is the season's own end. The contractor is what carries it
    # (`priced_contractor` shrinks a contractor that overshoots the season; a
    # shorter one is the caller's own look-ahead and is priced as it stands),
    # and everything below reads the same `days`: the LP's rows, the columns,
    # the bound.
    contractor = priced_contractor(contractor, obs)
    cfg = Config() if cfg is None else cfg
    iter_cap = int(cfg.iter_cap if iter_cap is None else iter_cap)
    days = int(contractor.days)
    p_mkt_full, w_stand_full = dual_stand_in(obs, cfg=cfg)
    p = p_mkt_full[:days]
    # #15: the product rows of `p` come from the market forecast (F035's
    # rising path); the flat stand-in is the documented fallback.
    rival_curve = None
    if int(getattr(cfg, "price_supply_rounds", 0)) > 0:
        try:
            from agent.belief.rival_calendar import supply_curve as _rival_curve
            rival_curve = np.asarray(_rival_curve(obs, tuple(PRODUCTS), days),
                                     dtype=np.float64)
        except Exception:                     # noqa: BLE001 - no rival to read
            rival_curve = None
    p, p_source = _product_price_path(obs, days, p, forecast_obj=forecast_obj,
                                      rival_supply=rival_curve, cfg=cfg)
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

    # ---- the column-generation loop -------------------------------------
    # One subproblem per CLASS, not per tile: tiles in the same graph state
    # have the same answer, so pricing both prices twice. `colgen` accumulates
    # the columns across rounds and stops on the reduced-cost certificate.
    from agent.planner.columns import shed_distance
    steps = shed_distance()
    reps, counts, of_tile = colgen.classes_of(owned, _owned_distances(obs, steps))
    result.classes = (reps, counts, of_tile)
    idle = [colgen.Column(cls=c, cost=np.zeros((days, N_COUPLING)),
                          spend=np.zeros(days), earn=np.zeros(days),
                          revenue=0.0, cls_key=tuple(reps[c]), key=("idle",))
            for c in range(len(reps))]

    w_cur = w_lag
    #: The produce credit the pricing closure handed the tiles on its LAST call:
    #: the guard reads it directly instead of inferring it from a certificate.
    credit_box: list = [None]
    state = {"w": w_lag, "cash": np.zeros(days), "failed": None}

    def price(duals):
        y, cash, shed = duals.y, duals.cash, duals.shed
        eta = duals.eta

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
        # The cash rows are CUMULATIVE, so a coin spent on day d sits in every
        # row from d onward and a coin earned on day d relieves every row after
        # it. `ahead[d]` is the shadow price of spending then, `later[d]` of
        # earning then — and matching the row is not optional: the reduced-cost
        # test is only a reduced cost of the LP whose duals it used.
        cash_arr = np.asarray(cash, dtype=np.float64)
        ahead = np.cumsum(cash_arr[::-1])[::-1]
        later = np.concatenate([ahead[1:], [0.0]])
        p_eff = p.copy()
        if shed is None:
            p_eff[:days, list(MARKET_IDS)] *= (1.0 + later[:days])[:, None]
        else:
            # With a shed the produce is worth what a unit IN THE SHED is worth,
            # not what the market pays on the day it is harvested — the balance
            # row's dual. That dual already carries the timing: it is the price
            # of the same LP whose cash rows the `later` term was standing in
            # for, and it carries the cap as well, which no scalar could.
            # σ arrives already clamped onto the non-negative orthant by the
            # caller: the tile graph is dominance-pruned and that pruning is
            # optimality-preserving only while every price is >= 0 (R006). A good
            # worth less than nothing is worth nothing to a plan that can only
            # choose to produce it. The clamp is applied ONCE, where the duals
            # are handed to the pricing, so the bound is computed at the same
            # multipliers the pricing used — clamping here alone would price the
            # tiles at one σ and bound them at another, and the bound would come
            # out below the objective it bounds.
            sig = np.asarray(shed[0], dtype=np.float64)[:days]
            # With the entry row the columns appear in the SPLIT rows and not in
            # the balance rows, so the produce's internal price is the split row's
            # dual: crediting sigma would price the tiles at multipliers this LP
            # does not use, the reduced-cost test would stop being about this LP,
            # and the bound identity would break.
            credit = sig
            if entry and eta is not None:
                credit = np.maximum(np.asarray(eta, dtype=np.float64), 0.0)
            credit_box[0] = np.array(credit[:days, :], dtype=np.float64)
            for gi, ii in enumerate(SELLABLE):
                rid = _resource_of(SHED_ITEMS[ii])
                if rid is not None:
                    p_eff[:days, rid] = credit[:days, ii]
        # `exact` is the price a plan PAYS for what it consumes, and with a shed it
        # is the WHOLE of `published_duals`: `quote·(1+ahead)` for a bought input,
        # because the spend is in the objective now (see `MasterLP.solve`). With a
        # slack purse that is the engine's own quote — 25 for wheat, 100 for
        # fertilizer — instead of the 0 the row alone charged (#142).
        #
        # It is deliberately NOT given σ: σ is the price of a unit in the shed, and
        # a plan's produce is credited at σ through `p_eff` above — but what the
        # plan EATS is bought from the market and charged to the cash row at its
        # quote. Passing σ in here made the DP pay `max(quote·(1+ahead), σ) − quote`
        # for feed and doses while the LP charged the cash row alone, so the
        # subproblem over-charged, the class value came back below the LP's own
        # value of a column the pool already held, and the bound came out below the
        # objective it bounds (measured on a real board: rc −12.93 / −62.10 /
        # −162.38 on the three working classes).
        exact = published_duals(y, days, ahead, supply.quotes)
        if shed is None:
            # The no-shed model credits a column its own revenue and charges the
            # spend through the cash row alone, so the coin it spends is worth
            # `quote·ahead`: the base quote stays in the row.
            #
            # The cash row prices SPENDING at `ahead` — not at the quote — so a
            # purchasable input reaches the tiles at `quote·ahead`. Charging
            # `quote·(1+ahead)` here made the DP optimise a different objective than
            # the master prices, so the class value underestimated its true
            # Lagrangian value and the bound came out below the objective
            # (measured: 35,026.9 against 35,048.8).
            for i, rid in enumerate(PURCHASE_IDS):
                exact[:, rid] = np.maximum(exact[:, rid] - float(supply.quotes[i]),
                                           0.0)
        state["w"] = np.maximum(
            np.maximum((1.0 - cfg.alpha) * state["w"] + cfg.alpha * y, 0.0),
            w_floor)
        state["cash"] = ((1.0 - cfg.alpha) * state["cash"]
                         + cfg.alpha * np.asarray(cash))
        # One board per distinct DISTANCE, not per state: the walk is charged on
        # the labour column of every worked day, so it has to be inside the DP's
        # own objective (`TileContractor._travel_edge_costs`), and a class is
        # (state, distance) — so the classes sharing a distance share a sweep.
        # Classes that differ only in state now run one sweep each, and that is
        # the price of an exact subproblem: with the walk added AFTER the argmax
        # the DP optimises one objective while the master prices another, so the
        # value is not the class's best dual-priced plan, the reduced-cost test
        # is no longer about this LP, and the bound can come out below the
        # objective it bounds (measured: 33,765.5 against 34,008.8).
        groups: dict[int, list[int]] = {}
        for state_id, dist in reps:
            group = groups.setdefault(int(dist), [])
            if int(state_id) not in group:
                group.append(int(state_id))
        boards: dict[int, tuple] = {}
        # One base sweep for every distance in the round, not one sweep each:
        # the distance only changes the walk on the labour column, so the gemv
        # over the graph is shared (`TileContractor.price_many`).
        priced = contractor.price_many(p_eff, exact, groups)
        for dist, group in sorted(groups.items()):
            board_d = priced[int(dist)]
            cost_d = board_d.per_day_cost[:, :days, COUPLING_IDS].astype(np.float64)
            _validate_cost(cost_d)
            produce_d = board_d.per_day_produce[:, :days, list(MARKET_IDS)] \
                .astype(np.float64)
            earn_d = (produce_d * p_mkt[:days][None, :, :]).sum(axis=2)
            spend_d, _ = column_cash(board_d, supply, days)
            boards[dist] = (board_d, {s: i for i, s in enumerate(group)},
                            cost_d, earn_d, spend_d)

        columns, values = [], []
        for c, (state_id, dist) in enumerate(reps):
            board, at, cost, earn, spend = boards[int(dist)]
            i = at[int(state_id)]
            # The walk is already in `cost`: it was charged before the DP chose,
            # so this is the class's best dual-priced plan and not the best plan
            # at prices nobody pays.
            hours = cost[i].copy()
            columns.append(colgen.Column(
                cls=c, cls_key=(state_id, dist),
                cost=hours, spend=spend[i], earn=earn[i],
                revenue=float(earn[i].sum()),
                produce=board.per_day_produce[i, :days, :],
                chains=tuple(board.plans[i]) if i < len(board.plans) else (),
                entities=_entities(board, i, days),
                key=colgen.column_key(board, i, days)))
            # The value is the class's own dual-priced value: revenue, less the
            # coupling duals (labour, the walk included), less the cash the plan
            # ties up. The floor is a belt rather than a correction now: the idle
            # column is worth exactly 0 and every class has it, so the DP cannot
            # return less than nothing.
            cash_use = float((ahead[:days] * spend[i][:days]).sum()
                             - (later[:days] * earn[i][:days]).sum())
            values.append(max(0.0, float(board.tile_values[i])))
        return np.asarray(values, dtype=np.float64), columns

    # The town's appetite AND the drain's spread, from one call: the risk shave
    # the ladder below is read with, and the cap the sell rows use further down.
    sell_cap, risk_pad = _sell_cap(obs, days, float(getattr(cfg, "sell_risk_z", 0.0)))

    # The market's DEPTH, from belief's own ladder (`belief.depth.sell_blocks`):
    # what a lot fetches, per good per day, as blocks an LP can price. Built from
    # the SAME forecast the price path came from, so the curve and the path are
    # one walk and the first block's price IS the day's quote. Any failure leaves
    # `depth = None`, which is the two-tier model that shipped — one degrade, and
    # never a second price.
    depth = None
    if forecast_obj is not None:
        try:
            from agent.belief.depth import day_envelope, sell_blocks
            goods = [SHED_ITEMS[ii] for ii in SELLABLE]
            # The day's ENVELOPE, not its hour-0 row: goods already in the shed
            # can reach any hour of the day, so the day is worth what its best
            # hour pays — and the hourly layer, which owns the hour, can only do
            # better than this number, never worse.
            _env_price, env_hour = day_envelope(forecast_obj, goods,
                                                int(obs.get("day", 0)), days)
            # The ladder is read from a market padded by the drain's own spread,
            # in the caller's good order (the pad is keyed by NAME, so no order
            # can silently misalign). An empty pad is the mean ladder.
            pad = (np.array([float(risk_pad.get(g, 0.0)) for g in goods],
                            dtype=np.float64) if risk_pad else None)
            depth = sell_blocks(forecast_obj, goods, int(obs.get("day", 0)),
                                days, int(supply.shed_capacity),
                                blocks=int(cfg.sell_blocks), hours=env_hour,
                                pad=pad)
        except Exception:                       # noqa: BLE001 - the flat tier stands
            depth = None

    try:
        # The warm pool is priced at TODAY's product prices before it is used:
        # a column's revenue was computed on the board it was built on, and the
        # market path moves (see `_repriced_pool`).
        cg = colgen.generate(price, supply.hours, supply.money, counts, days,
                             N_COUPLING, idle, rounds=max(1, iter_cap),
                             cfg=cfg, integral=integral,
                             shed=(supply.shed_stock, supply.shed_capacity),
                             prices=p_mkt,
                             market=SELLABLE,
                             sell_cap=sell_cap,
                             depth=depth,
                             entry=entry,
                             warm=_repriced_pool(pool, p_mkt, days),
                             smoothing=smoothing,
            buy_hands=buy_hands, hand_mult=hand_mult)
    except RuntimeError as exc:
        return _fallback(str(exc)[:200])
    except Exception as exc:                    # noqa: BLE001 - degraded, not dead
        return _fallback(f"pricing failed: {type(exc).__name__}: {exc}")

    w_cur = state["w"]
    result.cash_duals = state["cash"]
    result.rounds = cg.rounds
    result.bound = cg.bound
    result.gap = cg.gap
    result.pool = cg.pool
    result.certified = cg.certified
    result.stopped = cg.stopped
    result.history = list(cg.rc_history)
    if cg.solve is not None:
        # Two sides of one matrix. The DECISION (what to do) is the solve's own;
        # the PRICES (what a tile, a coin and a good are worth) can only come from
        # an LP, because a MIP has no marginals — so when the decision solve was
        # integral, every dual-derived field is read from the LP solve of the same
        # pool and the zero placeholders never leave this function.
        result.integral = bool(getattr(cg.solve, "integral", False))
        result.lp_final = cg.lp_final if result.integral else None
        dual_src = cg.lp_final if result.integral else cg.solve
        result.lam = cg.solve.lam
        result.objective = cg.solve.objective
        result.duals = state["w"]
        result.mu = dual_src.mu
        result.sigma = dual_src.sigma
        result.cash_lp = dual_src.cash
        result.now = getattr(dual_src, "now", None)
        result.defer = getattr(dual_src, "defer", None)
        result.eta = getattr(dual_src, "eta", None)
        result.credit = credit_box[0]
        result.defer_cap = getattr(dual_src, "defer_cap", None)
        result.sells = getattr(cg.solve, "sells", None)
        result.hands_bought = getattr(cg.solve, "hands_bought", None)
    converged = cg.certified

    # #87's dead-zone clamp, on the COUPLING dual before the publish map:
    # a published labour wage past the DP's dead edge (~147 on this graph)
    # cannot move any tile, and the season table showed the tâtonnement
    # reaching 947-2131 there. Clamping degrades to the busiest legal wage
    # and records how often the raw loop wanted past it. The edge is
    # `Config.labour_dead_edge`.
    lab_ix = COUPLING_IDS.index(LABOR_ID)
    edge = float(cfg.labour_dead_edge)
    clamped = int(np.sum(w_cur[:, lab_ix] > edge))
    w_cur = w_cur.copy()
    w_cur[:, lab_ix] = np.minimum(w_cur[:, lab_ix], edge)
    result.labour_clamped_cells = clamped

    result.converged = converged
    result.w = published_duals(w_cur, days, result.cash_duals, supply.quotes)
    return result


def to_mixes(result: "MasterResult", days: int) -> dict[int, "object"]:
    """`MasterResult` -> the `ClassMix` per class that `columns.py` rounds.

    Keyed by CLASS INDEX. A packed tile key stopped identifying a class the
    moment the distance to the shed joined the key: a board has many tiles of one
    state at many distances, and they are different classes because they cost
    different hours to reach. `result.classes[2]` is the class of every owned
    tile, in board order, which is what `assign_by_quota` consumes.

    Only `labour` and `cash_out` carry real numbers. `wheat_net`, `fert_net`
    and `stored` are zeros and say so here rather than in a surprise: the first
    two stopped being quantity rows when they turned out to be purchasable, and
    `stored` has never existed because a column carries no SELL decision.
    `columns.violations` skips a row it has no capacity for, so a zero row is
    inert rather than a lie the repair acts on.

    The VALUE of a plan is what the master's own objective credits its output
    with, and with a shed that is σ — the balance row's dual — and not the
    market quote. The produce feeds the stock and the stock is sold, so a unit
    produced on a cheap day and sold on a dear one is worth the dear price, and
    a plan valued at its production day's quote is valued below what the LP
    actually earns from it. Measured on the seeded day-0 board: the mix's own
    plans sum to 34,538 at the quotes against an LP objective of 39,942, and the
    #13 gap built on those numbers read 15 % for a rounding that had lost 2 %.
    """
    from agent.planner.columns import DAYS, ClassMix, Plan

    zero_row = (0.0,) * DAYS

    def pad(row: np.ndarray) -> tuple[float, ...]:
        r = tuple(float(v) for v in np.asarray(row, dtype=np.float64)[:DAYS])
        if len(r) < DAYS:
            r += (0.0,) * (DAYS - len(r))
        return r

    sigma = np.asarray(result.sigma if result.sigma is not None else [],
                       dtype=np.float64)
    # A degenerate LP has NO internal price: with every balance row slack HiGHS
    # hands back σ = 0, and valuing every plan at zero would leave the day layer
    # nothing to choose between them (ties fall to the lowest plan index, which
    # is the idle column). The pricing step prices at the market path in exactly
    # that case (`colgen._seed_sigma`), so the value falls back to the revenue
    # that path produces — the plan's own `revenue` at the day it is produced.
    if sigma.ndim != 2 or not sigma.any():
        sigma = np.zeros((0, 0), dtype=np.float64)
    rids = [_resource_of(item) for item in SHED_ITEMS]

    # The prices the PRICING charged a plan's spend: `quote·(1 + ahead)`, with
    # `ahead[d]` the cumulative cash dual from d on. `plan_value` needs them to
    # keep the #13 gap in the objective's accounting: with the spend in the
    # objective (#142), a value that counted only the output read 39,806 against a
    # 35,081 objective — the rounding looked 13 % BETTER than the LP, which is not
    # a gap but a metric, and it would have hidden a real regression behind a
    # `gap <= 0.03` that a negative number satisfies.
    cash_lp = np.asarray(result.cash_lp if result.cash_lp is not None else [],
                         dtype=np.float64)[:days]
    ahead = np.zeros(days, dtype=np.float64)
    if cash_lp.size:
        ahead[:cash_lp.size] = np.cumsum(cash_lp[::-1])[::-1]

    def plan_value(col) -> float:
        """The column's contribution to the objective, in the master's prices.

        Its output is credited at σ, and its spend is CHARGED at what the pricing
        charged the tiles — `quote·(1 + ahead)` — because with a shed the objective
        is `money + Σ p·sell − Σ spend` (#142).
        """
        produce = getattr(col, "produce", None)
        if produce is None or sigma.ndim != 2 or not sigma.size:
            return float(col.revenue)
        produced = np.asarray(produce, dtype=np.float64)[:days]
        total = 0.0
        for ii, rid in enumerate(rids):
            if (rid is None or ii >= sigma.shape[1]
                    or rid >= produced.shape[1]):
                continue
            total += float((sigma[:produced.shape[0], ii]
                            * produced[:, rid]).sum())
        spend = np.asarray(col.spend, dtype=np.float64)[:days]
        if spend.size:
            total -= float((spend * (1.0 + ahead[:spend.size])).sum())
        return total

    lam = np.asarray(result.lam, dtype=np.float64)
    reps, counts, _of_tile = result.classes
    by_class: dict[int, list[tuple[object, float]]] = {c: [] for c in range(len(reps))}
    for j, col in enumerate(result.pool):
        plan = Plan(chains=tuple(int(ch) for _d, _st, ch in col.chains),
                    value=plan_value(col),
                    rows={"labour": pad(col.cost[:, 0]),
                          "cash_out": pad(col.spend),
                          "wheat_net": zero_row,
                          "fert_net": zero_row,
                          "stored": zero_row})
        by_class[col.cls].append((plan, float(lam[j]) if j < lam.size else 0.0))

    mixes: dict[int, ClassMix] = {}
    for c, entries in by_class.items():
        if not entries:
            continue
        mixes[int(c)] = ClassMix(class_key=int(c), count=int(counts[c]),
                                 plans=tuple(e[0] for e in entries),
                                 lam=tuple(e[1] for e in entries))
    return mixes


def _entities(board, tile: int, days: int) -> tuple:
    """What each day's chosen edge constructs on this tile, by name."""
    from agent.tile_dp.chains import entity_of_code
    if board.per_day_entity is None or tile >= board.per_day_entity.shape[0]:
        return ()
    return tuple(entity_of_code(int(code))
                 for code in board.per_day_entity[tile, :days])


def _owned_distances(obs, steps: np.ndarray) -> list[int]:
    """Steps to the nearest shed door for every owned tile, in `owned` order.

    Walks the board exactly as `_owned_states` does, so the two lists line up
    by construction rather than by a comment promising they do.
    """
    from agent.obs import LOCKED_KEY, decode_world
    graph = _shipped_graph()
    view = decode_world(obs, at_day_start=True,
                        graph_keys=frozenset(graph.key_index))
    keys = np.asarray(view.me.keys).reshape(-1)
    return [int(steps[i]) for i, k in enumerate(keys)
            if int(k) != LOCKED_KEY and int(k) in graph.key_index]


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
    # thing that calls it is not a coordinator. `runtime` is still in the
    # signature and is no longer read for anything.
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
