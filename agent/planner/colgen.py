"""Column generation: the master combines MANY plans, and stops on a certificate.

The loop `master.equilibrate` ran was not column generation. Every round it
re-priced the board and **replaced** its column set with the fresh one, so the
master never had more than one plan per class to combine — it could only decide
how many tiles ran that single plan. And it stopped on dual movement, which
`DecisionCraft/notes/phase1_lesson9_dantzig_wolfe.md` §4.3 names exactly:

    "Stopping on 'objective stopped improving' is not column generation — it is
     wishful thinking with extra LPs."

The real loop, from that lesson:

    1. solve the master over EVERY column found so far   -> duals y, mu
    2. price each class at those duals                   -> its best plan
    3. reduced cost rc = (priced value) + mu_c ; rc > 0 means the plan is
       worth having, so ADD it to the pool and go again
    4. no class offers rc > 0  ->  the master's mix is OPTIMAL over the full
       column set, and the subproblems just certified it without enumerating

Step 4 is the point. The answer is not "the best we found before time ran out";
it is "no plan exists that would improve this", proven by the pricing step.

## What a column is

One class's plan for the whole horizon: what it consumes of each coupling row
per day, what it pays the market per day, what it banks per day, and its total
revenue. Classes, not tiles: 25 tiles in one graph state give one subproblem
and one column, and the convexity row `Σ_{j ∈ c} λ_j = N_c` shares that class's
tiles out over its plans. That is where a MIX of plans comes from — 6 tiles on
cows, 19 on wheat is two columns of the same class with λ 6 and 19, and no
scalar price can produce it because both are priced at the same prices.
"""

from __future__ import annotations

import os as _os
from typing import NamedTuple
from dataclasses import dataclass, field, replace

import numpy as np

from agent.config import Config
from agent.world.rules import LAND_PRICES
from agent.world.model import RESOURCE_ID, SHED_ITEMS


def _lost_sale_cost(prices, market, d: int) -> float:
    """What throwing one unit of shed stock away on day `d` costs.

    The owner's rule: the waste is charged the day's AVERAGE SELLING PRICE, so
    the manager is forced to zero it. The market always buys — just cheaper — so
    a discarded unit is a sale that did not happen, and what it lost is the
    day's own average. One number per day, not per item: the engine's night
    flush discards whatever is over the cap without asking which good it is, and
    a charge that is the same for the whole shed cannot make dumping the
    cheap-looking choice for one good and not another. Positive, because this is
    a cost in a min-form LP whose revenues are negated.
    """
    if prices is None or not len(market):
        return 0.0
    px = np.asarray(prices, dtype=np.float64)
    if d >= px.shape[0] or px.shape[1] == 0:
        return 0.0
    return float(px[d, :min(px.shape[1], len(market))].mean())


def _resource_of(item: str) -> int | None:
    """The resource column a shed item occupies, or None if it has none.

    The shed's vocabulary and a plan's are the same goods under two spellings:
    `private["shed"]` is keyed by `SHED_ITEMS` (products + the species names),
    while the columns a plan is priced over call the species `ANIMAL_GOOSE`
    and so on (world/model.py:146-172, kaggriculture.py:171).
    """
    if item in RESOURCE_ID:
        return RESOURCE_ID[item]
    return RESOURCE_ID.get("ANIMAL_" + item)

try:
    from scipy import sparse
    from scipy.optimize import _highspy
    HAS_HIGHS = True
except Exception:                                  # noqa: BLE001
    HAS_HIGHS = False

#: A reduced cost must clear this to be worth a column. Below it the plan is
#: within solver noise of the ones already in the pool, and adding it fattens
#: the master for nothing (lesson 1.9 §4.4: dedupe or the master fattens) — and
#: a reduced cost lives on the OBJECTIVE's scale, so the floor is read as
#: `max(cfg.rc_tol, cfg.rc_rel_tol · |objective|)`: `rc` is a difference of
#: coins, and 1e-6 absolute is below the noise floor of any board worth a few
#: thousand.
#:
#: The pricer's own precision is the binding one. `TileContractor` sweeps in
#: float32 (`DTYPE`), so every class value carries ~1.2e-7 relative error, and
#: the LP's duals come back in float64 from HiGHS. On a real board the loop
#: stalled on `rc 2.24e-4` — 6.3e-9 relative at an objective of 35,772, i.e.
#: inside the pricer's own noise — and the column it wanted was already in the
#: pool: raising the tolerance to 1e-3 certified the SAME objective
#: (35,772.2194) with the same bound (35,772.2202). The two constants are
#: `Config.rc_tol` and `Config.rc_rel_tol`; this function is their only reader.


def rc_tolerance(objective: float, cfg: "Config | None" = None) -> float:
    """The reduced-cost tolerance for a board whose LP objective is this.

    One definition, read by the loop that certifies and by the guards that check
    the certificate: `rc` is a difference of two quantities on the objective's
    scale (`value + mu`, with `mu` the convexity marginal), and the value comes
    out of a float32 sweep, so the cancellation floor is `|objective| · eps` —
    about 4.3e-3 at an objective of 35,772. Anything below that is the pricer's
    own rounding, and refusing to certify on it is refusing to certify at all.
    """
    cfg = Config() if cfg is None else cfg
    return max(float(cfg.rc_tol),
               float(cfg.rc_rel_tol) * abs(float(objective)))


@dataclass(frozen=True)
class Column:
    """One class's plan over the horizon, as the master sees it."""

    cls: int                      # which class this plan belongs to, by INDEX
    cost: np.ndarray              # (days, N_COUPLING) coupling consumption
    spend: np.ndarray             # (days,) coins paid to the market
    earn: np.ndarray              # (days,) coins banked
    revenue: float                # total, at the published product prices
    #: (days, N_RESOURCE) what the plan produces per day, in resource space.
    #: `earn` and `revenue` above are this priced at the product prices of the
    #: board the column was BUILT on, and those prices move (F035: the market
    #: path rises through the season). A pool carried between days therefore
    #: cannot be used as it stands — a column whose revenue is yesterday's makes
    #: the master's LP a hybrid, and its optimum can then exceed the Lagrangian
    #: bound built from today's class values (measured: 203.366 over, which was
    #: exactly the bound's shortfall). Keeping `produce` is what lets the warm
    #: adoption re-price instead of discarding.
    produce: np.ndarray | None = None
    #: The class's KEY — `(graph state, distance)`. The index above is
    #: positional and means nothing on another board: tomorrow's class 3 is not
    #: today's. A warm pool is matched on this and remapped, or a plan for an
    #: empty field arrives as a plan for a grown crop and the master prices a
    #: fiction it will then commit tiles to.
    cls_key: tuple = ()
    chains: tuple = ()            # (day, state, chain_id) — for columns.py
    #: What each day's chosen edge CONSTRUCTS, by name: the crop a PLANT sows,
    #: the animal a PLACE puts down, None for a day that builds nothing. wsr
    #: needs it — a chain of ops without the entity plants nothing in
    #: particular — and the chain id does not carry it.
    entities: tuple = ()
    key: tuple = ()               # dedupe signature


@dataclass
class MasterSolve:
    """One restricted-master solve."""

    lam: np.ndarray               # (n_cols,) weight per column
    y: np.ndarray                 # (days, N_COUPLING) quantity duals ≥ 0
    cash: np.ndarray              # (days,) shadow price of a coin ≥ 0
    mu: np.ndarray                # (n_classes,) convexity duals, FREE in sign
    objective: float
    #: (days, n_items) the BALANCE rows' duals — what one unit of a good sitting
    #: in the shed is worth. FREE in sign (the rows are equalities) and the
    #: internal price the contractor is credited at. Empty without a shed.
    sigma: np.ndarray = None
    #: (days,) the CAP rows' duals ≥ 0 — what one unit of shed room is worth.
    tau: np.ndarray = None
    #: (n_goods,) the APPETITE rows' duals ≥ 0 — what one more unit of the town's
    #: own demand is worth. Read from their own rows, never mixed into `tau`:
    #: the two are relaxed in the bound with different right-hand sides (the
    #: capacity is a per-day number, the appetite a whole-horizon one), so a
    #: dual read off the wrong row prices the wrong constraint.
    rho: np.ndarray = None
    #: (n_goods,) the appetite rows' RIGHT-HAND SIDES as the LP saw them: the
    #: town's cumulative demand per good, in `market` order. Kept beside `rho`
    #: because a bound is `dual · rhs` and the two have to be the same pair.
    appetite: np.ndarray = None
    #: (n_goods, days) what the master decided to SELL, in `market` order.
    sells: np.ndarray = None
    #: (days, items) what the plan drops by the last market hour of the day, and
    #: what waits for the night flush. None when the entry row is off.
    now: np.ndarray = None
    defer: np.ndarray = None
    #: (days, items) the entry rows' duals: the internal price of a harvested
    #: unit, which is the pricing's produce credit when the entry row is on.
    eta: np.ndarray = None
    #: (days, items) the UPPER bound the defer block was given. The last day's
    #: zero is a guard rail against free disposal (a plan that cannot sell what
    #: it harvested would otherwise defer it into the void and dodge the waste
    #: charge), and a rail is only checkable by reading the bound itself: on every
    #: board of sellable goods the LP never WANTS to defer there, so no behaviour
    #: can fail for it.
    defer_cap: np.ndarray = None
    #: True when this solve handed the matrix to HiGHS as a MIP: `lam` is then
    #: INTEGRAL and every dual field is a zero placeholder, not a price (a MIP
    #: has no marginals). The generator keeps pricing off the LP solve of the
    #: same matrix, which is where the real duals come from.
    integral: bool = False
    #: (nq,) the LAND rows' duals, from the LP solve: one number per quadrant, the
    #: value of being allowed one more purchase of it. A ROW dual, not per day --
    #: the row it comes from spans the whole horizon.
    #: binary is worth, read off its own slice. Zero placeholders on the MIP side,
    #: like every other dual there -- a MIP has no marginals.
    land_dual: np.ndarray = None
    #: (days,) the TILE row's dual: what one more tile of room on that day is worth,
    #: i.e. the rent a plan pays for occupying a tile. This is the price the tiles
    #: never saw, which is why land looked free to them.
    rent: np.ndarray = None
    #: (nq, days) the quadrants the DECISION solve bought, one row per entry of
    #: `rules.LAND_ORDER`, 1 in the day column of the purchase — or None when the
    #: land rows were off (the LP path).
    land_bought: np.ndarray = None
    #: (days,) the hands the HANDS block bought, one number per day: fractional
    #: on the LP path, whole on the MIP one. None when the block was off, which
    #: is a different statement from zero. The bill is already inside the
    #: objective, so no caller may add it again; the day layer asks wsr about
    #: the count and hires exactly that.
    hands_bought: np.ndarray = None

    def __post_init__(self) -> None:
        for name in ("sigma", "tau", "rho", "appetite", "sells"):
            if getattr(self, name) is None:
                object.__setattr__(self, name, np.zeros((0,)))


def cash_rows(pool: list[Column], days: int, *,
              with_earn: bool = True) -> np.ndarray:
    """The cumulative cash rows: `(days, len(pool))`, one row per day.

    Column `j`'s coefficient on day `d` is everything it spends through `d` less
    everything it banks before `d` (money is a stock — see `solve_master`). Both
    are running sums, so the whole block is two `cumsum`s over the pool:

    `earn[:d].sum()` is `cumsum(earn)[d-1]`, and the day-at-a-time loop this
    replaces was the single largest block in the season's profile — 18,082,088
    `ndarray.sum` calls, 30.9 s of the season's 136.5 s of `act`.

    Summing a whole column at once (rather than the day-at-a-time loop) left one
    call per column; stacking the pool and running the cumsum along the day axis
    leaves one call per row-block, measured 3.8x-5.3x faster than the per-column
    version at the pool sizes a season reaches (50 to 1250 columns).

    The summation order differs from the loop's, so the rows agree to float64
    rounding rather than bit-for-bit; `tests/test_colgen.py` pins them against
    the loop's own definition.

    `with_earn=False` drops the columns' own `earn` and leaves the spend alone.
    That is what a board WITH a shed needs: there the coins come from the SELL
    variables, a column's `earn` is its produce priced at the board it was built
    on, and counting both makes the purse twice as deep. It also keeps the row
    priceable: the subproblem is priced at `quote·ahead` for what a plan spends
    and credited nothing for what it banks, so an earn term in the row is a row
    the pricing step never priced.
    """
    if not pool:
        return np.zeros((days, 0))
    spend = np.cumsum(np.stack([np.asarray(col.spend, dtype=np.float64)
                                for col in pool]), axis=1)[:, :days]
    if not with_earn:
        return spend.T
    earn = np.cumsum(np.stack([np.asarray(col.earn, dtype=np.float64)
                               for col in pool]), axis=1)[:, :days]
    # `earn[:d].sum()` is `cumsum(earn)[d-1]`: day 0 banks nothing.
    lead = np.zeros((len(pool), 1))
    return spend.T - np.concatenate([lead, earn[:, :days - 1]], axis=1).T


def _extended_basis(basis, n_cols: int, n_rows: int):
    """The previous basis, with the new columns nonbasic at their lower bound.

    A round's pool is the round before's plus the columns the pricer just found,
    in that order (`generate` only ever appends), so the old basis still
    describes the old columns. The new ones enter at 0, their lower bound.
    """
    out = _highspy._core.HighsBasis()
    cols = list(basis.col_status) if basis is not None else []
    rows = list(basis.row_status) if basis is not None else []
    cols += [_highspy._core.HighsBasisStatus.kLower] * max(0, n_cols - len(cols))
    rows += [_highspy._core.HighsBasisStatus.kBasic] * max(0, n_rows - len(rows))
    out.col_status = cols[:n_cols]
    out.row_status = rows[:n_rows]
    out.valid = True
    return out


class MasterLP:
    """The master's LP, kept across a day's rounds so its basis carries.

    max Σ λ·revenue s.t. the coupling rows, the cash rows, one convexity row PER
    CLASS with the class's tile count on the right — not one global row: a global
    row lets a plan of class A absorb the weight class B could not use, which is
    not a plan any tile can run.

    A round solves almost the same LP as the round before. The pool only grows,
    the carried columns keep their order, and the right-hand side moves with the
    duals — so the previous round's basis is a warm start, and HiGHS is where it
    lives. Measured by replaying a real day's round sequence (every round's own
    pool/counts/hours/money captured out of `generate`, then solved three ways):
    a day-0 board, 6 rounds and 9-54 columns — `linprog` 21.3 ms, HiGHS cold
    6.6 ms, HiGHS with the previous basis 3.0 ms, 7.00x; a day-9 board, 40 rounds
    and 15-464 columns — 406 / 240 / 91 ms, 4.48x, and the gap GROWS with the
    pool (at 464 columns: 16.0 / 13.2 / 3.2 ms). The objectives agree to 1.1e-08.

    The solver is scipy's own HiGHS — the one `linprog(method="highs")` wraps —
    so there is no new dependency and no version to drift from, and
    `tests/test_colgen.py` pins the solution AND the three duals against
    `linprog` on the same matrices.
    """

    def __init__(self) -> None:
        if not HAS_HIGHS:
            raise RuntimeError("column generation needs scipy's HiGHS")
        self._highs = _highspy._core._Highs()
        self._highs.setOptionValue("output_flag", False)
        self._basis = None
        #: The row LAYOUT the stored basis was built for. A warm start is only a
        #: warm start if the rows are in the same order: the old basis's row
        #: statuses are positions, so handing them to a model that inserted the
        #: balance and cap rows before the convexity rows puts every convexity
        #: status on a balance row. That is not a worse start, it is a WRONG one
        #: (HiGHS took it and the loop stopped certifying), so the basis is kept
        #: only while the layout matches.
        self._layout = None

    def solve(self, pool: list[Column], counts: np.ndarray, hours: np.ndarray,
              money: float, days: int, n_coupling: int,
              shed_stock: np.ndarray | None = None,
              shed_capacity: float = 0.0,
              prices: np.ndarray | None = None,
              market: tuple[int, ...] = (),
              sell_cap: np.ndarray | None = None,
              depth: tuple[np.ndarray, np.ndarray] | None = None,
              entry: bool = False,
              integral: bool = False,
              land: int | None = None,
              cfg: "Config | None" = None,
              buy_hands: bool = False,
              hand_mult: int = 0) -> MasterSolve:
        """The restricted master over the pool, with the SHED as a stock.

        Variables: `lambda_j >= 0` per column, then per day the sells, the stock
        and the waste — `sell_shallow[g,d]`, `sell_deep[g,d]`, `stock[i,d]`,
        `waste[i,d]` over the shed's own items (`PRODUCTS + ANIMALS`, the
        engine's own order). Rows: the coupling rows (labour), the cumulative
        cash rows, the BALANCE rows

            stock[i,d+1] − stock[i,d] − Σ_j λ_j·produce_j[d,i]
            + sell_shallow[i,d] + sell_deep[i,d] + waste[i,d] = 0

        with `stock[i,0]` FIXED at the opening shed, the cap rows
        `Σ_i stock[i,d] <= capacity`, the APPETITE rows
        `Σ_d sell_shallow[g,d] <= appetite[g]`, and the convexity rows. The
        objective is `Σ (p·sell_shallow + φ·p·sell_deep)`: a column earns nothing
        directly — what it produces feeds the stock, and the stock's dual σ is
        the internal price of a good (`published_duals` reads it).

        This is the point of the change. The master used to credit each day's
        production at THAT day's market price, so holding had no value, a day of
        delay was free, the contractor tied between working and waiting, and the
        tie-break (lowest edge index = the idle chain) made the farm do nothing
        all season. `waste` is what keeps the cap row feasible: the engine
        discards the night flush's overflow, so the LP must be able to as well,
        and it is CHARGED the sale it lost or dumping is the LP's cheapest way
        out of every constraint.

        `land=k` adds the first `k` of the engine's quadrants as BUYABLE
        (`world.rules.LAND_PRICES`, prefix order): one binary per (quadrant, day)
        for the day it is bought, `<= 1` per quadrant, the purchase price charged
        on that day inside the cumulative cash rows (that IS the saving decision:
        the purse must cover it from that day onward), and one row that ties the
        total tile count to `25 x (1 + purchases)`. Off by default so the LP path
        is the matrix that shipped.

        `integral=True` hands the SAME matrix to HiGHS as a MIP: the `lambda`
        columns become integer (a column is a tile COUNT, and half a tile is not a
        plan), and nothing else changes. A MIP has no duals — its marginals are not
        shadow prices — so the PRICING loop must keep reading them from the LP solve
        of the same matrix (`integral=False`); this flag is for the DECISION solve,
        never for the generator.

        The sells are TWO TIERS because the town's appetite is finite and the
        market past it is not: a sale that is not sold into the town's own demand
        is sold at `Config.sell_deep_factor` of the price, which is what makes the
        revenue concave in the quantity sold — and what keeps σ, the value of a
        unit in the shed, from being the best price on the path.
        """
        from agent.planner.hands import MAX_HANDS

        names = SHED_ITEMS
        cfg = Config() if cfg is None else cfg
        items = len(names) if shed_stock is not None else 0
        rids = [_resource_of(n) for n in names]
        n_goods = len(market)
        n = len(pool)
        n_classes = int(counts.size)
        # The sells are K blocks per good per day when the caller hands in the
        # market's own curve (`belief.depth.sell_blocks`), and the legacy two
        # tiers when it does not. `tiers` is the ONLY thing that differs: the
        # layout, the objective, the bounds, the balance rows, the cash rows and
        # the read-back are all written per tier below, so the feature-absent
        # path is the model that was here before, unchanged.
        tiers = 2 if depth is None else int(np.asarray(depth[0]).shape[2])
        revenue = np.array([c.revenue for c in pool], dtype=np.float64)
        # quantity rows: (n_coupling·days, n)
        A_q = np.stack([c.cost.T.reshape(-1) for c in pool], axis=1) if n else \
            np.zeros((n_coupling * days, 0))
        # LABOUR is the only coupling row today (`COUPLING_IDS = (LABOR_ID,)`):
        # one row per day, `Σ_j λ_j·hours[j,d] ≤ budget[d]`. With `buy_hands` the
        # budget is not GIVEN any more — the row keeps only the FARMER's own 24
        # hours, scaled by the same safety margin `hours_for` applies, and the
        # day's hands become columns that buy 23 hours each (a hand hired in
        # turn 0 acts from hour 1, F040).
        margin = 1.0 - float(cfg.hours_overhead)
        b_q = (np.full(n_coupling * days, 24.0 * margin) if buy_hands
               else np.tile(hours[:days], n_coupling))

        # Cash rows, CUMULATIVE: everything spent up to and including day d, less
        # everything banked before it, against one purse.
        #
        #     Σ_{d' ≤ d} spend[d']  −  Σ_{d' < d} earn[d']  ≤  money
        #
        # Per-day rows were the bug that broke a season. `spend[d] ≤ money` on
        # every day independently says the farm may spend its whole purse on day
        # 0, and again on day 1, and again on day 2 — twenty times over. It
        # committed to a plan that did exactly that, went broke on day 1, and
        # scored 4,268 where doing nothing scores 3,000. Money is a STOCK; a row
        # that treats it as an allowance per day is not a budget.
        #
        # The earnings come from the SELLS now, not from the columns' own `earn`:
        # the same coins counted twice would make the purse look twice as deep.
        # With a shed that is not a preference but a requirement: the tiles are
        # priced at `quote·ahead` for what they spend and credited nothing for
        # what they make, so a cash row that also banks the column's own produce
        # revenue is a row the subproblem never priced — and a bound built on a
        # subproblem that priced a different LP is not a bound.
        A_c = cash_rows(pool, days, with_earn=not items)
        b_c = np.full(days, float(money))

        # convexity: one row per class
        A_e = np.zeros((n_classes, n))
        for j, col in enumerate(pool):
            A_e[col.cls, j] = 1.0

        # --- the shed: sells, stock, waste, and their rows --------------------
        # Layout: [lambda (n) | sell_shallow (n_goods·days) | sell_deep
        #          (n_goods·days) | stock (items·days) | waste (items·days)]
        #
        # The sells come in TWO tiers per good per day: the town's own appetite at
        # the day's price, and everything beyond it at a fraction of it. A hard
        # cap would forbid the rest of the harvest; the town really does buy more
        # than it wants, just cheaper. The revenue is CONCAVE in the total sold
        # (p > φp), so the LP fills the shallow tier first on its own and the
        # two-tier revenue is exact without an iteration. The engine's own curve
        # is an integer staircase (`kaggriculture.py::market_price`), and more
        # tiers are the refinement of this, not a different model.
        half = n_goods * days
        block = tiers * half
        stock0 = n + block
        waste0 = stock0 + items * days
        now0 = waste0 + items * days
        defer0 = now0 + items * days
        land0 = defer0 + items * days if entry else now0
        nq = 0 if not land else min(int(land), len(LAND_PRICES))
        land_width = nq * days
        # Every optional block's width is declared HERE, before any row or cost
        # vector is allocated. A block that APPENDS to them leaves the vectors
        # longer than `num_col_`; a block that grows `n_cols` after the rows
        # exist leaves them short of it. Both were live defects.
        hands_width = MAX_HANDS * days if buy_hands else 0
        hands0 = land0 + land_width
        n_cols = hands0 + hands_width

        def col_now(ii: int, d: int) -> int:
            return now0 + ii * days + d

        def col_defer(ii: int, d: int) -> int:
            return defer0 + ii * days + d

        def col_land(q: int, d: int) -> int:
            """The day quadrant `q` (0-based into LAND_ORDER) is bought."""
            return land0 + q * days + d

        def col_sell(gi: int, d: int, b: int = 0) -> int:
            return n + b * half + gi * days + d

        def col_sell_deep(gi: int, d: int) -> int:
            """Tier 1 of the legacy two-tier model (the depth path never calls it)."""
            return col_sell(gi, d, 1)

        # The per-block price and size: the objective, the bounds, the cash rows
        # and the balance rows all read THESE, built once. With a curve they are
        # the ladder's own blocks; without one they are the day's quote and
        # `Config.sell_deep_factor` of it, which is the model that shipped.
        block_price = np.zeros((n_goods, days, tiers), dtype=np.float64)
        block_units = np.full((n_goods, days, tiers), np.inf, dtype=np.float64)
        if items and n_goods:
            px = np.asarray(prices, dtype=np.float64)
            for b in range(tiers):
                if depth is None:
                    factor = 1.0 if b == 0 else float(cfg.sell_deep_factor)
                    block_price[:, :, b] = factor * px[:days, :n_goods].T
                else:
                    block_price[:, :, b] = np.asarray(
                        depth[1], dtype=np.float64)[:, :days, b]
                    block_units[:, :, b] = np.asarray(
                        depth[0], dtype=np.float64)[:, :days, b]

        def col_stock(ii: int, d: int) -> int:
            return stock0 + ii * days + d

        def col_waste(ii: int, d: int) -> int:
            return waste0 + ii * days + d

        cost = np.zeros(n_cols)
        if not items:
            # No shed: the model is the one that was here before, where a column
            # earns its own revenue on the day it produces. WITH a shed a column
            # earns nothing directly — its produce feeds the stock and the stock's
            # dual prices it — so leaving the revenue in would count the same
            # coins twice, once at production and once at the sale.
            cost[:n] = -revenue
        else:
            # THE SPEND IS IN THE OBJECTIVE, not only in the cash row. The score is
            # the final money, so a coin spent is a coin lost: this LP maximises
            # `money + Σ p·sell − Σ spend` (money is a constant), and the cash rows
            # keep only the TIMING — the purse must cover the spend on the day it
            # happens. Priced through the cash row ALONE, a slack purse made every
            # bought input free to the tiles: the DP's price for one is
            # `quote · ahead`, and `ahead` is the cumulative cash dual, which is 0
            # whenever the purse does not bind. Measured on the one-tile reduced
            # process: wheat 25/29/34 and fertilizer 100 on days 0/2/8 with the DP
            # charged 0.0000 for both, on all 30 days — while the engine charged
            # the quotes for real (#142).
            for j, col in enumerate(pool):
                cost[j] = float(np.asarray(col.spend,
                                           dtype=np.float64)[:days].sum())
        lower = np.zeros(n_cols)
        upper = np.full(n_cols, np.inf)
        if entry:
            # No day after the season: what waits on the last day is destroyed by
            # the night flush, so the LP may not leave anything there.
            for ii in range(items):
                upper[col_defer(ii, days - 1)] = 0.0
        # The appetite rows: ONE cumulative row per good, `Σ_d sell[g,d] ≤
        # appetite[g]`, never a per-day cap. The town eats so much over the
        # horizon and our sales may land on any day of it — belief's
        # `drain_forecast` is already that total, and a per-day split was both a
        # wrong scale and a decision the model does not have to make. Only the
        # SHALLOW tier counts against it; what is sold past the appetite goes
        # through the deep tier at φ·p.
        appetite_row = np.zeros((n_goods, n_cols))
        appetite_rhs = np.zeros(n_goods)
        if items:
            for gi, ii in enumerate(market):
                for d in range(days):
                    # `prices` is the (days, len(market)) market path, so the
                    # good's own index is `gi`, NOT the shed-item index `ii`.
                    for b in range(tiers):
                        j = col_sell(gi, d, b)
                        cost[j] = -block_price[gi, d, b]          # max Σ p·sell
                        if depth is not None:
                            # A block takes what the ladder's own step can hold;
                            # past the last block the market is out of depth, so
                            # the LP cannot sell it at any price.
                            upper[j] = block_units[gi, d, b]
            # `stock[i, 0]` is the opening shed: fixed, not a decision.
            for ii in range(items):
                j = col_stock(ii, 0)
                lower[j] = upper[j] = float(shed_stock[ii])
                for d in range(days):
                    # The waste is charged the sale it lost, or the LP dumps
                    # everything and the cap row never binds. The engine's night
                    # flush really does discard the overflow, so the variable
                    # cannot be deleted — it is priced instead.
                    cost[col_waste(ii, d)] = _lost_sale_cost(prices, market, d)
            if sell_cap is not None and n_goods and depth is None:
                # The appetite row is a PROXY for the town's demand and the
                # depth curve replaces it: the ladder already says what every
                # additional unit fetches, so a second cap on the cheap tier
                # double-counts the drain (#151). The row stays in the layout —
                # the dual offsets are positional — with zero coefficients and a
                # zero right-hand side, which constrains nothing and keeps
                # `rho·appetite` at zero in the bound.
                cap_arr = np.atleast_2d(np.asarray(sell_cap, dtype=np.float64))
                width = min(n_goods, cap_arr.shape[1])
                for gi in range(width):
                    appetite_rhs[gi] = max(0.0, float(cap_arr[:, gi].sum()))
                    for d in range(days):
                        appetite_row[gi, col_sell(gi, d)] = 1.0

        # balance rows: one per (item, day)

        # balance rows: one per (item, day)
        bal = np.zeros((items * days, n_cols))
        for ii in range(items):
            sellable = market.index(ii) if ii in market else -1
            for d in range(days):
                row = ii * days + d
                bal[row, col_stock(ii, d)] = -1.0
                if d + 1 < days:
                    bal[row, col_stock(ii, d + 1)] = 1.0
                # What is left on the last day has nowhere to go: the season ends
                # there and the goods are worth nothing after it, so it leaves
                # through the waste variable rather than by magic.
                bal[row, col_waste(ii, d)] = 1.0
                if sellable >= 0:
                    for b in range(tiers):
                        bal[row, col_sell(sellable, d, b)] = 1.0
        prod = np.zeros((items * days, n))
        if items and n:
            valid_rids = [(ii, rid) for ii, rid in enumerate(rids) if rid is not None]
            for j, col in enumerate(pool):
                if col.produce is None:
                    continue
                produced = np.asarray(col.produce, dtype=np.float64)
                d_max = min(days, produced.shape[0])
                if d_max <= 0:
                    continue
                for ii, rid in valid_rids:
                    if rid < produced.shape[1]:
                        prod[ii * days: ii * days + d_max, j] = produced[:d_max, rid]
            if entry:
                for ii in range(items):
                    for d in range(days):
                        bal[ii * days + d, col_now(ii, d)] = -1.0
                        if d >= 1:
                            bal[ii * days + d, col_defer(ii, d - 1)] = -1.0
            else:
                bal[:, :n] -= prod

        # cap rows: one per day, Σ_i stock[i, d] <= capacity
        cap = np.zeros((days, n_cols))
        for ii in range(items):
            for d in range(days):
                cap[d, col_stock(ii, d)] = 1.0

        # The SPLIT rows: `now + defer = produce`, an EQUALITY per (item, day).
        # The harvest is a decision the tile DP already took, so none of it may
        # vanish — the manager only chooses which way it reaches the shed. The
        # row's dual is the internal price of a harvested unit, and it is what the
        # pricing credits (`PricingDuals.eta`): with the entry row the columns no
        # longer appear in the balance rows at all.
        split = (np.zeros((items * days, n_cols)) if entry
                 else np.zeros((0, n_cols)))
        if entry:
            for ii in range(items):
                for d in range(days):
                    split[ii * days + d, col_now(ii, d)] = 1.0
                    split[ii * days + d, col_defer(ii, d)] = 1.0
            split[:, :n] -= prod

        # The cash rows earn from the sells: a sale on day d' is money in the
        # purse from d' onward. Same convention as the columns' own spend — the
        # coin is available the day AFTER it is earned, which is what the
        # original cumulative row did and what the engine does.
        if items and n_goods:
            A_c = np.hstack([A_c, np.zeros((days, n_cols - n))])
            px = np.asarray(prices, dtype=np.float64)
            for gi, ii in enumerate(market):
                for d in range(days):
                    # Same index rule as the objective: the market path is by
                    # GOOD (`gi`), not by shed item (`ii`).
                    for b in range(tiers):
                        A_c[d:, col_sell(gi, d, b)] = -block_price[gi, d, b]

        L_rows = np.zeros((0, n_cols))
        if nq:
            # One binary per (quadrant, day). The payment is a cumulative row's
            # coefficient: `price` in every row at or after the purchase day.
            if A_c.shape[1] < n_cols:
                # No shed: `cash_rows` is the pool alone, and the land columns
                # are to its right.
                A_c = np.hstack([A_c, np.zeros((days, n_cols - A_c.shape[1]))])
            cols = np.arange(land_width)
            d_of = cols % days
            price_of = np.asarray(LAND_PRICES[:nq], dtype=np.float64)[cols // days]
            pay = np.where(np.arange(days)[:, None] >= d_of[None, :],
                           price_of[None, :], 0.0)
            A_c[:, land0:land0 + land_width] += pay
            # Rows: `sum_d y_q <= 1` (one purchase each), the prefix order the
            # engine enforces (`sum y_q <= sum y_{q-1}`), and the tile tie.
            # ONE cap row per DAY: a quadrant bought on day k brings its 25 tiles
            # from day k, never from day 0. A single season-wide row let a purchase
            # on the last day pay for tiles worked on the first, which is why the
            # model could not see what buying EARLY is worth.
            n_land_rows = nq + (nq - 1) + days
            L_rows = np.zeros((n_land_rows, n_cols))
            L_rows[np.arange(nq)[:, None],
                   land0 + np.arange(nq)[:, None] * days
                   + np.arange(days)[None, :]] = 1.0
            if nq > 1:
                qq = np.arange(1, nq)
                # The row index carries [:, None] like the `sum y_q <= 1`
                # block above it: the column index below is (nq-1, days), and
                # a row index of (nq-1,) cannot broadcast against it.
                L_rows[qq[:, None] + nq - 1, land0 + qq[:, None] * days
                       + np.arange(days)[None, :]] = 1.0
                L_rows[qq[:, None] + nq - 1,
                       land0 + (qq[:, None] - 1) * days
                       + np.arange(days)[None, :]] = -1.0
            tile_rows = np.arange(nq + (nq - 1), n_land_rows)
            L_rows[tile_rows, :n] = 1.0
            if nq:
                cols = (land0 + np.arange(nq)[None, :, None] * days
                        + np.arange(days)[None, None, :])
                cum = (np.arange(days)[:, None, None]
                       >= np.arange(days)[None, None, :])
                L_rows[tile_rows[:, None, None],
                       np.broadcast_to(cols, (days, nq, days))] = np.where(cum, -25.0, 0.0)
            upper[land0:land0 + land_width] = 1.0     # one binary per (quadrant, day)
        if buy_hands:
            # --- the day's hands, as columns the model BUYS -------------------
            # One column per (rung, day): the k-th hand of that day. The rungs
            # are the Fibonacci ladder's own INCREMENTS (`world.rules.hire_cost`,
            # one definition of it), and they are convex, so the LP's own
            # relaxation of "how many hands" is integral with no binaries. The
            # bill enters the objective here, so the day layer must stop charging
            # it separately or the same coins are paid twice.
            from agent.world.rules import hire_cost

            # `hire_cost(k)` is the price of the NEXT hire when k are already on
            # the field today, so the ladder indexed at 0 is the FIRST hand's own
            # price, not a constant to difference away: `diff` made rung 0 free
            # and the model hired a hand for nothing.
            inc = np.array([hire_cost(k, hand_mult) for k in range(MAX_HANDS)],
                           dtype=np.float64)
            cost[hands0:hands0 + hands_width] = np.tile(inc, days)
            # The purse must COVER the bill, not only be charged it in the
            # objective: the spend is cumulative (`A_c`'s rows are "everything
            # spent up to and including day d"), so the day-d bill appears in
            # every row from d on — the same shape the land price uses.
            if A_c.shape[1] < n_cols:
                A_c = np.hstack([A_c, np.zeros((days, n_cols - A_c.shape[1]))])
            day_of = np.arange(days)
            at_or_after = np.where(day_of[:, None] >= day_of[None, :], 1.0, 0.0)
            bill = np.transpose(inc[:, None, None] * at_or_after[None, :, :],
                                (1, 0, 2)).reshape(days, MAX_HANDS * days)
            A_c[:, hands0:hands0 + MAX_HANDS * days] = bill

        target = counts.astype(np.float64)
        if n_cols > n:
            # The coupling and convexity rows only involve the columns; the shed
            # variables enter through the balance, the cap, the cash and the
            # appetite rows.
            pad = n_cols - n
            A_q = np.hstack([A_q, np.zeros((A_q.shape[0], pad))])
            A_e = np.hstack([A_e, np.zeros((n_classes, pad))])
        if buy_hands:
            # Row d of the labour block carries that day's rungs: each hand buys
            # 23 hours at the same margin the farmer's own 24 are scaled by, so
            # `Σ_j λ_j·hours[j,d] − 23·margin·Σ_k δ_{k,d} ≤ 24·margin` is exactly
            # the old row with the budget written as the farmer plus what the
            # model chose to buy.
            cols = hands0 + (np.arange(MAX_HANDS)[:, None] * days
                             + np.arange(days)[None, :])
            A_q[np.arange(days)[None, :], cols] = -23.0 * margin
        # Row order, and the duals are read in exactly this order:
        # [labour | cash | balance (EQUALITIES) | cap (≤) | appetite (≤) |
        #  convexity (EQUALITIES)]. The appetite rows sit AFTER the cap rows on
        # purpose: every offset below is positional, so a row inserted in the
        # middle silently re-labels τ as ρ and every price downstream is read off
        # the wrong constraint.
        rows = np.vstack([A_q, A_c, bal, cap, appetite_row, split, L_rows,
                          A_e])
        # The land rows sit BETWEEN the split rows and the convexity rows, and
        # `mu` is read as `marg[n_ineq:]`, so the count has to grow with them: a
        # row appended after the convexity block would be read as a class dual.
        n_ineq = (n_coupling * days + days + items * days + days + n_goods
                  + (items * days if entry else 0) + L_rows.shape[0])
        csc = sparse.csc_matrix(rows)

        lp = _highspy._core.HighsLp()
        lp.num_col_ = n_cols
        lp.num_row_ = int(rows.shape[0])
        lp.col_cost_ = cost
        lp.col_lower_ = lower
        lp.col_upper_ = upper
        lp.row_lower_ = np.concatenate([
            np.full(n_coupling * days + days, -np.inf),   # labour, cash
            np.full(items * days, 0.0),                   # balance: equality
            np.full(days, -np.inf),                       # cap: <=
            np.full(n_goods, -np.inf),                    # town appetite: <=
            np.zeros(items * days if entry else 0),        # split: equality
            np.full(L_rows.shape[0], -np.inf),            # land rows
            target])
        lp.row_upper_ = np.concatenate([
            b_q, b_c,
            np.zeros(items * days),
            np.full(days, float(shed_capacity)),
            appetite_rhs,
            np.zeros(items * days if entry else 0),        # split: equality
            np.concatenate([np.ones(nq),
                            np.zeros(max(0, nq - 1)),
                            np.full(days, 25.0)]) if nq else np.zeros(0),
            target])
        lp.sense_ = _highspy._core.ObjSense.kMinimize
        matrix = _highspy._core.HighsSparseMatrix()
        matrix.format_ = _highspy._core.MatrixFormat.kColwise
        matrix.num_col_ = n_cols
        matrix.num_row_ = lp.num_row_
        matrix.start_ = csc.indptr.astype(np.int32)
        matrix.index_ = csc.indices.astype(np.int32)
        matrix.value_ = csc.data
        lp.a_matrix_ = matrix

        if integral:
            # A column is a tile COUNT: half a tile is not a plan, and the LP's
            # fractional lambda is the reason the committed day was rounded by
            # quota afterwards. The rest of the layout is continuous.
            kinds = [_highspy._core.HighsVarType.kInteger] * n
            kinds += [_highspy._core.HighsVarType.kContinuous] * (n_cols - n)
            if nq:
                kinds[land0:hands0] = [_highspy._core.HighsVarType.kInteger] * land_width
            if buy_hands:
                kinds[hands0:] = [_highspy._core.HighsVarType.kInteger] * hands_width
            lp.integrality_ = kinds

        self._highs.passModel(lp)
        layout = (n_coupling, days, items, n_goods, n_classes)
        # A MIP has no basis to warm from (and passing one is refused): the
        # generator's LP solve is the one that keeps the warm start.
        if not integral and self._basis is not None and self._layout == layout:
            self._highs.setBasis(_extended_basis(self._basis, n_cols,
                                                 lp.num_row_))
        self._highs.run()
        status = self._highs.getModelStatus()
        if status != _highspy._core.HighsModelStatus.kOptimal:
            raise RuntimeError(
                f"master LP failed: {self._highs.modelStatusToString(status)}")
        solution = self._highs.getSolution()
        basis = self._highs.getBasis()
        if basis.valid:
            self._basis = basis
            self._layout = layout

        # HiGHS's ≤-row duals are ≤ 0 in min form; the shadow prices are −them,
        # clamped onto R006's orthant before anything downstream sees them.
        if integral:
            # A MIP returns no meaningful marginals. Scheduled tiles are integral
            # and the prices that priced them came from the LP solve of this same
            # matrix; publishing zeros here says "no prices", never "prices of 0".
            values = np.asarray(solution.col_value, dtype=np.float64)
            bought = (values[land0:].reshape(nq, days) if nq else None)
            sells = np.zeros((n_goods, days), dtype=np.float64)
            for b in range(tiers):
                start = n + b * half
                sells += (values[start:start + half].reshape(n_goods, days)
                          if half else 0.0)
            return MasterSolve(
                lam=values[:n], y=np.zeros((days, n_coupling)),
                cash=np.zeros(days), mu=np.zeros(n_classes),
                objective=-float(self._highs.getObjectiveValue()),
                sigma=np.zeros((days, items)), tau=np.zeros(days),
                rho=np.zeros(n_goods), appetite=appetite_rhs, sells=sells,
                now=(values[now0:defer0].reshape(items, days).T if entry else None),
                defer=(values[defer0:defer0 + items * days].reshape(items, days).T
                       if entry else None),
                eta=None,
                defer_cap=(np.asarray(upper[defer0:defer0 + items * days]
                                      ).reshape(items, days).T if entry else None),
                land_bought=bought, integral=True,
                hands_bought=(values[hands0:hands0 + hands_width]
                              .reshape(MAX_HANDS, days).sum(axis=0)
                              if buy_hands else None))

        marg = np.asarray(solution.row_dual, dtype=np.float64)
        y = np.maximum(-marg[:n_coupling * days], 0.0).reshape(n_coupling, days).T
        off = n_coupling * days
        cash = np.maximum(-marg[off:off + days], 0.0)
        # The balance rows are EQUALITIES: their duals are FREE in sign, because
        # a good in the shed is worth what it can be sold for later — and that is
        # the internal price the contractor is credited at. Negated like every
        # other HiGHS marginal: they are read from a MINIMISATION, and the price
        # this module publishes is the maximisation one. Left unnegated it comes
        # out negative, the orthant clamp in the caller flattens it to zero, and
        # the produce is credited nothing (measured: 180 negative entries, the
        # clamp to 0, every class priced at 0, the loop dead).
        sigma = -np.asarray(marg[off + days:off + days + items * days],
                            dtype=np.float64).reshape(items, days).T
        cap_end = off + days + items * days + days
        # The cap rows' duals and the appetite rows' duals are read from their
        # OWN slices: the cap is a per-day capacity and the appetite a whole
        # horizon's demand, so `τ·capacity` and `ρ·appetite` are different
        # numbers and a single `[:days]`-style slice over both would price one
        # constraint with the other's dual.
        tau = np.maximum(-marg[off + days + items * days:cap_end], 0.0)
        rho_end = cap_end + n_goods
        rho = np.maximum(-marg[cap_end:rho_end], 0.0)
        eta = (-np.asarray(marg[rho_end:rho_end + items * days],
                           dtype=np.float64).reshape(items, days).T
               if entry else None)
        # The convexity duals are EQUALITY marginals and are free in sign: a class
        # whose tiles are worth having carries a negative one. Clamping them would
        # break the reduced-cost test, which is the only reason they are read.
        # The LAND rows sit between the split rows and the convexity block (the
        # ordering note above `n_ineq` is what keeps `mu` off them). Their duals
        # are read from their OWN slice: the first `nq` rows are the one-purchase
        # rows, the next `nq-1` the prefix order, and the LAST row ties the tile
        # count to 25 per quadrant, so it alone carries the rent.
        eta_end = rho_end + (items * days if entry else 0)
        land_rows = int(L_rows.shape[0])
        land_dual = rent = None
        if land_rows:
            _lnd = np.maximum(-np.asarray(
                marg[eta_end:eta_end + land_rows], dtype=np.float64), 0.0)
            # One dual per ROW: the first nq rows are the one-purchase rows, and a
            # row is a number, not a day vector.
            land_dual = (_lnd[:nq] if nq else None)
            rent = (_lnd[land_rows - days:] if (nq and days) else None)
        mu = np.asarray(marg[n_ineq:], dtype=np.float64)
        values = np.asarray(solution.col_value, dtype=np.float64)
        # What the master decided to SELL is the SUM of the tiers: they are two
        # prices for one sale, not two sales, and a caller that read only the
        # first would see a farm that never sells past the town's appetite.
        sells = np.zeros((n_goods, days), dtype=np.float64)
        for b in range(tiers):
            start = n + b * half
            sells += values[start:start + half].reshape(n_goods, days)
        now = (values[now0:defer0].reshape(items, days).T if entry else None)
        defer = (values[defer0:defer0 + items * days].reshape(items, days).T
                 if entry else None)
        return MasterSolve(lam=values[:n], y=y, cash=cash, mu=mu, integral=False,
                           land_dual=land_dual, rent=rent,
                           objective=-float(self._highs.getObjectiveValue()),
                           sigma=sigma, tau=tau, rho=rho,
                           appetite=appetite_rhs, sells=sells,
                           now=now, defer=defer, eta=eta,
                           defer_cap=(np.asarray(
                               upper[defer0:defer0 + items * days]
                           ).reshape(items, days).T if entry else None),
                           hands_bought=(values[hands0:hands0 + hands_width]
                                         .reshape(MAX_HANDS, days).sum(axis=0)
                                         if buy_hands else None))


def solve_master(pool: list[Column], counts: np.ndarray, hours: np.ndarray,
                 money: float, days: int, n_coupling: int) -> MasterSolve:
    """One LP, cold: the master's own path is the `MasterLP` in `generate`."""
    return MasterLP().solve(pool, counts, hours, money, days, n_coupling)


def reduced_costs(values: np.ndarray, mu: np.ndarray) -> np.ndarray:
    """`rc_c` for each class: how much its best plan beats what the master pays.

    The contractor already maximises `p·produce − w·cost` at the published
    duals, so `values[c]` IS the dual-priced plan value of the lesson's
    subproblem. In the master's min form a column improves when
    `−value − mu_c < 0`, i.e. when `value + mu_c > 0` — and `mu` keeps its
    sign for exactly this line.
    """
    return np.asarray(values, dtype=np.float64) + np.asarray(mu, dtype=np.float64)


def cash_relief(cash: np.ndarray, days: int) -> np.ndarray:
    """`later[d]`: the shadow price of a coin EARNED on day d.

    The cash rows are cumulative, so a coin earned on day d relieves every row
    after it — and a SALE is such a coin. The pricing scales the product price
    by `1 + later`, and the bound needs the same number to know what a sale is
    worth. One definition, because the bound and the pricing must agree about it
    or the bound stops bounding.
    """
    arr = np.asarray(cash, dtype=np.float64)
    ahead = np.cumsum(arr[::-1])[::-1]
    return np.concatenate([ahead[1:], [0.0]])[:days]


def _seed_sigma(prices: np.ndarray | None, market: tuple[int, ...],
                days: int) -> np.ndarray:
    """The σ a degenerate LP cannot supply: the market path, by item.

    `prices` is the `(days, N_RESOURCE)` path and `market` the sellable items as
    indices into `SHED_ITEMS`; the resource of each is looked up the same way the
    balance rows do it. Anything not sellable seeds at zero.
    """
    out = np.zeros((days, len(SHED_ITEMS)), dtype=np.float64)
    if prices is None:
        return out
    px = np.asarray(prices, dtype=np.float64)
    for gi, ii in enumerate(market):
        if gi < px.shape[1]:
            out[:, ii] = px[:days, gi]
    return out


def lagrangian_bound(solve: MasterSolve, values: np.ndarray,
                     counts: np.ndarray, hours: np.ndarray, money: float,
                     days: int, n_coupling: int,
                     shed: tuple | None = None,
                     sell: tuple | None = None,
                     blocks: tuple | None = None,
                     market: tuple[int, ...] = ()) -> float:
    """`L(y) = y·b + Σ_c N_c · v_c(y)` — an UPPER bound on the master's optimum.

    Relax the coupling rows with their duals and the problem separates into
    the classes, each free to pick its best plan at those prices. `values[c]`
    IS that best dual-priced value: it is what the pricing step just computed.
    So the bound costs nothing beyond an inner product, and without it column
    generation has no idea how far from done it is — "no column has a positive
    reduced cost" is a certificate at the END and says nothing on the way.

    It is also what makes Wentges smoothing work: the stability centre is the
    dual with the BEST bound so far, not the last one seen.

    Every row that was relaxed contributes `dual · rhs`, and the rows this
    module's LP carries are `y·hours`, `cash·money`, `τ·capacity`, `ρ·appetite`
    and the opening stock at `σ` on day 0 (the stock telescopes over the days,
    so the opening balance is what survives). A term left out does not make the
    bound conservative — it makes it a different number, and one that can come
    out BELOW the objective it bounds.

    Sources: Wentges (1997); Pessoa, Sadykov, Uchoa & Vanderbeck (2018).
    """
    y = np.asarray(solve.y, dtype=np.float64)
    cash = np.asarray(solve.cash, dtype=np.float64)
    rhs = float((y[:days, :n_coupling].sum(axis=1) * hours[:days]).sum()
                + cash[:days].sum() * float(money))
    if shed is not None:
        # The shed's rows enter the bound too, or it stops being a bound. The
        # balance rows are equalities relaxed with the free σ, the cap rows are
        # ≤ rows relaxed with τ ≥ 0, and the stock telescopes over the days: what
        # survives is the OPENING stock at σ on day 0 and the capacity at τ.
        #
        # The sells must be non-positive in the relaxation, or the inner problem
        # is unbounded and this is not a bound at all — and a sale is worth
        # `p·(1 + later)`, not `p`: it relieves every cash row from its day
        # onward. So σ is raised onto that level for the bound. Any multipliers
        # give a valid Lagrangian bound, which is exactly why raising them here
        # is allowed; `sell_floor` is that raised σ, computed by the caller from
        # the SAME `cash_relief` the pricing uses.
        opening, capacity, sigma, tau = shed
        sig = np.asarray(sigma, dtype=np.float64)
        rhs += float((sig[0] * np.asarray(opening, dtype=np.float64)).sum())
        rhs += float((np.asarray(tau, dtype=np.float64)[:days].sum()
                      * float(capacity)))
    if sell is not None:
        # The town's appetite is a row of this LP like any other: a ≤ row
        # relaxed with ρ ≥ 0 contributes `ρ · appetite`, and the pricing is
        # unaffected by it because the sells it caps are not in the subproblem.
        # Leaving it out is not a rounding error — measured on a hand-built
        # board with a binding appetite, it is the whole difference between the
        # bound and the LP's optimum.
        rho, appetite = sell
        rhs += float((np.asarray(rho, dtype=np.float64)
                      * np.asarray(appetite, dtype=np.float64)).sum())
    if blocks is not None:
        # The SELLS' own inner problem. With finite block sizes the relaxation
        # is bounded at ANY sigma, so no raised floor is needed and the term is
        # `sum_b u_b * max(0, p_b - sigma)`: block b is worth selling whenever
        # the ladder pays more than the unit in the shed is carried at. A term
        # left out does not make this conservative, it makes it a different
        # number — measured on the seeded day-0 board, bound 29,161.7 against an
        # objective of 32,083.5 without it, which is not a bound.
        units, block_prices = blocks
        sig_g = np.asarray(sigma, dtype=np.float64)[:, list(market)]
        for b in range(np.asarray(units).shape[2]):
            u_b = np.asarray(units, dtype=np.float64)[:, :days, b]
            p_b = np.asarray(block_prices, dtype=np.float64)[:, :days, b]
            rhs += float((u_b * np.maximum(0.0, p_b - sig_g[:days, :].T)).sum())
    return rhs + float((np.asarray(counts, dtype=np.float64)
                        * np.asarray(values, dtype=np.float64)).sum())


class PricingDuals(NamedTuple):
    """Every dual the pricing step may need, as ONE argument.

    A callback that reads only `y` and `cash` ignores the rest; one that needs
    the shed's prices reads `shed`; the entry row's `eta` is here too. The point
    is that adding a ROW BLOCK adds a field and touches no signature — the
    positional form grew a slot per block and every hand-built callback in the
    guards had to follow it. A NamedTuple, not a dataclass: it is built once per
    pricing round and read field-wise, so the tuple's speed is what we want.
    """

    y: np.ndarray
    cash: np.ndarray
    shed: tuple | None = None
    #: The split rows' duals (the internal price of a harvested unit), or None
    #: when the entry row is off.
    eta: np.ndarray | None = None


@dataclass
class ColgenResult:
    """The mix, the duals, and whether the answer carries a proof."""

    pool: list[Column] = field(default_factory=list)
    #: The LP solve on the FINAL pool, when that is a different solve from
    #: `solve` (i.e. on the integral path). It exists because the two must be
    #: compared on ONE pool: the loop ADDS columns after its last LP solve, so
    #: the solve it exited with was priced on a SMALLER pool than the one a
    #: decision solve sees. A MIP on the final pool against a stale LP is not a
    #: comparison; this is the other half of it.
    lp_final: "MasterSolve | None" = None
    solve: MasterSolve | None = None
    rounds: int = 0
    #: True only when a pricing round found NO class above the reduced-cost
    #: tolerance (`Config.rc_tol`/`rc_rel_tol`, read by `rc_tolerance`). That is
    #: the optimality certificate. False means the round cap stopped the loop,
    #: and the mix is an incumbent, not an optimum.
    certified: bool = False
    stopped: str = ""             # why the loop ended, when it was not certified
    rc_history: list = field(default_factory=list)
    #: How many columns each round ADDED, one entry per round, beside rc_history.
    #: The pool size is cumulative and the rc says what was left on the table; this
    #: says whether a round was still finding anything, which is what a stalled loop
    #: and a converged one look identical without.
    added_history: list = field(default_factory=list)
    #: The best (smallest) Lagrangian bound seen. `inf` before the first
    #: pricing round. The master's objective is a lower bound and this an
    #: upper one, so the two together are the only honest statement of how
    #: close the answer is while the loop is still running.
    bound: float = float("inf")
    #: `(bound − objective) / bound` at the end, or `inf` with no bound yet.
    @property
    def gap(self) -> float:
        if self.solve is None or not np.isfinite(self.bound) or self.bound <= 0:
            return float("inf")
        return (self.bound - self.solve.objective) / abs(self.bound)


def classes_of(owned: list[int], distances: list[int] | None = None
               ) -> tuple[list[tuple[int, int]], np.ndarray, list[int]]:
    """`owned` state ids -> (class keys, counts, tile→class).

    A class used to be a graph state alone. It is `(state, distance to the
    nearest shed door)`, because the contractor prices a state and a WORKER
    walks to a square. The farm is cleared every night and the farmer respawns
    on a shed door (F040), so a tile is reached afresh on every day it is
    worked: a plan that works `v` days on a tile `d` steps out spends at least
    `v·d` hours walking, and that is not a rounding error. Measured on a day-0
    board, 25 tiles: 307 hours of pure travel against a labour budget of 15.6
    hours a DAY.

    With the distance in the key, the tiles of a class are interchangeable
    again — which is the only thing that makes a column honest, because a
    column is what every tile of its class runs.

    `distances` is per owned tile, in the same order. None keeps the old
    state-only classes, which is what a caller with no board means.
    """
    reps: list[tuple[int, int]] = []
    index: dict[tuple[int, int], int] = {}
    of_tile: list[int] = []
    for i, state in enumerate(owned):
        key = (int(state), int(distances[i]) if distances is not None else 0)
        if key not in index:
            index[key] = len(reps)
            reps.append(key)
        of_tile.append(index[key])
    counts = np.zeros(len(reps), dtype=np.int64)
    for c in of_tile:
        counts[c] += 1
    return reps, counts, of_tile


def column_key(board, tile: int, days: int) -> tuple:
    """A plan's signature: the chain it runs on each day, and what it constructs.

    Two plans that run the same chains are the same column however they were
    priced, and the pool must hold one of them — the lesson's fourth pitfall.

    The ENTITY is part of the signature. A chain id names the op (`PLANT`), not
    the crop: an empty tile can plant WHEAT on one round and CARROT on the next
    with the same chain id and the same per-day op sequence, and the two columns
    do not cost the same (a carrot seed is 20, a wheat seed 10). Leaving the
    entity out made them one key, so the loop refused to add the second and then
    reported `stalled: rc ... on a column the pool holds` — a positive reduced
    cost on a column it had never actually priced.
    """
    plan = board.plans[tile] if tile < len(board.plans) else ()
    entity = board.per_day_entity[tile, :days] if board.per_day_entity is not None else ()
    return tuple((int(d), int(chain), int(entity[i]) if i < len(entity) else 0)
                 for i, (d, _state, chain) in enumerate(plan[:days]))


def _shifted_key(key: tuple, step: int) -> tuple:
    """A plan's signature, moved `step` days forward: the day numbers renumber.

    `column_key` is `(day, chain, entity)` per day, so a plan carried one day
    forward is the same tuple with its first entry gone and every day index one
    smaller. A stale key is not harmless in either direction: the pool's
    `seen` set is what stops a plan being added twice, so a key that still
    carries yesterday's day 0 makes the loop refuse a plan the pool no longer
    holds — the `stalled: rc ... on a column the pool holds` failure the key
    exists to prevent.
    """
    return tuple((int(d) - step, chain, entity)
                 for d, chain, entity in key if int(d) >= step)


def advance_pool(pool: list[Column], lam: np.ndarray | None = None,
                 *, step: int = 1) -> list[Column]:
    """Yesterday's pool as today's plan: every day-indexed array moves up a day.

    A column is a plan for days `0..N-1` OF THE DAY IT WAS BUILT ON, and the
    horizon is the season that is left (F029), so the same plan carried into the
    next day is its days `step..N-1` — which is exactly the new horizon. Four
    arrays are day-indexed and all four move: `cost`, `spend`, `earn` and
    `produce`. `chains` and `entities` move with them and that is not cosmetic —
    the day layer commits DAY 0 of the plan it is given (`day.day_chains` reads
    `plan.chains[0]` and `column.entities[0]`), so a column that did not move
    would hand today's board yesterday's chain, on a tile whose state has moved
    on. `revenue` is re-derived from the shifted `earn`, so the two stay the
    pair they were.

    **The columns the LP gave no weight to are dropped** (`lam`, in pool order).
    A plan the master does not use is a plan the next day's pricing can
    rediscover if it is still worth having, and carrying it is what made the
    pool grow monotonically and without bound (measured by a reviewer: 126
    columns on day 0 to 1,054 on day 29). The weight is the last one the pool was
    solved at, so a column beyond `lam`'s length — added after the final solve,
    never priced — is KEPT: there is no evidence against it.

    A column whose plan cannot cover the new horizon is dropped: no `produce` to
    re-price from (an idle column; `master._repriced_pool` drops those anyway, and
    `generate` re-seeds one per class), or fewer days than `step` left in it.
    """
    out: list[Column] = []
    weights = None if lam is None else np.asarray(lam, dtype=np.float64)
    k = max(1, int(step))
    for j, column in enumerate(pool or ()):
        if weights is not None and j < weights.size and float(weights[j]) <= 0.0:
            continue
        if column.produce is None:
            continue
        cost = np.asarray(column.cost)[k:]
        spend = np.asarray(column.spend)[k:]
        earn = np.asarray(column.earn)[k:]
        produce = np.asarray(column.produce)[k:]
        if cost.shape[0] < 1 or produce.shape[0] < 1:
            continue
        out.append(replace(
            column, cost=cost, spend=spend, earn=earn,
            revenue=float(earn.sum()), produce=produce,
            chains=tuple(link for link in column.chains if int(link[0]) >= k),
            entities=tuple(column.entities[k:]),
            key=_shifted_key(column.key, k)))
    return out


def generate(price, supply_hours, money, counts, days, n_coupling,
             idle_columns, *, rounds: int | None = None,
             warm: list | None = None,
             smoothing: float = 0.0, shed: tuple | None = None,
             prices: np.ndarray | None = None,
             market: tuple[int, ...] = (),
             sell_cap: np.ndarray | None = None,
             depth: tuple[np.ndarray, np.ndarray] | None = None,
             entry: bool = False,
             integral: bool = False,
             cfg: "Config | None" = None,
             buy_hands: bool = False,
             hand_mult: int = 0) -> ColgenResult:
    """The loop: master over every column so far, price, add, repeat.

    The loop's ONLY stop besides the certificate is `rounds` (default:
    `cfg.iter_cap`): no clock is read
    between rounds, because a round count that depends on the machine makes two
    runs of one seed disagree (18,312 against 28,890 with every RNG in our code
    seeded and the threads pinned to one).

    `price(y, cash)` is the caller's pricing step: it publishes the duals to
    the contractor and hands back `(values, columns)` — one dual-priced plan
    value per class and the column each would add. Keeping it a callback is
    what lets this module be tested against a hand-built subproblem whose
    optimum is known, which is the only way the sign conventions get checked
    (lesson 1.9 §4.2: a wrong sign stalls the loop at its initial objective
    while looking like convergence).

    `idle_columns` seeds the pool with each class's do-nothing plan. Lesson 1.9
    §4.1, verified live there: a master born only of profit-maximal columns is
    INFEASIBLE and scipy hands back `marginals: None`.
    """

    # `warm` is a pool from an earlier call on almost this instance. The
    # columns are plans, and a plan is still a plan when the prices move — so
    # carrying them means the first master solve already has something to
    # combine instead of only the idle columns, and the pricing step spends its
    # rounds on what is MISSING rather than on rediscovering what is not.
    # Columns for classes this instance does not have are dropped: a class
    # index is an index into THIS board's classes and means nothing in another.
    result = ColgenResult(pool=list(idle_columns))
    cfg = Config() if cfg is None else cfg
    rounds = int(cfg.iter_cap if rounds is None else rounds)

    #: How many columns the pool held when the loop's last solve ran. The loop
    #: ADDS columns after that solve, so on a cap exit the solve it exited with
    #: was priced on a SMALLER pool than the one the decision sees (measured on
    #: the day-0 board: published 68,341.9333 against 96,069.7333 for the same
    #: pool). One more LP solve closes that, and it is only needed when the pool
    #: actually grew.
    n_at_last_solve = [0]

    def decide() -> "ColgenResult":
        """The integral decision solve, on the pool the LP loop built.

        It must run AFTER the loop: a MIP returns no marginals, so a round that
        solved the master integrally prices the next round at zeros — measured,
        the pool stayed at 18 columns instead of 27 and the decision was taken on
        a poorer pool than the LP's own. Same matrix, same columns, one extra
        solve: the LP drives the generation, the MIP takes the decision.
        """
        if result.solve is None:
            return result
        if not integral:
            # The LP path publishes the solve the loop last took, and the loop's
            # own bookkeeping (the pricing's credit, the certificate) is that
            # solve's. Re-solving here on the pool the last PRICING pass built
            # would hand back a plan priced at duals the pricing never used --
            # measured: `test_entry_row`'s identity (the credit the pricing
            # handed the tiles equals what was published) fails. A round the
            # model is worth is asked for in `Config.master_rounds`, where it is
            # visible and costs what it costs; it is not smuggled in here.
            return result
        args = (result.pool, counts, supply_hours, money, days, n_coupling)
        kwargs = dict(
            shed_stock=None if shed is None else shed[0],
            shed_capacity=0.0 if shed is None else float(shed[1]),
            prices=prices, market=market, sell_cap=sell_cap,
            depth=depth, entry=entry, cfg=cfg,
            # The land rows belong to BOTH solves: the decision buys
            # quadrants and the LP beside it publishes what a slot is worth.
            land=int(getattr(cfg, "land_quadrants", 0)) or None,
            buy_hands=buy_hands, hand_mult=hand_mult)
        final = solver.solve(*args, **kwargs)
        if integral:
            # The MIP decides; the LP beside it is what has marginals, and it is
            # the half that says how much the integer answer cost (the gate: a
            # MIP can never beat its own relaxation).
            result.lp_final = final
            result.solve = solver.solve(*args, integral=True, **kwargs)
        else:
            result.solve = final
        return result
    index_of = {c.cls_key: c.cls for c in result.pool if c.cls_key}
    for column in (warm or []):
        target = index_of.get(column.cls_key)
        if target is None or column.cost.shape != (days, n_coupling):
            continue                 # a class this board does not have
        result.pool.append(
            column if column.cls == target
            else Column(cls=target, cost=column.cost, spend=column.spend,
                        earn=column.earn, revenue=column.revenue,
                        chains=column.chains, entities=column.entities,
                        cls_key=column.cls_key, key=column.key,
                        produce=column.produce))
    seen = {(c.cls, c.key) for c in result.pool}

    # --- Wentges dual price smoothing -----------------------------------
    # The pricing step is fed `α·π_best + (1−α)·π_LP`, where π_best is the
    # dual that gave the BEST Lagrangian bound so far — the stability centre —
    # and not the previous iterate. That distinction is the method: the LP's
    # duals jump between extreme points of a degenerate dual polyhedron, and a
    # subproblem chasing them prices plans nobody will use. Smoothing toward
    # the last value (which is what this did) smooths toward whatever noise
    # came last; smoothing toward the incumbent smooths toward the best
    # information the run has.
    #
    # Wentges (1997); Pessoa, Sadykov, Uchoa & Vanderbeck (2018).
    #
    # DEFAULT OFF, and measured rather than assumed. On this problem, with the
    # PRICING AS IT WAS, it did not help and it corrupted the bound:
    #
    #     alpha 0.0   objective 34,197   bound 34,197   gap  0.0 %
    #     alpha 0.3   objective 33,921   bound 33,843   gap -0.2 %
    #     alpha 0.5   objective 33,800   bound 33,750   gap -0.1 %
    #
    # A bound BELOW the objective is not a bound, and two runs certifying at
    # different objectives are two different fixed points. Both symptoms had one
    # cause: the pricing was not an exact Lagrangian subproblem — the travel a
    # plan cannot avoid was added to the column AFTER the DP had chosen its
    # chain, and the purchasable inputs were charged their quote inside the
    # subproblem, which the cash row had not. Smoothing's guarantees assume the
    # subproblem IS the Lagrangian one, so it had nothing to stabilise and only
    # moved where the loop stopped.
    #
    # That is fixed (see `master.equilibrate`'s pricing step and
    # `TileContractor._travel_edge_costs`): the bound is now valid and tight on
    # the real board — bound 35,772.2202 against an objective of 35,772.2194,
    # gap +0.000000. The table above therefore PREDATES the fix and says nothing
    # about smoothing on an exact pricer. The knob stays off until it is
    # re-measured, which is the next thing to try (Wentges 1997; Pessoa,
    # Sadykov, Uchoa & Vanderbeck 2018).
    centre = None                        # (y, cash, mu) at the best bound
    alpha = float(smoothing)

    # One LP object for the whole day: the rounds share a basis (see MasterLP).
    solver = MasterLP()
    for _ in range(max(1, rounds)):
        result.solve = solver.solve(
            result.pool, counts, supply_hours, money, days, n_coupling,
            shed_stock=None if shed is None else shed[0],
            shed_capacity=0.0 if shed is None else float(shed[1]),
            prices=prices, market=market, sell_cap=sell_cap,
            depth=depth, entry=entry, cfg=cfg,
            # The land rows belong to BOTH solves: the decision buys quadrants
            # and the LP beside it prices a slot. A matrix without them can only
            # price a farm that cannot expand, and then the rent is never read.
            land=int(getattr(cfg, "land_quadrants", 0)) or None,
            buy_hands=buy_hands, hand_mult=hand_mult)
        n_at_last_solve[0] = len(result.pool)
        result.rounds += 1  # solves taken; the pricing passes it fed are free

        exact = (result.solve.y, result.solve.cash, result.solve.mu)
        # A degenerate first LP has NO unique dual: with an empty shed and only
        # do-nothing columns the whole model is zero and HiGHS hands back σ = 0,
        # so the pricing credits every plan nothing, no working column is ever
        # generated, and the loop sits at zero for good. Seed σ from the market
        # path in that case — the same job the old `p_eff` did — and use the
        # seeded σ in the bound too, which stays a bound because a Lagrangian
        # bound is valid at ANY multipliers, not only at the optimal ones.
        shed_duals = None
        sell_duals = None
        seeded = False
        if shed is not None:
            sig = np.asarray(result.solve.sigma, dtype=np.float64)
            if result.rounds <= 1 or not np.any(sig):
                # A degenerate LP has NO unique dual — and this one comes out
                # exactly zero even once working columns are in the pool (the
                # balance rows are slack, so a unit of stock carries no shadow
                # price). Pricing at σ = 0 credits every plan nothing, no column
                # is ever generated again, and the loop stalls at zero. Price at
                # the market path instead in that case — the job the old `p_eff`
                # factor did — and mark the round: no bound is computed on it.
                sig = _seed_sigma(prices, market, days)
                seeded = True
            shed_duals = (np.maximum(sig, 0.0), result.solve.tau)
            # The appetite rows' duals and right-hand sides, as ONE pair: a bound
            # is `dual · rhs`, and `solve` publishes both because they are read
            # off rows whose offsets only it knows.
            sell_duals = (np.asarray(result.solve.rho, dtype=np.float64),
                          np.asarray(result.solve.appetite, dtype=np.float64))
        # The reduced-cost tolerance for THIS board: an absolute floor, raised to
        # the pricer's own precision on the objective's scale (`Config.rc_tol`
        # and `Config.rc_rel_tol`, read by `rc_tolerance`).
        tol = rc_tolerance(result.solve.objective, cfg)
        added, rc = 0, np.zeros(0)
        for attempt in range(2):
            # Attempt 0 prices at the smoothed dual; attempt 1 is the MISPRICE
            # retry at the LP's own. A misprice is not only "no column beats
            # its price" — a column that beats it and is ALREADY IN THE POOL is
            # the same thing from the master's side: the round bought nothing.
            # Both send the loop to the true duals, because no certificate and
            # no stall may be declared on a dual the master did not produce.
            smoothed = centre is not None and alpha > 0.0 and attempt == 0
            used = (tuple(alpha * np.asarray(c) + (1.0 - alpha) * np.asarray(e)
                          for c, e in zip(centre, exact))
                    if smoothed else exact)
            values, columns = price(PricingDuals(
                y=used[0], cash=used[1], shed=shed_duals,
                eta=(result.solve.eta
                     if (entry and result.solve is not None) else None)))
            if shed is not None and seeded:
                # No bound on the seeded round. The bound is a Lagrangian bound
                # at the multipliers it was computed with, and those are the
                # pricing's — but on the first round the pricing ran at the SEED
                # (σ = the market path), where a sale is worth p·(1+later) and
                # not p, so the sells' term in the relaxation is positive and
                # dropping it puts the bound below the objective. The LP has no
                # dual of its own yet, so there is nothing to bound yet either:
                # the bound starts from the round the LP has a real model.
                bound = float("inf")
                if _os.environ.get("CHISTA_DEBUG_BOUND"):
                    # Not a `[bound]` line: there is no finite bound at these
                    # multipliers, and printing one would be printing a number
                    # the identity below cannot check.
                    print(f"[seed]  r={result.rounds} "
                          f"obj={result.solve.objective:.1f} bound=n/a "
                          f"(pricing at the σ seed: no finite Lagrangian bound "
                          f"at those multipliers)")
            else:
                bound = lagrangian_bound(
                    MasterSolve(result.solve.lam, np.asarray(used[0]),
                                np.asarray(used[1]), np.asarray(used[2]),
                                result.solve.objective),
                    values, counts, supply_hours, money, days, n_coupling,
                    shed=(None if shed is None or shed_duals is None else
                          (shed[0], float(shed[1]), shed_duals[0],
                           shed_duals[1])),
                    sell=sell_duals, blocks=depth, market=market)
            if bound < result.bound:
                result.bound, centre = bound, used
            if _os.environ.get("CHISTA_DEBUG_BOUND") and not (
                    shed is not None and seeded):
                # The bound's pieces, each one the term the Lagrangian actually
                # carries, so the identity `bound = obj + Σ N_c·rc_c` is readable
                # off the line: `check` is that difference and it is 0 by
                # construction when every row's dual is read from its own row.
                _M = MasterSolve(result.solve.lam, np.asarray(used[0]),
                                 np.asarray(used[1]), np.asarray(used[2]),
                                 result.solve.objective)
                _y = np.asarray(_M.y)
                _c = np.asarray(_M.cash)
                _lab = float((_y[:days, :n_coupling].sum(axis=1)
                              * supply_hours[:days]).sum())
                _csh = float(_c[:days].sum() * float(money))
                _nc = float((np.asarray(counts) * np.asarray(values)).sum())
                _rc = float((np.asarray(counts)
                             * reduced_costs(values, _M.mu)).sum())
                _sg = _tp = _ap = 0.0
                if shed_duals is not None:
                    _sg = float((np.asarray(shed_duals[0])[0]
                                 * np.asarray(shed[0])).sum())
                    _tp = float(np.asarray(shed_duals[1])[:days].sum()
                                * float(shed[1]))
                if sell_duals is not None:
                    _ap = float((sell_duals[0] * sell_duals[1]).sum())
                print(f"[bound] r={result.rounds} obj={_M.objective:.1f} "
                      f"lab={_lab:.1f} cash={_csh:.1f} sig0={_sg:.1f} "
                      f"tau={_tp:.1f} app={_ap:.1f} Nv={_nc:.1f} Nrc={_rc:.1f} "
                      f"bound={bound:.1f} "
                      f"check={bound - _M.objective - _rc:.2f} "
                      f"| sig_max={0.0 if shed_duals is None else float(np.max(shed_duals[0])):.3f} "
                      f"sig_mean={0.0 if shed_duals is None else float(np.mean(shed_duals[0])):.3f} "
                      f"y_max={float(np.max(_y)):.3f} v_max={float(np.max(values)):.3f} "
                      f"sells_tot={float(np.sum(result.solve.sells)):.1f} "
                      f"sells_d0={float(np.sum(result.solve.sells[:, 0])):.1f} "
                      f"sells_d1={float(np.sum(result.solve.sells[:, 1])):.1f} "
                      f"sells_d9={float(np.sum(result.solve.sells[:, 9])):.1f} "
                      f"sig_nz={int(np.count_nonzero(result.solve.sigma))}")
            rc = reduced_costs(values, used[2])

            added = 0
            for c in np.argsort(-rc):
                if rc[c] <= tol:
                    break
                col = columns[int(c)]
                if (col.cls, col.key) in seen:
                    continue           # the pool already holds this plan
                seen.add((col.cls, col.key))
                result.pool.append(col)
                added += 1
            if added or not smoothed:
                break
            alpha *= 0.5               # the smoothed dual bought nothing
        result.rc_history.append(float(np.max(rc)) if rc.size else 0.0)
        result.added_history.append(int(added))

        if not rc.size or float(np.max(rc)) <= tol:
            # No class offers a plan worth having, at the TRUE duals — the
            # retry above guarantees the test was made there. The mix is
            # optimal over the full column set and the subproblems proved it.
            result.certified = True
            return decide()
        if added == 0:
            # Every improving column was already in the pool, at the true
            # duals. That is not a proof: the pricing step says a plan beats
            # what the master pays for it while the master already holds that
            # plan, which means the reduced-cost test disagrees with the LP it
            # came from — a dual sign error, a degenerate tie, or a pricer whose
            # arithmetic is coarser than the tolerance (see RC_REL_TOL).
            result.stopped = (f"stalled: rc {float(np.max(rc)):.6g} above tol "
                              f"{tol:.6g} on a column the pool holds")
            return decide()

    result.stopped = "round cap"
    return decide()
