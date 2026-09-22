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

import time
from dataclasses import dataclass, field

import numpy as np

from agent.world.model import RESOURCE_ID, SHED_ITEMS


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
#: the master for nothing (lesson 1.9 §4.4: dedupe or the master fattens).
RC_TOL = 1e-6

#: ...and a reduced cost lives on the OBJECTIVE's scale, so the floor above is
#: not enough on its own: `rc` is a difference of coins, and 1e-6 absolute is
#: below the noise floor of any board worth a few thousand.
#:
#: The pricer's own precision is the binding one. `TileContractor` sweeps in
#: float32 (`DTYPE`), so every class value carries ~1.2e-7 relative error, and
#: the LP's duals come back in float64 from HiGHS. On the real board below, the
#: loop stalled on `rc 2.24e-4` — 6.3e-9 relative at an objective of 35,772,
#: i.e. inside the pricer's own noise — and the column it wanted was already in
#: the pool: raising the tolerance to 1e-3 certified the SAME objective
#: (35,772.2194) with the same bound (35,772.2202). A tolerance is therefore
#: read as `max(RC_TOL, RC_REL_TOL · |objective|)`, and the constant is float32's
#: epsilon with room for a horizon's accumulation.
RC_REL_TOL = 1e-6


def rc_tolerance(objective: float) -> float:
    """The reduced-cost tolerance for a board whose LP objective is this.

    One definition, read by the loop that certifies and by the guards that check
    the certificate: `rc` is a difference of two quantities on the objective's
    scale (`value + mu`, with `mu` the convexity marginal), and the value comes
    out of a float32 sweep, so the cancellation floor is `|objective| · eps` —
    about 4.3e-3 at an objective of 35,772. Anything below that is the pricer's
    own rounding, and refusing to certify on it is refusing to certify at all.
    """
    return max(RC_TOL, RC_REL_TOL * abs(float(objective)))


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
    #: (n_goods, days) what the master decided to SELL, in `market` order.
    sells: np.ndarray = None

    def __post_init__(self) -> None:
        for name in ("sigma", "tau", "sells"):
            if getattr(self, name) is None:
                object.__setattr__(self, name, np.zeros((0,)))


def cash_rows(pool: list[Column], days: int) -> np.ndarray:
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
    """
    if not pool:
        return np.zeros((days, 0))
    spend = np.cumsum(np.stack([np.asarray(col.spend, dtype=np.float64)
                                for col in pool]), axis=1)[:, :days]
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
              market: tuple[int, ...] = ()) -> MasterSolve:
        """The restricted master over the pool, with the SHED as a stock.

        Variables: `lambda_j >= 0` per column, then per day the sells, the stock
        and the waste — `sell[g,d]`, `stock[i,d]`, `waste[i,d]` over the shed's
        own items (`PRODUCTS + ANIMALS`, the engine's own order). Rows: the
        coupling rows (labour), then the BALANCE rows

            stock[i,d+1] − stock[i,d] − Σ_j λ_j·produce_j[d,i] + sell[i,d]
            + waste[i,d] = 0

        with `stock[i,0]` FIXED at the opening shed, the cap rows
        `Σ_i stock[i,d] <= capacity`, the cumulative cash rows, and the
        convexity rows. The objective is `Σ p[i,d]·sell[i,d]`: a column earns
        nothing directly — what it produces feeds the stock, and the stock's
        dual σ is the internal price of a good (`published_duals` reads it).

        This is the point of the change. The master used to credit each day's
        production at THAT day's market price, so holding had no value, a day of
        delay was free, the contractor tied between working and waiting, and the
        tie-break (lowest edge index = the idle chain) made the farm do nothing
        all season. `waste` is what keeps the cap row feasible: the engine
        discards the night flush's overflow, so the LP must be able to as well.
        """
        names = SHED_ITEMS
        items = len(names) if shed_stock is not None else 0
        rids = [_resource_of(n) for n in names]
        n_goods = len(market)
        n = len(pool)
        n_classes = int(counts.size)
        revenue = np.array([c.revenue for c in pool], dtype=np.float64)
        # quantity rows: (n_coupling·days, n)
        A_q = np.stack([c.cost.T.reshape(-1) for c in pool], axis=1) if n else \
            np.zeros((n_coupling * days, 0))
        b_q = np.tile(hours[:days], n_coupling)   # LABOUR is the only row today

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
        A_c = cash_rows(pool, days)
        b_c = np.full(days, float(money))

        # convexity: one row per class
        A_e = np.zeros((n_classes, n))
        for j, col in enumerate(pool):
            A_e[col.cls, j] = 1.0

        # --- the shed: sells, stock, waste, and their rows --------------------
        # Layout: [lambda (n) | sell (n_goods·days) | stock (items·days)
        #          | waste (items·days)]
        block = n_goods * days
        stock0 = n + block
        waste0 = stock0 + items * days
        n_cols = waste0 + items * days

        def col_sell(gi: int, d: int) -> int:
            return n + gi * days + d

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
        lower = np.zeros(n_cols)
        upper = np.full(n_cols, np.inf)
        if items:
            px = np.asarray(prices, dtype=np.float64)
            for gi, ii in enumerate(market):
                for d in range(days):
                    # `prices` is the (days, len(market)) market path, so the
                    # good's own index is `gi`, NOT the shed-item index `ii`.
                    cost[col_sell(gi, d)] = -float(px[d, gi])   # max p·sell
            # `stock[i, 0]` is the opening shed: fixed, not a decision.
            for ii in range(items):
                j = col_stock(ii, 0)
                lower[j] = upper[j] = float(shed_stock[ii])

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
                    bal[row, col_sell(sellable, d)] = 1.0
        if items:
            prod = np.zeros((items * days, n))
            for j, col in enumerate(pool):
                if col.produce is None:
                    continue
                produced = np.asarray(col.produce, dtype=np.float64)
                for ii, rid in enumerate(rids):
                    if rid is None or rid >= produced.shape[1]:
                        continue
                    for d in range(min(days, produced.shape[0])):
                        prod[ii * days + d, j] = float(produced[d, rid])
            bal[:, :n] -= prod

        # cap rows: one per day, Σ_i stock[i, d] <= capacity
        cap = np.zeros((days, n_cols))
        for ii in range(items):
            for d in range(days):
                cap[d, col_stock(ii, d)] = 1.0

        # The cash rows earn from the sells: a sale on day d' is money in the
        # purse from d' onward. Same convention as the columns' own spend — the
        # coin is available the day AFTER it is earned, which is what the
        # original cumulative row did and what the engine does.
        if items and n_goods:
            A_c = np.hstack([A_c, np.zeros((days, n_cols - n))])
            px = np.asarray(prices, dtype=np.float64)
            for gi, ii in enumerate(market):
                for d in range(days):
                    A_c[d:, col_sell(gi, d)] = -float(px[d, ii])

        target = counts.astype(np.float64)
        if n_cols > n:
            # The coupling and convexity rows only involve the columns; the shed
            # variables enter through the balance, the cap and the cash rows.
            pad = n_cols - n
            A_q = np.hstack([A_q, np.zeros((A_q.shape[0], pad))])
            A_e = np.hstack([A_e, np.zeros((n_classes, pad))])
        rows = np.vstack([A_q, A_c, bal, cap, A_e])
        n_ineq = n_coupling * days + days + items * days + days
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
            target])
        lp.row_upper_ = np.concatenate([
            b_q, b_c,
            np.zeros(items * days),
            np.full(days, float(shed_capacity)),
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

        self._highs.passModel(lp)
        layout = (n_coupling, days, items, n_goods, n_classes)
        if self._basis is not None and self._layout == layout:
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
        marg = np.asarray(solution.row_dual, dtype=np.float64)
        y = np.maximum(-marg[:n_coupling * days], 0.0).reshape(n_coupling, days).T
        off = n_coupling * days
        cash = np.maximum(-marg[off:off + days], 0.0)
        # The balance rows are EQUALITIES: their duals are FREE in sign, because
        # a good in the shed is worth what it can be sold for later — and that is
        # the internal price the contractor is credited at. The cap rows' duals
        # are ≥ 0 (a ≤ row) and say what a unit of shed room is worth.
        sigma = np.asarray(marg[off + days:off + days + items * days],
                           dtype=np.float64).reshape(items, days).T
        tau = np.maximum(-marg[off + days + items * days:n_ineq], 0.0)
        # The convexity duals are EQUALITY marginals and are free in sign: a class
        # whose tiles are worth having carries a negative one. Clamping them would
        # break the reduced-cost test, which is the only reason they are read.
        mu = np.asarray(marg[n_ineq:], dtype=np.float64)
        values = np.asarray(solution.col_value, dtype=np.float64)
        return MasterSolve(lam=values[:n], y=y, cash=cash, mu=mu,
                           objective=-float(self._highs.getObjectiveValue()),
                           sigma=sigma, tau=tau,
                           sells=values[n:n + block].reshape(n_goods, days))


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
                     shed: tuple | None = None) -> float:
    """`L(y) = y·b + Σ_c N_c · v_c(y)` — an UPPER bound on the master's optimum.

    Relax the coupling rows with their duals and the problem separates into
    the classes, each free to pick its best plan at those prices. `values[c]`
    IS that best dual-priced value: it is what the pricing step just computed.
    So the bound costs nothing beyond an inner product, and without it column
    generation has no idea how far from done it is — "no column has a positive
    reduced cost" is a certificate at the END and says nothing on the way.

    It is also what makes Wentges smoothing work: the stability centre is the
    dual with the BEST bound so far, not the last one seen.

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
    return rhs + float((np.asarray(counts, dtype=np.float64)
                        * np.asarray(values, dtype=np.float64)).sum())


@dataclass
class ColgenResult:
    """The mix, the duals, and whether the answer carries a proof."""

    pool: list[Column] = field(default_factory=list)
    solve: MasterSolve | None = None
    rounds: int = 0
    #: True only when a pricing round found NO class with rc > RC_TOL. That is
    #: the optimality certificate. False means the budget or the round cap
    #: stopped the loop, and the mix is an incumbent, not an optimum.
    certified: bool = False
    stopped: str = ""             # why the loop ended, when it was not certified
    rc_history: list = field(default_factory=list)
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


def generate(price, supply_hours, money, counts, days, n_coupling,
             idle_columns, *, rounds: int = 12, poll=None,
             deadline=None, warm: list | None = None,
             smoothing: float = 0.0, shed: tuple | None = None,
             prices: np.ndarray | None = None,
             market: tuple[int, ...] = ()) -> ColgenResult:
    """The loop: master over every column so far, price, add, repeat.

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

    spent = 0.0
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
        if poll is not None:
            poll()
        if deadline is not None and time.perf_counter() + spent >= deadline:
            result.stopped = "budget"
            return result
        started = time.perf_counter()
        result.solve = solver.solve(
            result.pool, counts, supply_hours, money, days, n_coupling,
            shed_stock=None if shed is None else shed[0],
            shed_capacity=0.0 if shed is None else float(shed[1]),
            prices=prices, market=market)
        result.rounds += 1

        exact = (result.solve.y, result.solve.cash, result.solve.mu)
        # A degenerate first LP has NO unique dual: with an empty shed and only
        # do-nothing columns the whole model is zero and HiGHS hands back σ = 0,
        # so the pricing credits every plan nothing, no working column is ever
        # generated, and the loop sits at zero for good. Seed σ from the market
        # path in that case — the same job the old `p_eff` did — and use the
        # seeded σ in the bound too, which stays a bound because a Lagrangian
        # bound is valid at ANY multipliers, not only at the optimal ones.
        shed_duals = None
        if shed is not None:
            sig = np.asarray(result.solve.sigma, dtype=np.float64)
            if result.rounds <= 1:
                # The FIRST LP has only do-nothing columns, so it is entirely
                # zero and its dual is not unique — HiGHS returns σ = 0 (or a
                # meaningless negative) and the pricing would credit every plan
                # nothing, so no working column is ever generated and the loop
                # sits at zero for good. Price the first round at the market path
                # instead: that generates the working columns, and from round 2
                # the LP has a real model and its own σ is used. The bound is
                # computed with the SAME σ, and stays a bound, because a
                # Lagrangian bound holds at any multipliers, not only optimal.
                sig = _seed_sigma(prices, market, days)
            shed_duals = (np.maximum(sig, 0.0), result.solve.tau)
        # The reduced-cost tolerance for THIS board: an absolute floor, raised to
        # the pricer's own precision on the objective's scale (see RC_REL_TOL).
        tol = rc_tolerance(result.solve.objective)
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
            values, columns = price(used[0], used[1], shed_duals)
            if shed is not None and result.rounds <= 1:
                # No bound on the seeded round. The bound is a Lagrangian bound
                # at the multipliers it was computed with, and those are the
                # pricing's — but on the first round the pricing ran at the SEED
                # (σ = the market path), where a sale is worth p·(1+later) and
                # not p, so the sells' term in the relaxation is positive and
                # dropping it puts the bound below the objective. The LP has no
                # dual of its own yet, so there is nothing to bound yet either:
                # the bound starts from the round the LP has a real model.
                bound = float("inf")
            else:
                bound = lagrangian_bound(
                    MasterSolve(result.solve.lam, np.asarray(used[0]),
                                np.asarray(used[1]), np.asarray(used[2]),
                                result.solve.objective),
                    values, counts, supply_hours, money, days, n_coupling,
                    shed=(None if shed is None or shed_duals is None else
                          (shed[0], float(shed[1]), shed_duals[0],
                           shed_duals[1])))
            if bound < result.bound:
                result.bound, centre = bound, used
            import os as _os
            if _os.environ.get("CHISTA_DEBUG_BOUND") and result.rounds <= 2:
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
                _sg = _tp = 0.0
                if shed_duals is not None:
                    _sg = float((np.asarray(shed_duals[0])[0]
                                 * np.asarray(shed[0])).sum())
                    _tp = float(np.asarray(shed_duals[1])[:days].sum()
                                * float(shed[1]))
                print(f"[bound] r={result.rounds} obj={_M.objective:.1f} "
                      f"lab={_lab:.1f} cash={_csh:.1f} sig0={_sg:.1f} "
                      f"tau={_tp:.1f} Nv={_nc:.1f} Nrc={_rc:.1f} "
                      f"bound={bound:.1f} "
                      f"check={bound - _M.objective - _rc:.2f}")
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

        if not rc.size or float(np.max(rc)) <= tol:
            # No class offers a plan worth having, at the TRUE duals — the
            # retry above guarantees the test was made there. The mix is
            # optimal over the full column set and the subproblems proved it.
            result.certified = True
            return result
        if added == 0:
            # Every improving column was already in the pool, at the true
            # duals. That is not a proof: the pricing step says a plan beats
            # what the master pays for it while the master already holds that
            # plan, which means the reduced-cost test disagrees with the LP it
            # came from — a dual sign error, a degenerate tie, or a pricer whose
            # arithmetic is coarser than the tolerance (see RC_REL_TOL).
            result.stopped = (f"stalled: rc {float(np.max(rc)):.6g} above tol "
                              f"{tol:.6g} on a column the pool holds")
            return result

        spent = max(spent, time.perf_counter() - started)

    result.stopped = "round cap"
    return result
