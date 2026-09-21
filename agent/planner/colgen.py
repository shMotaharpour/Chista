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

try:
    from scipy.optimize import linprog
    HAS_SCIPY = True
except Exception:                                  # noqa: BLE001
    HAS_SCIPY = False

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


def solve_master(pool: list[Column], counts: np.ndarray, hours: np.ndarray,
                 money: float, days: int, n_coupling: int) -> MasterSolve:
    """max Σ λ·revenue  s.t. the coupling rows, the cash rows, convexity.

    One convexity row PER CLASS, with the class's tile count on the right —
    not one global row. A global row lets a plan of class A absorb the weight
    class B could not use, which is not a plan any tile can run.
    """
    if not HAS_SCIPY:
        raise RuntimeError("column generation needs scipy.optimize.linprog")
    n = len(pool)
    n_classes = int(counts.size)

    revenue = np.array([c.revenue for c in pool], dtype=np.float64)
    # quantity rows: (n_coupling·days, n)
    A_q = np.stack([c.cost.T.reshape(-1) for c in pool], axis=1) if n else \
        np.zeros((n_coupling * days, 0))
    b_q = np.tile(hours[:days], n_coupling)       # LABOR is the only row today

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
    A_c = np.zeros((days, n))
    for j, col in enumerate(pool):
        spend = np.cumsum(np.asarray(col.spend, dtype=np.float64))
        earn = np.asarray(col.earn, dtype=np.float64)
        for d in range(days):
            A_c[d, j] = spend[d] - (earn[:d].sum() if d else 0.0)
    b_c = np.full(days, float(money))

    # convexity: one row per class
    A_e = np.zeros((n_classes, n))
    for j, col in enumerate(pool):
        A_e[col.cls, j] = 1.0

    res = linprog(-revenue,
                  A_ub=np.vstack([A_q, A_c]),
                  b_ub=np.concatenate([b_q, b_c]),
                  A_eq=A_e, b_eq=counts.astype(np.float64),
                  bounds=[(0.0, None)] * n, method="highs")
    if not res.success:
        raise RuntimeError(f"master LP failed: {res.message}")

    # scipy's ≤-marginals are ≤ 0 in min form; the shadow prices are −them,
    # clamped onto R006's orthant before anything downstream sees them.
    marg = np.asarray(res.ineqlin.marginals)
    y = np.maximum(-marg[:n_coupling * days], 0.0).reshape(n_coupling, days).T
    cash = np.maximum(-marg[n_coupling * days:], 0.0)
    # The convexity duals are EQUALITY marginals and are free in sign: a class
    # whose tiles are worth having carries a negative one. Clamping them would
    # break the reduced-cost test, which is the only reason they are read.
    mu = np.asarray(res.eqlin.marginals, dtype=np.float64)
    return MasterSolve(np.asarray(res.x), y, cash, mu, -float(res.fun))


def reduced_costs(values: np.ndarray, mu: np.ndarray) -> np.ndarray:
    """`rc_c` for each class: how much its best plan beats what the master pays.

    The contractor already maximises `p·produce − w·cost` at the published
    duals, so `values[c]` IS the dual-priced plan value of the lesson's
    subproblem. In the master's min form a column improves when
    `−value − mu_c < 0`, i.e. when `value + mu_c > 0` — and `mu` keeps its
    sign for exactly this line.
    """
    return np.asarray(values, dtype=np.float64) + np.asarray(mu, dtype=np.float64)


def lagrangian_bound(solve: MasterSolve, values: np.ndarray,
                     counts: np.ndarray, hours: np.ndarray, money: float,
                     days: int, n_coupling: int) -> float:
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
             smoothing: float = 0.0) -> ColgenResult:
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

    for _ in range(max(1, rounds)):
        if poll is not None:
            poll()
        if deadline is not None and time.perf_counter() + spent >= deadline:
            result.stopped = "budget"
            return result
        started = time.perf_counter()
        result.solve = solve_master(result.pool, counts, supply_hours, money,
                                    days, n_coupling)
        result.rounds += 1

        exact = (result.solve.y, result.solve.cash, result.solve.mu)
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
            values, columns = price(used[0], used[1])
            bound = lagrangian_bound(
                MasterSolve(result.solve.lam, np.asarray(used[0]),
                            np.asarray(used[1]), np.asarray(used[2]),
                            result.solve.objective),
                values, counts, supply_hours, money, days, n_coupling)
            if bound < result.bound:
                result.bound, centre = bound, used
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
