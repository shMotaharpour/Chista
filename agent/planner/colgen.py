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


@dataclass(frozen=True)
class Column:
    """One class's plan over the horizon, as the master sees it."""

    cls: int                      # which class this plan belongs to, by INDEX
    cost: np.ndarray              # (days, N_COUPLING) coupling consumption
    spend: np.ndarray             # (days,) coins paid to the market
    earn: np.ndarray              # (days,) coins banked
    revenue: float                # total, at the published product prices
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
    """A plan's signature: the chain it runs on each day, and nothing else.

    Two plans that run the same chains are the same column however they were
    priced, and the pool must hold one of them — the lesson's fourth pitfall.
    """
    plan = board.plans[tile] if tile < len(board.plans) else ()
    return tuple((int(d), int(chain)) for d, _state, chain in plan[:days])


def generate(price, supply_hours, money, counts, days, n_coupling,
             idle_columns, *, rounds: int = 12, poll=None,
             deadline=None, warm: list | None = None) -> ColgenResult:
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
                        cls_key=column.cls_key, key=column.key))
    seen = {(c.cls, c.key) for c in result.pool}

    spent = 0.0
    for _ in range(max(1, rounds)):
        if poll is not None:
            poll()
        # A round costs about what the last one cost, so one that starts with
        # less than that left runs PAST the deadline rather than up to it —
        # the caller's turn is a second and the overrun draws on a bank meant
        # for something else. Checking only at the END of the round is what
        # put 25 turns of a season over a 965 ms budget, the worst at 2.7 s.
        # (wsr reached the same rule for its own budget, in #70.)
        if deadline is not None and time.perf_counter() + spent >= deadline:
            result.stopped = "budget"
            return result
        started = time.perf_counter()
        result.solve = solve_master(result.pool, counts, supply_hours, money,
                                    days, n_coupling)
        result.rounds += 1

        values, columns = price(result.solve.y, result.solve.cash)
        rc = reduced_costs(values, result.solve.mu)
        result.rc_history.append(float(np.max(rc)) if rc.size else 0.0)

        if not rc.size or float(np.max(rc)) <= RC_TOL:
            # No class offers a plan worth having. The mix is optimal over the
            # FULL column set and the subproblems just proved it. THIS is the
            # certificate, and it is the reduced costs that carry it — not the
            # fact that nothing was appended.
            result.certified = True
            return result

        added = 0
        for c in np.argsort(-rc):
            if rc[c] <= RC_TOL:
                break
            col = columns[int(c)]
            if (col.cls, col.key) in seen:
                continue           # the pool already holds this plan
            seen.add((col.cls, col.key))
            result.pool.append(col)
            added += 1

        spent = max(spent, time.perf_counter() - started)
        if added == 0:
            # Every improving column was already in the pool. That is NOT a
            # proof: the pricing step says a plan beats what the master pays
            # for it while the master already holds that plan, which means the
            # reduced-cost test disagrees with the LP it was derived from — a
            # dual sign error, or a degenerate tie. Certifying here is how a
            # flipped sign passes for convergence (lesson 1.9 §4.2), so it is
            # reported as a stall and the caller is told the mix is unproven.
            result.stopped = f"stalled: rc {float(np.max(rc)):.6g} on a column the pool holds"
            return result


    result.stopped = "round cap"
    return result
