"""Column generation: does it combine plans, and does it stop on a proof?

The subproblem here is a hand-built enumeration, not the tile DP, for one
reason: with every column enumerable the LP can also be solved DIRECTLY, and
the two answers must agree. `DecisionCraft/notes/phase1_lesson9_dantzig_wolfe.md`
§4.2 records why that check is not optional — a wrong dual sign produces a loop
that *stalls at its initial objective while appearing to converge*, and only the
direct solve catches it.

R007: flip the sign in `reduced_costs` and watch the direct-LP comparison go red.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.planner.colgen import (Column, MasterLP, cash_rows, classes_of, generate,
                                  reduced_costs,
                                  solve_master)

DAYS = 3
N_COUPLING = 1


def _plans():
    """Three plans a class may run, as (hours per day, spend, earn, revenue).

    Deliberately non-dominated: the rich plan earns most and eats the hours,
    the cheap plan earns least and barely touches them, and the middle one is
    what a mix should reach for.
    """
    return [
        # hours/day        spend/day         earn/day        revenue
        (np.array([3.0, 3.0, 3.0]), np.array([90.0, 0, 0]), np.array([0, 0, 200.0]), 200.0),
        (np.array([1.0, 1.0, 1.0]), np.array([20.0, 0, 0]), np.array([0, 0, 55.0]), 55.0),
        (np.array([2.0, 0.0, 0.0]), np.array([10.0, 0, 0]), np.array([0, 40.0, 0]), 40.0),
    ]


def _column(cls, i, spec):
    hours, spend, earn, revenue = spec
    return Column(cls=cls, cost=hours.reshape(DAYS, 1), spend=spend, earn=earn,
                  revenue=revenue, key=("plan", i))


def _idle(cls):
    return Column(cls=cls, cost=np.zeros((DAYS, 1)), spend=np.zeros(DAYS),
                  earn=np.zeros(DAYS), revenue=0.0, key=("idle",))


def _direct(counts, hours, money):
    """The same LP over EVERY column, solved in one go."""
    pool = [_idle(c) for c in range(counts.size)]
    pool += [_column(c, i, spec) for c in range(counts.size)
             for i, spec in enumerate(_plans())]
    return solve_master(pool, counts, hours, money, DAYS, N_COUPLING), pool


def _pricer(counts, flip=False):
    """The subproblem: each class picks its best plan at the published duals."""
    specs = _plans()

    def price(y, cash, shed=None):
        values, columns = [], []
        for c in range(counts.size):
            best, best_col = -np.inf, None
            for i, spec in enumerate(specs):
                hours, spend, earn, revenue = spec
                # priced value = revenue − y·hours − cash·(spend − earn banked)
                cost = float((y[:, 0] * hours).sum())
                # The cash rows are cumulative, so the subproblem prices a
                # coin spent on day d at the sum of the duals from d onward.
                # Pricing it at that day's dual alone is the master and the
                # pricing step disagreeing about the same plan.
                ahead = np.cumsum(np.asarray(cash)[::-1])[::-1]
                later = np.concatenate([ahead[1:], [0.0]])
                cash_use = float((ahead * spend).sum() - (later * earn).sum())
                value = revenue - cost - cash_use
                if value > best:
                    best, best_col = value, _column(c, i, spec)
            values.append(best)
            columns.append(best_col)
        return np.array(values), columns

    return price


def test_the_cash_rows_are_the_loop_they_replaced():
    """`cash_rows` collapses a per-(column, day) loop into two `cumsum`s.

    The loop it replaced was the single largest block in the season's profile —
    18,082,088 `ndarray.sum` calls, 30.9 s of the season's 136.5 s of `act` —
    and the two are the same arithmetic, because `earn[:d].sum()` is
    `cumsum(earn)[d-1]`.

    The guard is the loop's own definition written out, plus a column whose
    earnings land BEFORE its last spend, because that is the only shape that can
    tell the day-at-a-time sum from the running sum: with the shift dropped, a
    plan would be charged for the money it banks on the same day it spends, and
    the cash rows are where "money is a stock" lives — a row off by one day lets
    a plan spend the same purse twice (measured once: 4,268 against 3,000 for
    doing nothing).
    """
    early = (np.array([1.0, 1.0, 1.0]), np.array([10.0, 0, 0]),
             np.array([7.0, 11.0, 0.0]), 18.0)
    pool = [_idle(0), _idle(1)]
    pool += [_column(0, i, spec) for i, spec in enumerate(_plans())]
    pool += [_column(0, 9, early)]

    reference = np.zeros((DAYS, len(pool)))
    for j, col in enumerate(pool):
        spend = np.cumsum(np.asarray(col.spend, dtype=np.float64))
        earn = np.asarray(col.earn, dtype=np.float64)
        for d in range(DAYS):
            reference[d, j] = spend[d] - (earn[:d].sum() if d else 0.0)

    got = cash_rows(pool, DAYS)
    assert got.shape == reference.shape
    assert np.allclose(got, reference, rtol=0.0, atol=1e-12)

    # The hand-computed column: spend 10 on day 0, bank 7 then 11 after it.
    assert got[0, -1] == 10.0        # nothing banked before day 0
    assert got[1, -1] == 3.0         # 10 spent, 7 already in
    assert got[2, -1] == -8.0        # 10 spent, 18 banked: the row may go negative
    assert got[:, 0].tolist() == [0.0] * DAYS


def test_the_master_LP_agrees_with_scipy_linprog_and_with_a_cold_solve():
    """The same matrices, the same optimum, and the same three duals.

    `MasterLP` drives HiGHS directly — the solver `linprog(method="highs")` wraps
    — so a day's rounds can share a basis, measured 4.48x on a real round
    sequence at 15-464 columns. This is the guard that the warm start did not
    move the answer: the solution, the objective, the coupling duals `y`, the
    cash duals and the class marginals `mu`, all against `linprog` on the very
    same matrices, and then a second solve on a GROWN pool against a cold one —
    which is the path a basis-carrying round takes.
    """
    from scipy.optimize import linprog

    counts = np.array([4])
    pool = [_idle(0)] + [_column(0, i, spec) for i, spec in enumerate(_plans())]
    money = 300.0
    A_q = np.stack([c.cost.T.reshape(-1) for c in pool], axis=1)
    A_c = cash_rows(pool, DAYS)
    b_c = np.full(DAYS, float(money))
    A_e = np.zeros((1, len(pool)))
    for j, col in enumerate(pool):
        A_e[col.cls, j] = 1.0

    def reference(hours):
        """`linprog` on the same matrices — the path this replaced."""
        b_q = np.tile(hours[:DAYS], N_COUPLING)
        ref = linprog(-revenue, A_ub=np.vstack([A_q, A_c]),
                      b_ub=np.concatenate([b_q, b_c]), A_eq=A_e,
                      b_eq=counts.astype(np.float64),
                      bounds=[(0.0, None)] * len(pool), method="highs")
        marg = np.asarray(ref.ineqlin.marginals)
        return ref, np.maximum(-marg[:N_COUPLING * DAYS], 0.0) \
            .reshape(N_COUPLING, DAYS).T, \
            np.maximum(-marg[N_COUPLING * DAYS:], 0.0), \
            np.asarray(ref.eqlin.marginals)

    revenue = np.array([c.revenue for c in pool], dtype=np.float64)
    # Slack: the labour row has room to spare. Loose on purpose — a binding row
    # is the case below.
    mine = solve_master(pool, counts, np.full(DAYS, 20.0), money, DAYS, N_COUPLING)
    ref, y_ref, cash_ref, mu_ref = reference(np.full(DAYS, 20.0))
    assert np.allclose(mine.lam, ref.x, rtol=0.0, atol=1e-9)
    assert abs(mine.objective - (-float(ref.fun))) <= 1e-6
    assert np.allclose(mine.y, y_ref, rtol=0.0, atol=1e-9)
    assert np.allclose(mine.cash, cash_ref, rtol=0.0, atol=1e-9)
    assert np.allclose(mine.mu, mu_ref, rtol=0.0, atol=1e-9)

    # Binding: three hours a day is exactly one rich plan's worth, so the row is
    # tight, `y` is nonzero, and a sign error in the duals CAN fail this guard —
    # with a slack row the duals are 0 and `np.maximum(±0, 0)` hides it.
    tight = np.full(DAYS, 3.0)
    bound_case = solve_master(pool, counts, tight, money, DAYS, N_COUPLING)
    assert bound_case.y.max() > 0.0, "the tight case must bind, or it guards nothing"
    ref, y_ref, cash_ref, mu_ref = reference(tight)
    assert np.allclose(bound_case.lam, ref.x, rtol=0.0, atol=1e-9)
    assert np.allclose(bound_case.y, y_ref, rtol=0.0, atol=1e-9)
    assert np.allclose(bound_case.cash, cash_ref, rtol=0.0, atol=1e-9)
    assert np.allclose(bound_case.mu, mu_ref, rtol=0.0, atol=1e-9)

    # The basis-carrying path: one instance, two pools, the second grown.
    roomy = np.full(DAYS, 20.0)
    grower = MasterLP()
    grower.solve(pool, counts, roomy, money, DAYS, N_COUPLING)
    grown = pool + [_column(0, 9, (
        np.array([2.0] * DAYS), np.array([5.0] + [0.0] * (DAYS - 1)),
        np.array([0.0] * (DAYS - 1) + [80.0]), 80.0))]
    warm = grower.solve(grown, counts, roomy, money, DAYS, N_COUPLING)
    cold = MasterLP().solve(grown, counts, roomy, money, DAYS, N_COUPLING)
    assert abs(warm.objective - cold.objective) <= 1e-6
    assert np.allclose(warm.lam, cold.lam, rtol=0.0, atol=1e-8)
    assert np.allclose(warm.mu, cold.mu, rtol=0.0, atol=1e-8)


def test_the_pool_grows_and_the_master_mixes_plans():
    """One class, many tiles: the answer is a MIX, not n copies of one plan."""
    counts = np.array([10])
    hours = np.full(DAYS, 12.0)
    res = generate(_pricer(counts), hours, 300.0, counts, DAYS, N_COUPLING,
                   [_idle(0)], rounds=12)

    assert res.certified, f"the loop stopped without a proof: {res.stopped}"
    assert len(res.pool) > 2, (
        f"the pool holds {len(res.pool)} columns — a loop that replaces its "
        f"columns instead of accumulating them can never hold more than two")
    mixed = int((np.asarray(res.solve.lam) > 1e-6).sum())
    assert mixed >= 2, (
        f"only {mixed} column carries weight: the master is scaling one plan, "
        f"not combining several")


def test_column_generation_finds_the_direct_LPs_optimum():
    """The whole point: same LP, different path, same number.

    Lesson 1.9's own acceptance check, and the one that catches a sign error —
    which does not look like a crash, it looks like convergence at the wrong
    objective.
    """
    counts = np.array([7, 4])
    hours = np.full(DAYS, 9.0)
    money = 240.0

    direct, _ = _direct(counts, hours, money)
    res = generate(_pricer(counts), hours, money, counts, DAYS, N_COUPLING,
                   [_idle(0), _idle(1)], rounds=25)

    assert res.certified, f"no certificate: {res.stopped}"
    assert res.solve.objective == pytest.approx(direct.objective, rel=1e-9), (
        f"column generation {res.solve.objective:.9f} != direct LP "
        f"{direct.objective:.9f}")


def test_the_certificate_is_the_stopping_rule_not_the_round_cap():
    """`certified` may only be true when a pricing round found nothing to add."""
    counts = np.array([5])
    hours = np.full(DAYS, 8.0)
    res = generate(_pricer(counts), hours, 200.0, counts, DAYS, N_COUPLING,
                   [_idle(0)], rounds=1)
    assert not res.certified and res.stopped == "round cap", (
        "one round cannot both price and prove there is nothing left to price")


def test_a_class_is_a_graph_state_and_a_distance():
    """Two tiles of one state at different distances are different classes.

    The contractor prices a STATE; a worker walks to a SQUARE. The farm is
    cleared every night and the farmer respawns on a shed door (F040), so a
    tile is reached afresh on every day it is worked — `v` working days on a
    tile `d` steps out is at least `v·d` hours of walking, and a column that
    every tile of its class runs cannot be honest about that unless the tiles
    really are the same distance out.
    """
    reps, counts, of_tile = classes_of([7, 7, 7, 9, 7, 9],
                                       [0, 0, 3, 1, 0, 1])
    assert reps == [(7, 0), (7, 3), (9, 1)]
    assert list(counts) == [3, 1, 2]
    assert of_tile == [0, 0, 1, 2, 0, 2]

    # Without distances the old state-only classes come back, which is what a
    # caller with no board in hand means.
    reps, counts, of_tile = classes_of([7, 7, 9])
    assert reps == [(7, 0), (9, 0)] and list(counts) == [2, 1]


def test_the_convexity_dual_keeps_its_sign():
    """`mu` is an EQUALITY marginal and is free in sign; clamping breaks the test."""
    counts = np.array([3])
    pool = [_idle(0)] + [_column(0, i, s) for i, s in enumerate(_plans())]
    solve = solve_master(pool, counts, np.full(DAYS, 20.0), 500.0, DAYS, N_COUPLING)
    rc = reduced_costs(np.array([solve.objective / 3.0]), solve.mu)
    assert np.isfinite(rc).all()
    assert (np.asarray(solve.y) >= 0).all() and (np.asarray(solve.cash) >= 0).all()


def test_a_wrong_dual_sign_is_reported_as_a_stall_and_never_as_a_proof():
    """The nastiest failure in lesson 1.9 §4.2, made visible.

    A flipped `mu` does not crash — it makes the pricing step claim a plan
    beats what the master pays for it while the master already holds that
    plan. The loop then appends nothing and, before this was fixed, called
    that a certificate. It is the opposite: the reduced-cost test disagreeing
    with the LP it came from.

    The case needs a class that FULLY COMMITS — with an idle column carrying
    weight, `mu` is exactly 0 and no sign error can show. Here nothing is
    scarce, so every tile takes the best plan, idle goes to zero and
    `mu = −200`.
    """
    counts = np.array([3])
    hours = np.full(DAYS, 1e6)          # nothing scarce: the class commits fully
    money = 1e6

    good = generate(_pricer(counts), hours, money, counts, DAYS, N_COUPLING,
                    [_idle(0)], rounds=25)
    assert good.certified and good.rounds == 2

    import agent.planner.colgen as CG
    original = CG.reduced_costs
    CG.reduced_costs = lambda v, mu: np.asarray(v, float) - np.asarray(mu, float)
    try:
        bad = generate(_pricer(counts), hours, money, counts, DAYS, N_COUPLING,
                       [_idle(0)], rounds=25)
    finally:
        CG.reduced_costs = original

    assert not bad.certified, (
        "a flipped dual sign was reported as a proof — which is exactly how "
        "lesson 1.9's stalled loop passes for convergence")
    assert bad.stopped.startswith("stalled"), bad.stopped


def test_mu_is_zero_while_a_class_still_has_idle_weight():
    """Why the sign check needs a fully committed class, recorded not assumed.

    With the idle column absorbing weight, forcing one more tile into the
    class costs nothing, so its convexity dual is 0 — and `value ± 0` is the
    same number whichever sign the test uses.
    """
    counts = np.array([10])
    solve = solve_master([_idle(0)], counts, np.full(DAYS, 12.0), 300.0,
                         DAYS, N_COUPLING)
    assert solve.mu[0] == pytest.approx(0.0, abs=1e-9)


def test_the_lagrangian_bound_is_a_bound():
    """`L(y) = y·b + Σ_c N_c·v_c(y)` must never fall below what it bounds.

    It did: −71,123 against an objective of 34,197. A class's value came back
    negative because the tile DP chooses its chain before the travel term is
    added to the column, so a chain it liked can be a loss once the walk is
    paid for — and a class is never worth less than its idle column, which is
    worth zero. Flooring there is not a fudge: it is including a column the
    pricing step has and was not reporting.
    """
    counts = np.array([7, 4])
    hours = np.full(DAYS, 9.0)
    money = 240.0
    res = generate(_pricer(counts), hours, money, counts, DAYS, N_COUPLING,
                   [_idle(0), _idle(1)], rounds=25)

    assert res.certified, res.stopped
    assert np.isfinite(res.bound)
    assert res.bound >= res.solve.objective - 1e-6, (
        f"the bound {res.bound:.3f} is below the objective "
        f"{res.solve.objective:.3f}, so it is not a bound")
    assert res.gap >= -1e-9


def test_smoothing_never_certifies_on_a_dual_the_master_did_not_produce():
    """A misprice goes to the TRUE duals before anything is called optimal.

    Wentges prices at `α·π_best + (1−α)·π_LP`. A certificate declared there
    would certify a problem nobody solved — and a column that beats its price
    while ALREADY BEING IN THE POOL is a misprice too, from the master's side:
    the round bought nothing either way.
    """
    counts = np.array([10])
    hours = np.full(DAYS, 12.0)
    plain = generate(_pricer(counts), hours, 300.0, counts, DAYS, N_COUPLING,
                     [_idle(0)], rounds=25, smoothing=0.0)
    smoothed = generate(_pricer(counts), hours, 300.0, counts, DAYS,
                        N_COUPLING, [_idle(0)], rounds=25, smoothing=0.8)

    for name, res in (("plain", plain), ("smoothed", smoothed)):
        assert res.certified, f"{name}: {res.stopped}"
        assert res.bound >= res.solve.objective - 1e-6, name
    assert smoothed.solve.objective == pytest.approx(plain.solve.objective,
                                                     rel=1e-9), (
        "smoothing moved the optimum, which means it reached the pricing "
        "step's own objective and not only the path to it")
