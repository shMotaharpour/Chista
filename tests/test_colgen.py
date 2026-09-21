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

from agent.planner.colgen import (Column, classes_of, generate, reduced_costs,
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

    def price(y, cash):
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
