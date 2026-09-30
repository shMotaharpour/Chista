"""The risk block must agree with CVaR's closed form, or it is the wrong block.

The LP's own `t + mean(u)` at the optimum IS the CVaR of the profit
distribution (Rockafellar-Uryasev), so the guard solves a tiny LP with the block
spliced in and compares that value with `cvar_of` -- a hand-checkable number,
not a tolerance on a plausible-looking row. It can fail: flip the sign of the
scenario row and the two disagree.
"""
import numpy as np
from scipy.optimize import linprog

from agent.planner.risk import cvar_of, risk_block


def test_the_block_matches_the_closed_form_and_kappa_zero_is_the_mean():
    profits = np.array([10.0, 0.0, -10.0])       # the plan's profit, per scenario
    x = np.array([1.0])                          # one existing column, held fixed
    cost = np.array([0.0])                       # it costs nothing to hold
    scen_cost = -np.outer(profits, x)            # profit_s = -scen_cost_s . x
    kappa, alpha = 1.0, 1.0 / 3.0

    base, new_cost, rows, rl, ru = risk_block(cost, scen_cost, kappa, alpha)
    # variables: [x (fixed), t, u_1..u_S]; the rows are `u_s - t + profit_s >= 0`
    c = np.concatenate([base, new_cost])
    A_ub = -rows                                 # A x >= rl  <=>  -A x <= -rl
    b_ub = -rl
    bounds = [(1.0, 1.0)] + [(None, None)] + [(0.0, None)] * len(profits)
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
    assert res.success, res.message
    t, u = res.x[1], res.x[2:]
    assert abs((t + u.mean()) - cvar_of(profits, alpha)) < 1e-9

    # kappa = 0 leaves the mean objective and no tail weight at all
    zero_base, zero_new, _, _, _ = risk_block(cost, scen_cost, 0.0, alpha)
    assert np.allclose(zero_base, scen_cost.mean(axis=0))
    assert np.allclose(zero_new, 0.0)
