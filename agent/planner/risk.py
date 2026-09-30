"""The cash tail, as rows of the same LP (#111, Rockafellar-Uryasev).

The master is a MEAN model: a sale on day 20 and a sale on day 0 are worth the
same, so postponing is free and nothing prices the chance that the purse runs
out. This module builds the block that prices the tail -- and it is built HERE,
on its own, so the arithmetic can be checked by hand before it is spliced into
`colgen`'s positional row layout (where a row inserted in the wrong place
silently re-labels one dual as another).

The formulation is the standard one:

    CVaR_alpha(profit) = min_t  t + (1/(alpha*S)) * sum_s max(0, t - profit_s)

so the LP gains ONE column `t`, ONE column per scenario `u_s`, and 2S rows
(`u_s >= 0` is the column's own bound, `u_s >= t - profit_s` is the row). No
binary variable appears anywhere, which is exactly why CVaR and not VaR is the
one that fits an LP.

`profit_s` is a linear function of the plan, so each row is a coefficient vector:
`u_s - t + profit_s >= 0`. The caller supplies, per scenario, the cost vector
`c_s` with `profit_s = -c_s . x` -- that is the same `cost` vector the LP
minimises, read with that scenario's own prices.

`kappa` mixes the two objectives: `(1 - kappa) * mean + kappa * CVaR`. At
`kappa = 0` the block is not built at all, which is the bit-identical case the
whole change depends on.
"""

from __future__ import annotations

import numpy as np


def risk_block(cost: np.ndarray, scenario_cost: np.ndarray, kappa: float,
               alpha: float) -> tuple[np.ndarray, np.ndarray, np.ndarray,
                                    np.ndarray, np.ndarray]:
    """The R-U columns and rows for an LP whose objective is `cost . x`.

    `cost` is the problem's own objective vector -- positive entries are money
    spent, negative entries money earned -- and `scenario_cost` is `(S, n_cols)`
    the same vector repriced under each scenario. `profit_s = -scenario_cost[s].x`.

    Returns `(col_cost, rows, row_lower, row_upper)`: the cost of the `1 + S`
    new columns (`t` first, then the scenarios), the `S` new rows (sparse-dense:
    `(S, n_cols + 1 + S)`, the existing columns first, then `t`, then `u_s`),
    and the rows' bounds. The caller appends them, and `u_s >= 0` is left to the
    bounds so no row is spent on it.
    """
    cost = np.asarray(cost, dtype=np.float64)
    scenario_cost = np.asarray(scenario_cost, dtype=np.float64)
    if scenario_cost.ndim != 2 or scenario_cost.shape[1] != cost.size:
        raise ValueError(
            f"scenario_cost: expected (S, {cost.size}), got {scenario_cost.shape}")
    n_scen = scenario_cost.shape[0]
    if n_scen == 0:
        raise ValueError("risk_block needs at least one scenario")
    kappa = float(kappa)
    alpha = float(alpha)
    if not 0.0 < alpha <= 1.0:
        raise ValueError(f"alpha must be in (0, 1], got {alpha}")

    # `(1 - kappa) * mean + kappa * CVaR`. The mean is the scenario average of
    # the same objective, so both terms land on the existing columns and the
    # whole vector is one expression rather than two passes over it.
    mean_cost = scenario_cost.mean(axis=0)
    col_cost_existing = (1.0 - kappa) * mean_cost

    # The master MINIMISES `cost . x` and a sale is a negative cost, so the
    # tail enters negated: minimising `-kappa * (t + mean(u))` maximises the
    # CVaR of the profit. Getting this sign wrong prices risk as a reward,
    # silently -- verified against the closed form on three distributions.
    # `t` earns -kappa, each `u_s` earns -kappa / (alpha * S): the tail is averaged
    # over `alpha * S` scenarios, so the weight per scenario is that quotient.
    tail_weight = kappa / (alpha * n_scen)
    col_cost = np.concatenate([[-kappa], np.full(n_scen, -tail_weight)])

    n_cols = cost.size
    rows = np.zeros((n_scen, n_cols + 1 + n_scen), dtype=np.float64)
    rows[:, :n_cols] = -scenario_cost      # + profit_s  ==  - scenario_cost . x
    rows[:, n_cols] = 1.0                  # + t  (the shortfall form)
    rows[np.arange(n_scen), n_cols + 1 + np.arange(n_scen)] = 1.0   # + u_s
    row_lower = np.zeros(n_scen)
    row_upper = np.full(n_scen, np.inf)
    return col_cost_existing, col_cost, rows, row_lower, row_upper


def cvar_of(profits: np.ndarray, alpha: float) -> float:
    """The closed form, for the guard: the mean of the worst `alpha` tail.

    Kept next to the LP form so a test can assert the two agree -- the LP's
    `t + mean(u)` at the optimum IS this number, and a block that disagrees with
    the closed form is wrong however plausible its rows look.
    """
    p = np.sort(np.asarray(profits, dtype=np.float64).ravel())
    n = p.size
    if n == 0:
        raise ValueError("no profits")
    a = float(alpha) * n
    k = int(np.floor(a))                     # whole scenarios in the tail
    tail = p[:k].sum() if k else 0.0
    frac = a - k                             # the partial one, weighted exactly
    if frac > 0.0 and k < n:
        tail += frac * p[k]
    return float(tail / a)
