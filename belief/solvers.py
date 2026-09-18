"""Phase 3 — the two solvers, and the matrices they are handed.

`linprog(method="highs")` where the problem relaxes to an LP/MILP; `minimize(
method="SLSQP")` where it does not. Nothing here samples, and nothing here needs
a GPU: the platform probe (F058) put `scipy 1.15.3` with `milp` on the grader,
`ortools` off it, and measured SciPy-shaped work at 4.27 ms solo versus 6.01 ms
when the other seat is thinking at the same time (+41 %), so the per-turn budget
carries a 1.4x multiplier.

Three pieces of structure, each with a reason:

* **The slot game.** One good's lot, sold over the turns of a day, against the
  rival's schedule for the same good. Same-index quotes are equal and the ladder
  is walked unit by unit, so whoever sits at the lower index takes the better
  quote and pushes the price down for the other. The maximin mix of that matrix
  is an LP, and for a two-player zero-sum matrix game that mix *is* the Nash
  equilibrium — a security level, which is the guarantee worth having. A Nash
  solve of the whole 30-day game would not give one.

* **The risk-adjusted sibling**, for the continuum the three discrete schedules
  cannot express: split fractions with a penalty on the worst-scenario shortfall,
  which is non-linear, hence SLSQP.

* **The season LP.** Sell quantities per good per day plus a tile mix, with the
  price linear at the neutral inventory and one row per good capping the season's
  sales at `drain x worst-case rival share` — the **absorption row**. Without it
  the same LP prints 188M coins, because it happily sells 4,000 units into a
  525-unit drain; with it, the answer is 36,702 on the five-good demo. The tile
  mix is relaxed here and rounded by `round_tiles`; `scipy.optimize.milp` is on
  the grader and is the natural upgrade when the integrality gap is measured.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.optimize import linprog, minimize

from agent.world.prices import price_of
from agent.world.vocabulary import SHED_CAP


def maximin_mixed_lp(A: np.ndarray) -> tuple[np.ndarray, float]:
    """Exact maximin mix of the zero-sum game `max_p min_j (p^T A)[j]`.

    Variables `[p, v]`; rows `-A^T p + v <= 0`, `sum p = 1`, `p >= 0`. The value
    returned is the security level: the coins guaranteed against any schedule the
    rival picks inside the modelled matrix.
    """
    n_ours, n_theirs = A.shape
    c = np.zeros(n_ours + 1)
    c[-1] = -1.0
    A_ub = np.hstack([-A.T, np.ones((n_theirs, 1))])
    A_eq = np.zeros((1, n_ours + 1))
    A_eq[0, :n_ours] = 1.0
    res = linprog(c, A_ub=A_ub, b_ub=np.zeros(n_theirs), A_eq=A_eq,
                  b_eq=np.array([1.0]), bounds=[(0.0, 1.0)] * n_ours + [(None, None)],
                  method="highs")
    if not res.success:
        raise RuntimeError(f"maximin LP failed: {res.message}")
    p = np.clip(res.x[:n_ours], 0.0, None)
    return p / p.sum(), float(res.x[-1])


def maximin_mixed_slsqp(A: np.ndarray, risk_lambda: float = 0.35,
                        scenario_weights: np.ndarray | None = None
                        ) -> tuple[np.ndarray, float, float]:
    """Risk-adjusted continuous sibling: returns (mix, worst case, mean).

    Variables are split fractions over the turns (a continuum no list of three
    discrete schedules can hold). The non-linear term is the shortfall penalty,
    which is why this goes to SLSQP; the equality and the bounds go in as
    constraints, never as penalties.
    """
    n_ours, n_theirs = A.shape
    w = (np.full(n_theirs, 1.0 / n_theirs) if scenario_weights is None
         else np.asarray(scenario_weights, dtype=float))
    w = w / w.sum()

    def objective(x: np.ndarray) -> float:
        per_scenario = A.T @ x
        mean = float(w @ per_scenario)
        return -(mean - risk_lambda * float(max(0.0, mean - per_scenario.min())))

    def gradient(x: np.ndarray) -> np.ndarray:
        """Exact gradient of `-((1-lambda)·mean + lambda·min)`.

        The first version carried only the mean term, which is right only at
        `risk_lambda = 0`: with the risk weight on, SLSQP was handed a gradient
        that ignored the worst scenario and stopped short of the maximin answer
        (`tests/test_market_analyzer.py::test_the_two_solvers_agree_on_the_discrete_game`
        caught it, 2145.75 against the LP's 2189.0). The min term's subgradient is
        the column that is currently worst.
        """
        per_scenario = A.T @ x
        worst_column = int(np.argmin(per_scenario))
        return -((1.0 - risk_lambda) * (A @ w) + risk_lambda * A[:, worst_column])

    res = minimize(objective, np.full(n_ours, 1.0 / n_ours), jac=gradient,
                   method="SLSQP", bounds=[(0.0, 1.0)] * n_ours,
                   constraints=({"type": "eq", "fun": lambda x: float(np.sum(x) - 1.0),
                                 "jac": lambda x: np.ones_like(x)},),
                   options={"maxiter": 200, "ftol": 1e-10})
    if not res.success:
        raise RuntimeError(f"SLSQP failed: {res.message}")
    x = np.clip(res.x, 0.0, None)
    x = x / x.sum()
    per_scenario = A.T @ x
    return x, float(per_scenario.min()), float(per_scenario.mean())


def slot_game_matrix(good: str, inventory: float, our_schedules: np.ndarray,
                     rival_schedules: np.ndarray, drain_per_turn: float = 1.0,
                     turns: int = 24) -> np.ndarray:
    """Payoff matrix (our schedules x rival schedules), engine-faithful.

    Inside a turn the ladder is walked one unit at a time, each unit quoted at the
    inventory before either player commits; between turns the town drains, which
    is what makes spreading a lot across turns worth coins.
    """
    A = np.zeros((len(our_schedules), len(rival_schedules)))
    for i, ours in enumerate(our_schedules):
        for j, theirs in enumerate(rival_schedules):
            inv = float(inventory)
            revenue = 0.0
            for t in range(turns):
                ours_left = float(ours[t]) if t < len(ours) else 0.0
                theirs_left = float(theirs[t]) if t < len(theirs) else 0.0
                while ours_left > 0 or theirs_left > 0:
                    quote = price_of(good, max(0.0, inv))
                    if ours_left > 0:
                        revenue += quote
                        inv += 1.0
                        ours_left -= 1.0
                    if theirs_left > 0:
                        inv += 1.0
                        theirs_left -= 1.0
                inv = max(0.0, inv - drain_per_turn)
            A[i, j] = revenue
    return A


def default_schedules(lot: float, turns: int = 3) -> np.ndarray:
    """Dump now, halve, thirds, hold — the four shapes a seller actually weighs."""
    return np.array([[lot, 0.0, 0.0],
                     [lot / 2, lot / 2, 0.0],
                     [lot / 3, lot / 3, lot / 3],
                     [0.0, 0.0, lot]])


def season_plan_maximin(goods: Sequence[str], days: int,
                        units_per_tile_day: np.ndarray, tile_budget: float,
                        prices0: np.ndarray, scenarios: np.ndarray,
                        drain_per_day: np.ndarray) -> dict:
    """Maximin season allocation as an LP (`linprog(method="highs")`).

    `scenarios` are rival behaviour columns; the objective uses the worst case
    over them, so the plan does not depend on a forecast of their mood. The
    absorption row is what keeps the answer honest (see the module docstring).
    """
    G = len(goods)
    nvar = G * days + G
    worst_col = scenarios.min(axis=0) if scenarios.ndim == 2 else np.ones(G)
    absorb = np.asarray(drain_per_day, dtype=float)
    if absorb.ndim == 1:
        absorb = np.tile(absorb, (days, 1)).T

    c = np.zeros(nvar)
    for gi in range(G):
        for d in range(days):
            c[gi * days + d] = -float(prices0[gi]) * float(worst_col[gi])

    A_ub, b_ub = [], []
    row = np.zeros(nvar)
    row[G * days:] = 1.0
    A_ub.append(row)
    b_ub.append(tile_budget)
    for gi in range(G):                          # tiles bound the units they can grow
        row = np.zeros(nvar)
        row[G * days + gi] = -float(units_per_tile_day[gi]) * days
        row[gi * days:(gi + 1) * days] = 1.0
        A_ub.append(row)
        b_ub.append(0.0)
    for gi in range(G):                          # absorption: the town is the buyer
        row = np.zeros(nvar)
        row[gi * days:(gi + 1) * days] = 1.0
        A_ub.append(row)
        b_ub.append(float(absorb[gi].sum() * worst_col[gi]))
    for d in range(days):                        # the shed caps one day's sales
        row = np.zeros(nvar)
        for gi in range(G):
            row[gi * days + d] = 1.0
        A_ub.append(row)
        b_ub.append(float(SHED_CAP))

    res = linprog(c, A_ub=np.array(A_ub), b_ub=np.array(b_ub),
                  bounds=[(0.0, None)] * nvar, method="highs")
    if not res.success:
        raise RuntimeError(f"season LP failed: {res.message}")
    return dict(objective=float(-res.fun),
                units=res.x[:G * days].reshape(G, days),
                tiles=res.x[G * days:],
                status=int(res.status))


def round_tiles(tiles: np.ndarray) -> np.ndarray:
    """Greedy rounding of the relaxed tile mix (largest fractional part first)."""
    base = np.floor(tiles)
    order = np.argsort(-(tiles - base))
    out = base.astype(int).copy()
    budget = int(round(float(tiles.sum())))
    for i in order:
        if out.sum() >= budget:
            break
        out[i] += 1
    return out
