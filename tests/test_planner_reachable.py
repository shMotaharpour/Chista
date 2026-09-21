"""The planner is importable, and the master runs on a real board.

This suite exists because `agent/planner/master.py` — the Walrasian master of
#12, merged in #36 — spent days behind an ImportError and nobody noticed:

    planner/__init__ -> master -> agent.replan -> wsr.routing.plan_day

`plan_day` was removed when `routing.py` was rewritten, so the master,
`columns.py` and `land.py` were all unreachable while looking merged. A layer
that cannot be imported fails exactly like a layer that is not written, and the
only difference is how long it takes to find out.

R007: break the import chain (re-point `inputs.dual_stand_in` at `agent.replan`)
and watch the first test go red.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_the_planner_package_imports():
    """Every planner module, by name. An unreachable layer is an absent layer."""
    for name in ("agent.planner", "agent.planner.inputs", "agent.planner.master",
                 "agent.planner.columns", "agent.planner.land",
                 "agent.planner.repair"):
        importlib.import_module(name)


def test_the_master_does_not_reach_for_a_rung():
    """`planner/` may not IMPORT a rung or the engine.

    `agent.replan` pulls `market_layer` and a `plan_day` that no longer exists,
    and `kaggle_environments` is the dynamic engine read the whole `world/`
    transcription exists to replace — the submission ships `agent/` and cannot
    rely on the engine being importable, and a table read two ways is a table
    that can disagree with itself.

    Parsed, not grepped: a docstring that NAMES the forbidden module (this one
    does, and so does `inputs.py`) is a record of why the rule exists, not a
    breach of it.
    """
    import ast

    root = Path(__file__).resolve().parents[1] / "agent" / "planner"
    forbidden = ("agent.replan", "kaggle_environments")
    for path in sorted(root.glob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            for name in names:
                assert not any(name == f or name.startswith(f + ".")
                               for f in forbidden), (
                    f"{path.name}:{node.lineno} imports {name}")


def test_the_stand_in_duals_are_a_floor_and_non_negative():
    """#12's rule: a negative `w` breaks the shipped graph's dominance pruning."""
    from agent.planner.inputs import dual_stand_in
    from agent.world.model import N_RESOURCE, RESOURCE_ID
    from offline_lab.kaggle_env import new_environment

    env = new_environment()
    env.reset(2)
    p, w = dual_stand_in(env.state[0].observation, days=20)
    assert p.shape == (20, N_RESOURCE) and w.shape == (20, N_RESOURCE)
    assert (p >= 0).all() and (w >= 0).all()
    # The wage floor is the marginal hire over the 23 turns a hand actually
    # works (F040), not over 24, and never zero.
    assert w[0, RESOURCE_ID["LABOR"]] > 0


def test_the_master_runs_a_real_board_and_publishes_non_negative_duals():
    """One `equilibrate` on a day-0 board: it answers, and the duals are legal.

    Not an assertion that it CONVERGES — today it does not, and #79 is that
    work. What is asserted is the contract #12 states without qualification:
    every published dual is non-negative, in every round.
    """
    from agent.planner import master as M
    from agent.planner.inputs import load_contractor
    from offline_lab.kaggle_env import new_environment

    env = new_environment()
    env.reset(2)
    obs = env.state[0].observation
    result = M.equilibrate(object(), obs, load_contractor(days=20),
                           M.supply_from_obs(obs))

    assert result.rounds >= 1, "the master did not complete a single round"
    assert not result.used_fallback, f"master fell back: {result.fallback_reason}"
    assert (np.asarray(result.w) >= 0).all(), "a published wage is negative"
    assert (np.asarray(result.p) >= 0).all(), "a published price is negative"
    assert (np.asarray(result.duals) >= 0).all(), "a published dual is negative"


def test_the_master_prices_every_owned_tile_not_every_worker():
    """25 owned tiles, one farmer — the LP must see 25 columns, not one.

    `_owned_states` returned `unit_state_ids`, the state under each WORKER,
    while its docstring said "the tiles we own". On a day-0 board that is one
    farmer: the convexity row read `Σλ = 1`, the idle column took all of it,
    and the master's objective was 0 however long it ran.
    """
    from agent.planner import master as M
    from offline_lab.kaggle_env import new_environment

    env = new_environment()
    env.reset(2)
    obs = env.state[0].observation
    owned = M._owned_states(object(), obs)
    n_units = 1 + len(obs["farms"][int(obs.get("player", 0))].get("hands", []) or [])
    assert len(owned) == 25, f"day 0 owns 25 tiles, the master sees {len(owned)}"
    assert len(owned) > n_units, "the master is counting workers, not tiles"


def test_a_purchasable_input_is_not_capped_by_a_stock_of_zero():
    """Seeds, animals, wheat and fertiliser are bought (F001, F016), not endowed.

    Eight seed/animal rows and the wheat and fertiliser rows were bounded by the
    stock in the purse, which on a day-0 board is zero — so every column that
    plants or places violated a `≤ 0` row, the LP kept only the idle column, and
    the duals on those rows diverged (2.44e3 per hour after 8 rounds). Hours are
    the only thing the farm cannot buy, so hours are the only quantity row.
    """
    from agent.planner import master as M
    from agent.world.model import RESOURCE_NAMES

    rows = [RESOURCE_NAMES[i] for i in M.COUPLING_IDS]
    assert rows == ["LABOR"], f"quantity rows are {rows}; only LABOR may be one"
    for name in ("SEED_WHEAT", "ANIMAL_COW", "WHEAT", "FERTILIZER"):
        assert RESOURCE_NAMES.index(name) in M.PURCHASE_IDS, \
            f"{name} is purchasable and must be priced through the cash row"


def test_the_cash_row_stops_the_plan_buying_what_it_cannot_afford():
    """A smaller purse buys a smaller plan, and a coin has a price when it bites.

    Stated as a PROPERTY of the row, not as a tile count against a calibrated
    purse: the first version of this guard picked a purse from what the
    stand-in duals happened to price, and the master converges somewhere else,
    so it asserted a number that had stopped meaning anything. Monotonicity in
    money needs no calibration and cannot pass with the row deleted.
    """
    import numpy as np
    from agent.planner import master as M
    from agent.planner.inputs import load_contractor
    from offline_lab.kaggle_env import new_environment

    env = new_environment()
    env.reset(2)
    obs = env.state[0].observation
    contractor = load_contractor(days=20)
    base = M.supply_from_obs(obs)

    def with_money(coins):
        supply = M.CouplingSupply(
            hours=np.full(30, 10_000.0), seed_stock=base.seed_stock,
            animal_stock=base.animal_stock, fert_stock=base.fert_stock,
            wheat_feed_stock=base.wheat_feed_stock, money=float(coins),
            quotes=base.quotes)
        return M.equilibrate(object(), obs, contractor, supply, iter_cap=60)

    rich, poor = with_money(base.money), with_money(base.money / 50.0)

    assert rich.objective > poor.objective, (
        f"a purse 50x smaller bought the same plan ({rich.objective:.1f} vs "
        f"{poor.objective:.1f}) — the cash row is not binding anything")
    cash = np.asarray(poor.cash_duals, dtype=float)
    assert cash.size and cash.max() > 0.0, (
        "no day has a positive shadow price on a coin, so the purse never "
        "bound — the cash row is missing from the LP")


def test_the_master_accumulates_columns_and_mixes_them():
    """Many plans, combined — not one plan scaled.

    The loop used to re-price and REPLACE its column set every round, so the
    pool never held more than one plan per class and `lam` could only say how
    many tiles ran it. On a day-0 board of 25 identical tiles that is the whole
    difference between "25 cows or none" and a mix.
    """
    import numpy as np
    from agent.planner import master as M
    from agent.planner.inputs import load_contractor
    from offline_lab.kaggle_env import new_environment

    env = new_environment()
    env.reset(2)
    obs = env.state[0].observation
    result = M.equilibrate(object(), obs, load_contractor(days=20),
                           M.supply_from_obs(obs), iter_cap=40)

    assert not result.used_fallback, result.fallback_reason
    assert len(result.pool) > 3, (
        f"{len(result.pool)} columns after 40 rounds — the pool is not growing")
    lam = np.asarray(result.lam, dtype=float)
    assert int((lam > 1e-6).sum()) >= 3, (
        f"only {int((lam > 1e-6).sum())} columns carry weight: the master is "
        f"scaling one plan, not combining several")
    reps, counts, of_tile = result.classes
    assert len(of_tile) == 25 and int(counts.sum()) == 25


def test_the_certificate_is_reachable_on_a_real_board():
    """Given rounds, the pricing step proves there is nothing left to add.

    This is the whole claim of Dantzig-Wolfe and it is asserted rather than
    assumed: measured at 83 rounds and 740 ms on a day-0 board, so a cap of
    150 has headroom without making the test a benchmark.
    """
    from agent.planner import master as M
    from agent.planner.inputs import load_contractor
    from offline_lab.kaggle_env import new_environment

    env = new_environment()
    env.reset(2)
    obs = env.state[0].observation
    result = M.equilibrate(object(), obs, load_contractor(days=20),
                           M.supply_from_obs(obs), iter_cap=150)

    assert result.certified, (
        f"no certificate after {result.rounds} rounds: {result.stopped}")
    assert result.history and result.history[-1] <= M.colgen.RC_TOL, (
        f"certified with a positive reduced cost {result.history[-1]}")
    assert result.objective > 0


def test_the_optimum_does_not_depend_on_the_damping_constant():
    """`ALPHA` shapes what the agent READS, never what the loop proves.

    This test used to assert that damping reached the optimum in fewer rounds,
    which was true while the pricing step was fed damped duals — and that was
    the bug: the reduced-cost test `value + mu` is only a reduced cost of the
    LP the duals came from, so pricing at anything else stops it being about
    that LP. The stall detector caught it (rc 1597 on a column the pool already
    held, forever). Pricing now uses the master's own duals, and `ALPHA` only
    damps `result.w`, the published price the rest of the agent consumes.

    So the invariant is the stronger one: the certified optimum is the same
    number whatever `ALPHA` is. If it ever is not, damping has leaked back into
    the pricing path.
    """
    from agent.planner import master as M
    from agent.planner.inputs import load_contractor
    from offline_lab.kaggle_env import new_environment

    env = new_environment()
    env.reset(2)
    obs = env.state[0].observation
    contractor = load_contractor(days=20)
    supply = M.supply_from_obs(obs)

    original = M.ALPHA
    runs = {}
    try:
        for alpha in (0.3, 1.0):
            M.ALPHA = alpha
            runs[alpha] = M.equilibrate(object(), obs, contractor, supply,
                                        iter_cap=150)
    finally:
        M.ALPHA = original

    for alpha, res in runs.items():
        assert res.certified, f"ALPHA={alpha}: no certificate ({res.stopped})"
    assert runs[0.3].objective == pytest.approx(runs[1.0].objective, rel=1e-9), (
        f"the optimum moved with the damping constant: "
        f"{runs[0.3].objective:.6f} vs {runs[1.0].objective:.6f} — damping has "
        f"leaked into the pricing step")
    assert runs[0.3].rounds == runs[1.0].rounds


def test_every_good_the_DP_is_paid_for_is_a_good_the_master_counts():
    """The subproblem and the master must value the same plan the same way.

    FERTILIZER is a product the market quotes, and a tile that collects it was
    paid for it in `tile_values` and credited nothing in its column's revenue.
    The reduced cost then never reached zero — measured, the loop stalled 60
    short of a proof on a 52,279 objective, with the same column coming back
    round after round. With it counted the loop certifies in 47 rounds.
    """
    from agent.planner import master as M
    from agent.world.model import PRODUCTS, RESOURCE_NAMES

    priced = {RESOURCE_NAMES[i] for i in M.MARKET_IDS}
    assert priced == set(PRODUCTS), (
        f"the DP is paid for {sorted(set(PRODUCTS) - priced)} and the master "
        f"counts none of it")


def test_travel_is_charged_and_the_far_tiles_are_left_alone():
    """A tile is reached afresh every day it is worked, and the master knows.

    The farm is cleared every night and the farmer respawns on a shed door
    (F040), so a plan that works `v` days on a tile `d` steps out spends at
    least `v·d` hours walking there. It used to be nowhere in the model except
    a flat 35 % haircut on the supply, and that is not a rounding error:
    measured on a day-0 board, the assignment's travel alone was 307 hours
    against a budget of 15.6 hours a DAY.

    With the distance in the class key and the walk in the column's labour
    row, the answer says it plainly — the tiles the master commits are the
    ones near the shed.
    """
    import numpy as np
    from agent.planner import master as M
    from agent.planner.inputs import load_contractor
    from offline_lab.kaggle_env import new_environment

    env = new_environment()
    env.reset(2)
    obs = env.state[0].observation
    result = M.equilibrate(object(), obs, load_contractor(days=20),
                           M.supply_from_obs(obs), iter_cap=200)
    assert result.certified, result.stopped

    reps, _counts, _of_tile = result.classes
    assert all(isinstance(r, tuple) and len(r) == 2 for r in reps), (
        "a class must carry its distance, or its column cannot price the walk")
    assert len({d for _s, d in reps}) > 1, (
        "one distance band on a 25-tile quadrant: the classes are not banded")

    lam = np.asarray(result.lam, dtype=float)
    worked = {reps[c.cls][1] for j, c in enumerate(result.pool)
              if j < lam.size and lam[j] > 1e-6 and c.revenue > 0.0}
    assert worked, "nothing was committed at all"
    assert max(worked) <= min(d for _s, d in reps) + 4, (
        f"the master committed a tile {max(worked)} steps out while nearer "
        f"ones idled — travel is not reaching the labour row")


def test_the_labour_row_carries_the_walk_not_just_the_ops():
    """The column's hours on a worked day exceed the chain's own hours."""
    import numpy as np
    from agent.planner import master as M
    from agent.planner.inputs import load_contractor
    from offline_lab.kaggle_env import new_environment

    env = new_environment()
    env.reset(2)
    obs = env.state[0].observation
    result = M.equilibrate(object(), obs, load_contractor(days=20),
                           M.supply_from_obs(obs), iter_cap=40)
    reps, _c, _o = result.classes
    far = max(range(len(reps)), key=lambda c: reps[c][1])
    distance = reps[far][1]
    columns = [c for c in result.pool if c.cls == far and c.cost[:, 0].sum() > 0]
    assert columns, "the farthest class has no working column to check"
    for column in columns:
        worked = column.cost[:, 0] > 0
        assert (column.cost[worked, 0] >= distance).all(), (
            f"a worked day on a tile {distance} steps out costs less than the "
            f"walk to reach it")


def test_the_bound_the_master_reports_is_a_bound_on_a_real_board():
    """`L(y)` must never fall below the objective it bounds — on THIS board.

    The synthetic pricer in `test_colgen` cannot catch this: its classes never
    price a plan at a loss. The real one does. The tile DP chooses its chain
    before the travel term reaches the column, so a chain it liked can be a
    loss once the walk is paid for, and the class value went negative — bound
    −71,123 against an objective of 34,197, which is not a bound at all. A
    class is never worth less than its idle column, which is worth zero.
    """
    import numpy as np
    from agent.planner import master as M
    from agent.planner.inputs import load_contractor
    from offline_lab.kaggle_env import new_environment

    env = new_environment()
    env.reset(2)
    obs = env.state[0].observation
    result = M.equilibrate(object(), obs, load_contractor(days=20),
                           M.supply_from_obs(obs), iter_cap=200)

    assert result.certified, result.stopped
    assert np.isfinite(result.bound), "no bound was computed at all"
    assert result.bound >= result.objective - 1e-6, (
        f"the bound {result.bound:.1f} is below the objective "
        f"{result.objective:.1f}: it is not a bound")
    assert result.gap >= -1e-9
