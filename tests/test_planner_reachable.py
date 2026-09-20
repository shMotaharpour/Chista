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
    """A purse that binds before the hours do, and a master that reads it.

    On a real day-0 board the HOURS row binds first (15.6 for 25 tiles), so a
    season-shaped test would pass with the cash row deleted — which is what the
    first version of this guard did, and a guard that cannot fail is not a
    guard. So the hours are made abundant and the purse is made the scarce
    thing, which is the case the row exists for.

    This is also the case a scalar price cannot answer: 25 tiles in one graph
    state, so the DP offers ONE column and no per-resource price makes it offer
    two. What tells them apart is the allocation — λ — and the row that bounds
    it here is cash.
    """
    import numpy as np
    from agent.planner import master as M
    from agent.planner.inputs import dual_stand_in, load_contractor
    from offline_lab.kaggle_env import new_environment

    env = new_environment()
    env.reset(2)
    obs = env.state[0].observation
    days = 20
    base = M.supply_from_obs(obs)
    contractor = load_contractor(days=days)
    owned = M._owned_states(object(), obs)

    p_stand, w_stand = dual_stand_in(obs, days=days)
    board = contractor.price(p_stand, w_stand, owned)
    spend, _ = M.column_cash(board, base, days)
    purse = float(spend[:, 0].min()) * 2.5          # ~2 tiles' worth of day-0 spend
    rich_hours = M.CouplingSupply(
        hours=np.full(30, 10_000.0), seed_stock=base.seed_stock,
        animal_stock=base.animal_stock, fert_stock=base.fert_stock,
        wheat_feed_stock=base.wheat_feed_stock, money=purse, quotes=base.quotes)

    result = M.equilibrate(object(), obs, contractor, rich_hours)
    lam = np.asarray(result.lam, dtype=float)
    committed = float(lam[:-1].sum())

    assert result.objective > 0, "the master took only the idle column"
    assert committed < 25.0 - 1e-6, (
        f"with hours abundant and only {purse:.0f} coins, the master still "
        f"committed {committed:.2f} of 25 tiles — it is not reading the purse")
    # A binding purse has a positive shadow price: that IS the cash row, and
    # it is zero on every day if the row is not in the LP at all.
    #
    # The λ cannot be re-checked against a board priced here, because the
    # weights belong to the LAST ROUND's columns and `MasterResult` does not
    # carry them — re-pricing at the published `w` gives a different board and
    # a different spend. TODO(#79): the result has to hand back the columns it
    # weighted, or `columns.assign_tiles` cannot round the mix it was given.
    cash = np.asarray(result.cash_duals, dtype=float)
    assert cash.size and cash.max() > 0.0, (
        "no day has a positive shadow price on a coin, so the purse never "
        "bound — the cash row is missing from the LP")
