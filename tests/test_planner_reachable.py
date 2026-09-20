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
