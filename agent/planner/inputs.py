"""What the master needs from the board, and from nothing else.

These four helpers used to live in `agent/replan.py`, which meant
`planner/__init__` -> `master` -> `replan` -> `wsr.routing.plan_day` — a name
that `routing.py`'s rewrite removed. So the master, `columns.py` and `land.py`
have all been behind an ImportError: built, merged, and unreachable. Nothing
here imports a rung, and nothing may.

`dual_stand_in` is also re-expressed over `agent/world/`. Its old home read the
engine through `kaggle_environments.envs.kaggriculture` at import time, which is
the dynamic read the whole `world/` transcription exists to avoid — the
submission ships `agent/` and cannot rely on the engine being importable, and a
table read two ways is a table that can disagree with itself.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from agent.artifact import artifact_path
from agent.obs import LOCKED_KEY, WorldView, _nearest_modelled
from agent.tile_dp.contractor import HORIZON_DAYS, TileContractor
from agent.tile_dp.graph import TileGraph
from agent.world.model import N_RESOURCE, PRODUCTS, RESOURCE_ID
from agent.world.rules import ANIMAL_RULES, CROP_RULES, hire_cost

#: A hand hired in turn 0 first acts at hour 1, so it works 23 of the day's 24
#: turns (F040). The marginal wage is the hire's price over the hours it buys,
#: not over the day.
HOURS_PER_HAND = 23

#: The floor price of the FARMER's own hour (#87 item 1). The old floor —
#: the marginal hand's 1-coin hire over 23 hours, 0.043/h — made destroying
#: and rebuilding a pasture free, because at that price nothing the farm
#: owns is worth preserving. The farmer's hour is what a chain first
#: consumes, and its honest floor is what one working day of his produces
#: for the farm's own pipeline: one MILK (the cheapest animal product,
#: ~160 at season quotes) over the 24·(1−0.35) ≈ 15.6 working hours the
#: overhead model grants. ~10.3/h. Not the answer — the floor the master's
#: tâtonnement starts from.
FARMER_HOUR_FLOOR: float = 160.0 / (24.0 * (1.0 - 0.35))

GRAPH_PATH = artifact_path("tile_graph", ".npz")


def load_contractor(days: int = HORIZON_DAYS) -> TileContractor:
    """The shipped graph, cast once per process — never rebuilt at runtime."""
    return TileContractor(TileGraph.load(GRAPH_PATH), days=days)


def unit_positions(view: WorldView) -> list[tuple[int, int]]:
    """The farmer first, then the hands in `hands` order (F030)."""
    positions = [(int(view.me.farmer[0]), int(view.me.farmer[1]))]
    positions += [(int(x), int(y)) for x, y in view.me.hands]
    return positions


def unit_state_ids(view: WorldView, graph: TileGraph) -> list[int | None]:
    """The graph state under each unit, or None where there is nothing to price.

    A LOCKED tile (F042) has no state, and a key the graph does not model falls
    back to its nearest neighbour exactly as `agent/obs.py` does for the
    classes — the same rule, so the pricing and the counts cannot disagree.
    """
    keys = view.me.keys
    known = frozenset(graph.key_index)
    ids: list[int | None] = []
    for x, y in unit_positions(view):
        key = int(keys[y][x])
        if key == LOCKED_KEY:
            ids.append(None)
            continue
        state_id = graph.key_index.get(key)
        if state_id is None:
            state_id = _nearest_modelled(key, known)
        ids.append(None if state_id is None else int(state_id))
    return ids


def dual_stand_in(obs: Any, days: int = HORIZON_DAYS) -> tuple[np.ndarray, np.ndarray]:
    """The master's duals (#12), stood in for by the engine's own quotes.

    Every component has a source (R005): products are the observation's own
    market prices, seeds and animals are the engine tables as `agent/world/`
    transcribes them, and the wage is the marginal hire over the hours a hand
    actually works.

    This is a FLOOR, not an answer. The master's job is to move the internal
    prices off it by tâtonnement until the coupling rows clear; publishing this
    unchanged is the price vector of a farm where nothing is scarce, which is
    how 25 identical tiles all price the same chain.
    """
    market = obs.get("market", {}) if isinstance(obs, dict) else {}
    prices = market.get("prices", {}) or {}
    farms = obs.get("farms", []) if isinstance(obs, dict) else []
    player = int(obs.get("player", 0)) if isinstance(obs, dict) else 0
    farm = farms[player] if len(farms) > player else {}

    p = np.zeros((days, N_RESOURCE))
    for product in PRODUCTS:
        p[:, RESOURCE_ID[product]] = float(prices.get(product, 0.0))

    w = np.zeros((days, N_RESOURCE))
    for crop, spec in CROP_RULES.items():
        w[:, RESOURCE_ID[f"SEED_{crop}"]] = float(spec["seed"])
    for species, spec in ANIMAL_RULES.items():
        w[:, RESOURCE_ID[f"ANIMAL_{species}"]] = float(spec["cost"])
    # FEED eats wheat and FERTILIZE eats fertilizer: what an animal's feed and a
    # fertilizer dose cost is the price they are bought back at.
    w[:, RESOURCE_ID["WHEAT"]] = float(prices.get("WHEAT", 0.0))
    w[:, RESOURCE_ID["FERTILIZER"]] = float(prices.get("FERTILIZER", 0.0))
    # The farmer's own hour is the farm's real first labour unit: he stands
    # on the field at no marginal hire, and his 23 working hours (F040) are
    # what a chain first consumes. Pricing his hour at the marginal hand's
    # 1-coin hire (0.043/h) made tearing up and rebuilding a pasture free —
    # the replay behaviour #87 opens with. The farmer's hour is priced at
    # the value of what his hour produces for the farm's OWN pipeline: one
    # MILK (~160 at the season's quotes) over the ~15.6 working hours the
    # overhead model gives, floored at the marginal hand so hiring never
    # looks cheaper than the man already there. This is the FLOOR, not the
    # answer: the master's tâtonnement moves the internal prices off it.
    w[:, RESOURCE_ID["LABOR"]] = np.maximum(
        float(hire_cost(int(farm.get("hires_today", 0)))) / HOURS_PER_HAND,
        FARMER_HOUR_FLOOR)
    return p, w
