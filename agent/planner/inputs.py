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

GRAPH_PATH = artifact_path("tile_graph", ".npz")


def load_contractor(days: int = HORIZON_DAYS) -> TileContractor:
    """The shipped graph, cast once per process — never rebuilt at runtime."""
    return TileContractor(with_idle_edges(TileGraph.load(GRAPH_PATH)), days=days)


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
    w[:, RESOURCE_ID["LABOR"]] = (
        float(hire_cost(int(farm.get("hires_today", 0)))) / HOURS_PER_HAND)
    return p, w


def with_idle_edges(graph: TileGraph) -> TileGraph:
    """Every state gets the option of doing nothing, and the day passing.

    Four states in the shipped artifact have no empty-chain edge — `NONE`,
    `EMPTY_COOP`, `EMPTY_PASTURE` and `WEED` — and they are exactly the four
    whose night changes nothing. Zero of the 699 idle edges that DO exist are
    self-loops, so the builder appears to drop an edge that leads back where it
    started, and these are the only states for which it would.

    The consequence was not cosmetic. Every owned tile starts in `NONE` and a
    `DIG` returns it there, so **every tile was forced to act every day**. At
    `w_labour = 100` the DP returned a plan worth −30 and at 500 one worth
    −5,130, while doing nothing is worth exactly 0: it was not choosing a loss,
    it had no alternative. The master could never price work down to nothing —
    only choose which loss — and BUILD → DIG → BUILD was the graph speaking,
    not the plan.

    So a self-loop is added for each: chain 0, zero cost, zero produce, zero
    steps. Doing nothing on bare ground leaves bare ground.

    This belongs in the artifact builder and is filed there (#84). It is here
    because the agent must not ship a model in which idling is illegal, and
    because doing it at load leaves the artifact byte-identical — the fix and
    the thing it fixes stay visible to each other.
    """
    import dataclasses

    offsets = np.asarray(graph.edge_offsets)
    chain = np.asarray(graph.edge_chain)
    missing = [s for s in range(int(graph.n_states))
               if not (chain[int(offsets[s]):int(offsets[s + 1])] == 0).any()]
    if not missing:
        return graph

    nxt, ent = np.asarray(graph.edge_next), np.asarray(graph.edge_entity)
    cost, produce = np.asarray(graph.edge_cost), np.asarray(graph.edge_produce)
    steps = np.asarray(graph.edge_steps)
    need = set(missing)

    new_next, new_chain, new_entity = [], [], []
    new_cost, new_produce, new_steps = [], [], []
    new_offsets = np.zeros_like(offsets)
    for s in range(int(graph.n_states)):
        lo, hi = int(offsets[s]), int(offsets[s + 1])
        new_offsets[s] = len(new_chain)
        new_next.extend(nxt[lo:hi]); new_chain.extend(chain[lo:hi])
        new_entity.extend(ent[lo:hi]); new_steps.extend(steps[lo:hi])
        new_cost.extend(cost[lo:hi]); new_produce.extend(produce[lo:hi])
        if s in need:
            new_next.append(s)          # the night leaves it where it is
            new_chain.append(0)         # the empty chain
            new_entity.append(0)        # constructs nothing
            new_steps.append(0)
            new_cost.append(np.zeros(cost.shape[1], dtype=cost.dtype))
            new_produce.append(np.zeros(produce.shape[1], dtype=produce.dtype))
    new_offsets[-1] = len(new_chain)

    return dataclasses.replace(
        graph,
        edge_offsets=new_offsets,
        edge_next=np.asarray(new_next, dtype=nxt.dtype),
        edge_chain=np.asarray(new_chain, dtype=chain.dtype),
        edge_entity=np.asarray(new_entity, dtype=ent.dtype),
        edge_steps=np.asarray(new_steps, dtype=steps.dtype),
        edge_cost=np.asarray(new_cost, dtype=cost.dtype),
        edge_produce=np.asarray(new_produce, dtype=produce.dtype))
