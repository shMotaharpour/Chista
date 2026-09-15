"""Replanner: the day's plan, priced by the contractor (issue #11 §6).

The first heavy rung of the fallback ladder, and the socket `agent/runtime.py`
already had (`_rung_plan` dispatches `self.plan` when it is not None).

It runs **once per day**, at hour 0, and the spine dispatches the stored plan for
the remaining 23 hours. The graph is day-invariant, so within a day nothing the
DP sees changes except our own execution — replanning every turn would burn the
budget for nothing.

It **polls the published deadline** between steps and raises `TimeoutError`
mid-work. The ladder's between-rung gate can only see a rung that already spent
the turn; `self._deadline` exists so the rung itself can bail (the drill in
`tests/test_agent_runtime.py::test_deadline_gates_the_ladder` pins the contract,
this module is the real thing it was pinning).

## Two stand-ins, both named, both somebody else's issue

**The duals (#12).** The master that produces prices and wages does not exist,
so the sweep is handed the engine's own quotes: the observation's market price
for each product, the engine's seed and animal costs, and the price of an hour
from the engine's own hire rule (`_hire_cost`, F039/F040). They are held flat
across the horizon — F035 says prices rise through the season, and a stand-in
that forecasts them would be a model, which is exactly what #12 is for.

**The secretary (#14).** The contractor prices ONE TILE at a time; turning tile
chains into per-unit op lists — movement, pickups, market batching — is the
secretary's job and is not written. So each unit works the tile it already
stands on, and only the chain's own worker ops are dispatched: the purchases and
carries the chain assumes are missing, and the engine refuses an op whose input
the unit does not carry, **in silence** (F047).

That is why the rung is **off by default** (`CHISTA_REPLAN=1` turns it on). On
today's code greedy is the honest brain; this module is the socket, wired and
tested, waiting for the layer that can carry the inputs. Turning it on today
would replace a working policy with a plan whose purchases nobody carries —
a regression in play, not a step forward, and the arena has not been built yet
(#18/#20) to catch it.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from kaggle_environments.envs.kaggriculture import kaggriculture as K

from agent.obs import LOCKED_KEY, WorldView, _nearest_modelled, decode_world
from tile_dp.chains import (NO_ACT, N_RESOURCE, RESOURCE_ID, chain_ops,
                            entity_of_code)
from tile_dp.contractor import HORIZON_DAYS, TileContractor
from tile_dp.graph import TileGraph

from pathlib import Path

GRAPH_PATH = (Path(__file__).resolve().parents[1] / "tile_dp" / "models"
              / "graph_tile_lifecycle.npz")

# The engine's own hire cost is imported, never transcribed (R002): the n-th
# hire of a day costs `_hire_cost(n)`. A hand hired at hour 0 first acts at
# hour 1 (F040), so a hand buys 23 hours and that is what an hour costs.
HOURS_PER_HAND = 23

# Ops that name the entity a chain constructs, and therefore cannot be
# dispatched without one.
_ENTITY_OPS = frozenset(("PLANT", "BUILD", "PLACE", "PLACE_ANIMAL"))


def load_contractor() -> TileContractor:
    """The shipped graph, cast once per process — never rebuilt at runtime."""
    return TileContractor(TileGraph.load(GRAPH_PATH))


def dual_stand_in(obs: Any) -> tuple[np.ndarray, np.ndarray]:
    """The master's duals (#12), stood in for by the engine's own quotes.

    Every component has a source (R005): products are the observation's own
    market prices, seeds and animals are the engine tables, and the wage is the
    engine's marginal hire over the hours a hand actually works.
    """
    market = obs.get("market", {}) if isinstance(obs, dict) else {}
    prices = market.get("prices", {}) or {}
    farms = obs.get("farms", []) if isinstance(obs, dict) else []
    player = int(obs.get("player", 0)) if isinstance(obs, dict) else 0
    farm = farms[player] if len(farms) > player else {}

    p = np.zeros((HORIZON_DAYS, N_RESOURCE))
    for product in K.PRODUCTS:
        if product in RESOURCE_ID:
            p[:, RESOURCE_ID[product]] = float(prices.get(product, 0.0))

    w = np.zeros((HORIZON_DAYS, N_RESOURCE))
    for crop, spec in K.CROPS.items():
        w[:, RESOURCE_ID[f"SEED_{crop}"]] = float(spec["seed"])
    for species, spec in K.ANIMALS.items():
        w[:, RESOURCE_ID[f"ANIMAL_{species}"]] = float(spec["cost"])
    # FEED eats wheat and FERTILIZE eats fertilizer: what an animal's feed and a
    # fertilizer dose cost is the price they are bought back at.
    w[:, RESOURCE_ID["WHEAT"]] = float(prices.get("WHEAT", 0.0))
    w[:, RESOURCE_ID["FERTILIZER"]] = float(prices.get("FERTILIZER", 0.0))
    w[:, RESOURCE_ID["LABOR_HOURS"]] = (
        float(K._hire_cost(int(farm.get("hires_today", 0)))) / HOURS_PER_HAND)
    return p, w


def chain_turns(ops: tuple[str, ...], entity: str | None) -> list[list[str]]:
    """One chain -> the unit's op per turn, in canonical order.

    The worker side of the expansion `tile_dp/graph.py::_exec_chain` runs on a
    scratch sim at build time. The purchases that function also realises are
    deliberately absent here — see the module docstring: nothing carries them
    yet (#14).
    """
    turns: list[list[str]] = []
    for op in ops:
        if op in _ENTITY_OPS and entity is None:
            raise ValueError(
                f"chain {ops} runs {op} but names no entity: the graph should "
                "never carry such an edge")
        if op == NO_ACT or op == "PASS":
            turns.append(["PASS"])
        elif op == "BUILD":
            if entity not in K.ANIMALS:
                raise ValueError(f"BUILD {entity!r} is not an animal")
            turns.append([f"BUILD_{K.ANIMALS[entity]['structure']}"])
        elif op in ("PLACE", "PLACE_ANIMAL"):
            if entity not in K.ANIMALS:
                raise ValueError(f"PLACE {entity!r} is not an animal")
            turns.append(["PLACE", str(entity)])
        elif op == "PLANT":
            if entity not in K.CROPS:
                raise ValueError(f"PLANT {entity!r} is not a crop")
            turns.append(["PLANT", str(entity)])
        else:                      # WATER, HARVEST, DIG, FERTILIZE, FEED, CARE,
            turns.append([op])     # COLLECT_FERTILIZER - single worker ops
    return turns


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
        ids.append(int(state_id))
    return ids


def project_day(board, columns: list[int | None]) -> dict:
    """The priced chains -> the dispatcher's plan for this day.

    One unit per column, in `unit_positions` order; a unit with no priceable
    tile gets an empty list (it passes). Chains are one day long by contract
    (`ChainSpansDays`), so day 0 of the recovered plan is the day we are in.
    """
    units: list[list[list[str]]] = []
    for column in columns:
        if column is None:
            units.append([])
            continue
        _day, _state_id, chain_id = board.plans[column][0]
        entity = entity_of_code(int(board.per_day_entity[column, 0]))
        units.append(chain_turns(chain_ops(chain_id), entity))
    return {"units": units, "market": []}


def _poll(deadline) -> None:
    """The rung's own bail: the ladder only gates BETWEEN rungs."""
    if deadline is not None and deadline.expired():
        raise TimeoutError("replanner over budget: the plan rung bailed")


def replan_day(runtime, obs, graph: TileGraph | None = None,
               contractor: TileContractor | None = None) -> dict:
    """One day's plan: decode, price the board, project it. Polls at each step.

    `graph` / `contractor` are injectable for tests; in the runtime they are
    loaded once and cached on the runtime object.
    """
    deadline = getattr(runtime, "_deadline", None)
    _poll(deadline)
    if graph is None or contractor is None:
        cached = getattr(runtime, "_replan_resources", None)
        if cached is None:
            loaded = load_contractor()
            cached = (loaded.graph, loaded)
            runtime._replan_resources = cached
        graph = graph or cached[0]
        contractor = contractor or cached[1]
    _poll(deadline)
    # `at_day_start` asserts hour 0: a mid-day decode fed the DP the wrong state
    # once already (the pre-v15 rebuild bug, #10).
    view = decode_world(obs, at_day_start=True,
                        graph_keys=frozenset(graph.key_index))
    _poll(deadline)
    p, w = dual_stand_in(obs)
    _poll(deadline)
    state_ids = unit_state_ids(view, graph)
    owned = [state_id for state_id in state_ids if state_id is not None]
    board = contractor.price(p, w, owned)
    _poll(deadline)
    # map each unit back onto its column in `owned`
    columns: list[int | None] = []
    column = 0
    for state_id in state_ids:
        if state_id is None:
            columns.append(None)
        else:
            columns.append(column)
            column += 1
    return project_day(board, columns)


def enable(runtime) -> None:
    """Turn the rung on for a runtime object (the tests' handle on it)."""
    runtime.replanner = replan_day
