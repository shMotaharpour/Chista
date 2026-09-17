"""The canonical vocabulary: the engine's names, in one module, for every layer.

See `docs/ARCHITECTURE.md` §1. Three vocabularies grew for this one world — the
engine's action strings, the tile DP's chain ops (`tile_dp/chains.py`), and the
WSR secretary's enums (`secretary/models.py`) — and every consumer paid for the
divergence (the DP's market ops looked incomplete, `PLACE` existed twice, the
secretary's `PRODUCT_ITEMS` was missing every crop). This module ends the
duplication by **deriving** from the engine (R002) instead of transcribing:

- `GOODS`, `CROPS`, `ANIMALS` are the engine's own tuples;
- `ACTIONS` is what the engine's action handlers accept, **extracted from their
  source** (`engine_action_names`), so a new engine op appears here without anyone
  editing a list;
- `RESOURCES` is the DP's 18 columns, derived from the three above;
- `CHAIN_OPS` is the DP's abstract chain vocabulary, and `COMPILE` is the single
  abstract -> engine table (what each chain op means as engine actions, and what
  it must carry or buy first).

Nothing in this module makes a decision, reads an observation, or imports a layer.
"""

from __future__ import annotations

import inspect
import re
from typing import Mapping

from kaggle_environments.envs.kaggriculture import kaggriculture as K

# --------------------------------------------------------------------------- #
# what exists: the engine's own tables
# --------------------------------------------------------------------------- #

GOODS: tuple[str, ...] = tuple(str(g) for g in K.PRODUCTS)          # 9
CROPS: tuple[str, ...] = tuple(str(c) for c in K.CROPS)             # 5
ANIMALS: tuple[str, ...] = tuple(str(a) for a in K.ANIMALS)         # 3

#: The DP's resource columns, in the order the shipped graph stores them:
#: labour, fertiliser, wheat, the five seeds, the four crop products, the three
#: animal products, the three animals.
RESOURCES: tuple[str, ...] = (
    "LABOR_HOURS", "FERTILIZER", "WHEAT",
    *(f"SEED_{crop}" for crop in CROPS),
    "CARROT", "TOMATO", "STRAWBERRY", "MELON",
    "EGG", "MILK", "WOOL",
    *(f"ANIMAL_{species}" for species in ANIMALS),
)
RESOURCE_ID: dict[str, int] = {name: i for i, name in enumerate(RESOURCES)}

#: What the market sells and buys: the nine goods.
SELLABLE: tuple[str, ...] = GOODS

# --------------------------------------------------------------------------- #
# what can be done: extracted from the engine's handlers
# --------------------------------------------------------------------------- #

#: The handlers that receive an action: the worker's list, the market's committer,
#: and the order parser the market queue walks.
_HANDLER_NAMES = ("_apply_unit_action", "_commit_unit", "_parse_order")


def engine_action_names() -> frozenset[str]:
    """Every action string the engine's own handlers test for.

    Extracted from their source rather than typed out: `if op == "PLANT":`,
    `elif op in (...)` and the movement table. A name in this set is an action the
    engine will act on; anything else is a silent no-op (F047), which is the whole
    reason this module exists.
    """
    names: set[str] = set()
    for handler in _HANDLER_NAMES:
        source = inspect.getsource(getattr(K, handler))
        names.update(re.findall(r'op == "([A-Z_]+)"', source))
        for group in re.findall(r"op in \(([^)]*)\)", source):
            names.update(re.findall(r'"([A-Z_]+)"', group))
    names.update(str(move) for move in K.FARMER_MOVES)      # NORTH, SOUTH, EAST, WEST
    names.add("PASS")                                       # `if op == "PASS": return`
    return frozenset(names)


#: The engine's action vocabulary, as the engine itself defines it.
ACTIONS: frozenset[str] = engine_action_names()

#: Movement, the only actions that change a unit's position (one tile each).
MOVEMENT: tuple[str, ...] = tuple(str(move) for move in K.FARMER_MOVES)

#: The shed's four access tiles: PICKUP and DROP work nowhere else (`:344, 359`).
SHED_ACCESS: tuple[tuple[int, int], ...] = tuple(
    (int(x), int(y)) for x, y in K._shed_access_tiles(10))

# --------------------------------------------------------------------------- #
# what a day is made of: the DP's chain vocabulary, and the one compile table
# --------------------------------------------------------------------------- #

#: The DP's chain ops. `NO_ACT` is a whole-chain day-pass; the `BUY_*` ops are the
#: market's actions that a chain names when it needs an input (never a worker op).
#: `PLACE_ANIMAL` is the DP's legacy alias for `PLACE` (the duplicate the owner
#: flagged in review). It stays in the vocabulary because the shipped graph stores
#: chain ids on disk, and it compiles to the same action; `docs/ARCHITECTURE.md` §5
#: step 5 retires it.
CHAIN_OPS: frozenset[str] = frozenset((
    "PLANT", "WATER", "FERTILIZE", "HARVEST", "DIG", "BUILD", "PLACE",
    "PLACE_ANIMAL", "FEED", "CARE", "COLLECT_FERTILIZER", "NO_ACT",
    "BUY_SEED", "BUY_PRODUCT", "BUY_ANIMAL",
))

#: Chain ops a worker spends a turn on.
WORKER_OPS: frozenset[str] = frozenset((
    "PLANT", "WATER", "FERTILIZE", "HARVEST", "DIG", "BUILD", "PLACE", "FEED",
    "CARE", "COLLECT_FERTILIZER",
))

#: Chain ops the MARKET performs: one turn of a worker's day, never a purchase.
MARKET_OPS: frozenset[str] = frozenset(("BUY_SEED", "BUY_PRODUCT", "BUY_ANIMAL"))

#: What an op must have in the unit's bag before it can run (engine handlers).
CARRIES: Mapping[str, str] = {"FERTILIZE": "FERTILIZER", "FEED": "WHEAT"}

#: Ops that need the entity itself carried (an animal, `PLACE` at `:377-392`).
PLACING_OPS: frozenset[str] = frozenset(("PLACE", "PLACE_ANIMAL"))

#: Ops that need a seed in `private["seeds"]` (which never travels, `:367-368`).
SEED_OPS: frozenset[str] = frozenset(("PLANT",))

#: Ops that name the entity they construct, and so cannot run without one.
ENTITY_OPS: frozenset[str] = frozenset(("PLANT", "BUILD", "PLACE", "PLACE_ANIMAL"))


def compile_op(op: str, entity: str | None = None) -> tuple[str, ...]:
    """One chain op -> the engine action it means (the single translation).

    `BUILD` becomes the structure the entity needs, `PLACE`/`PLACE_ANIMAL` names
    the animal, `NO_ACT` is a `PASS`. Raises on an op that is not in `CHAIN_OPS`,
    because an unknown op silently does nothing on the engine (F047) and a plan
    must never emit one.
    """
    if op not in CHAIN_OPS:
        raise ValueError(f"{op!r} is not a chain op: {sorted(CHAIN_OPS)}")
    if op == "BUILD":
        if entity not in K.ANIMALS:
            raise ValueError(f"BUILD {entity!r} is not an animal")
        return (f"BUILD_{K.ANIMALS[entity]['structure']}",)
    if op in PLACING_OPS:
        if entity not in K.ANIMALS:
            raise ValueError(f"{op} {entity!r} is not an animal")
        return ("PLACE", str(entity))
    if op == "PLANT":
        if entity not in K.CROPS:
            raise ValueError(f"PLANT {entity!r} is not a crop")
        return ("PLANT", str(entity))
    if op == "NO_ACT":
        return ("PASS",)
    return (str(op),)


def compile_chain(ops: tuple[str, ...], entity: str | None = None
                  ) -> list[tuple[str, ...]]:
    """A chain -> the worker actions of one day, in order, one per turn.

    This is the **only** expansion: the build-time executor
    (`tile_dp/graph.py::_exec_chain`) realises a chain on a scratch sim, the day
    compiler (`secretary/routing.py`) realises it on the live board, and both must
    agree on what the chain means. `NO_ACT` is a whole-chain op.
    """
    if "NO_ACT" in ops and len(ops) > 1:
        raise ValueError(f"NO_ACT is a whole-chain op, got {ops}")
    return [compile_op(op, entity) for op in ops]
