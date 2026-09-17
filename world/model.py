"""Canonical names for the whole system, derived from the engine (R002).

See `docs/ARCHITECTURE.md` §1. Nothing here is typed by hand: the goods, crops
and species are the engine's tables, and the action set is extracted from the
engine's own handlers.
"""

from __future__ import annotations

import inspect
import re
from enum import Enum
from typing import Mapping

from kaggle_environments.envs.kaggriculture import kaggriculture as K

# --- what exists ------------------------------------------------------------ #

GOODS: tuple[str, ...] = tuple(str(g) for g in K.PRODUCTS)      # 9, all sellable
PRODUCTS: tuple[str, ...] = GOODS                               # the market's goods
CROPS: tuple[str, ...] = tuple(str(c) for c in K.CROPS)         # 5
ANIMALS: tuple[str, ...] = tuple(str(a) for a in K.ANIMALS)     # 3

#: Wheat and fertiliser are both: the farm buys them and the market sells them.
DUAL: tuple[str, ...] = ("WHEAT", "FERTILIZER")

#: Resources: what the farm buys or consumes. 11 — labour, fertiliser, wheat,
#: the five seeds, the three animals. Crops and animal products are NOT resources.
RESOURCES: tuple[str, ...] = (
    "LABOR_HOURS", "FERTILIZER", "WHEAT",
    *(f"SEED_{crop}" for crop in CROPS),
    *(f"ANIMAL_{species}" for species in ANIMALS),
)

#: The DP graph's 18 columns = resources + products, in the order the shipped
#: graph stores them (11 + 9 − 2 shared = 18).
VECTOR: tuple[str, ...] = (
    "LABOR_HOURS", "FERTILIZER", "WHEAT",
    *(f"SEED_{crop}" for crop in CROPS),
    "CARROT", "TOMATO", "STRAWBERRY", "MELON",
    "EGG", "MILK", "WOOL",
    *(f"ANIMAL_{species}" for species in ANIMALS),
)

# --- what can be done ------------------------------------------------------- #

#: Handlers whose source defines the action vocabulary.
_HANDLERS = ("_apply_unit_action", "_commit_unit", "_parse_order")


def engine_action_names() -> frozenset[str]:
    """Every action string the engine's handlers test for, read from their source."""
    names: set[str] = set()
    for handler in _HANDLERS:
        source = inspect.getsource(getattr(K, handler))
        names.update(re.findall(r'op == "([A-Z_]+)"', source))
        for group in re.findall(r"op in \(([^)]*)\)", source):
            names.update(re.findall(r'"([A-Z_]+)"', group))
    names.update(str(move) for move in K.FARMER_MOVES)
    names.add("PASS")
    return frozenset(names)


#: 24 actions today.
ACTIONS: frozenset[str] = engine_action_names()

#: One tile per op, in the engine's own table.
MOVEMENT: tuple[str, ...] = tuple(str(move) for move in K.FARMER_MOVES)
MOVE_DELTA: Mapping[str, tuple[int, int]] = {str(k): (int(v[0]), int(v[1]))
                                             for k, v in K.FARMER_MOVES.items()}

#: The four shed-access tiles: PICKUP and DROP work nowhere else.
SHED_ACCESS: frozenset[tuple[int, int]] = frozenset(
    (int(x), int(y)) for x, y in K._shed_access_tiles(10))

#: Market actions: the market layer's, never a worker's. `SELL`, `BUY_LAND` and
#: `HIRE` are the market's own decisions; the `BUY_*` three are also what a chain
#: names when it needs an input.
MARKET_ACTIONS: frozenset[str] = frozenset((
    "SELL", "BUY_SEED", "BUY_PRODUCT", "BUY_ANIMAL", "HIRE", "BUY_LAND",
))

#: Worker ops: what a unit spends a turn on. No market op appears here — the
#: contractor and the WSR do not touch the market.
WORKER_OPS: frozenset[str] = frozenset((
    "PLANT", "WATER", "FERTILIZE", "HARVEST", "DIG", "BUILD", "PLACE", "FEED",
    "CARE", "COLLECT_FERTILIZER",
))

#: Chain ops a chain may name: worker ops + the market buys it needs + the day pass.
CHAIN_OPS: frozenset[str] = WORKER_OPS | frozenset(
    ("BUY_SEED", "BUY_PRODUCT", "BUY_ANIMAL", "NO_ACT"))

#: The DP's duplicate, kept for the shipped graph's on-disk chains.
LEGACY_ALIASES: Mapping[str, str] = {"PLACE_ANIMAL": "PLACE"}

# --- what an op needs ------------------------------------------------------- #

#: Goods an op must carry (engine handlers).
CARRIES: Mapping[str, str] = {"FERTILIZE": "FERTILIZER", "FEED": "WHEAT"}

#: What COLLECT_FERTILIZER yields into the bag (`_inv_add(inv, "FERTILIZER", 1)`).
COLLECT_ITEM = "FERTILIZER"

#: Ops that carry the entity itself (an animal).
PLACING_OPS: frozenset[str] = frozenset(("PLACE", "PLACE_ANIMAL"))

#: Ops that need a seed in `private["seeds"]`; seeds never travel.
SEED_OPS: frozenset[str] = frozenset(("PLANT",))

#: Ops that name the entity they construct.
ENTITY_OPS: frozenset[str] = frozenset(("PLANT", "BUILD", "PLACE", "PLACE_ANIMAL"))


def compile_op(op: str, entity: str | None = None) -> tuple[str, ...]:
    """One chain op -> the engine action it means. Raises on an unknown op."""
    op = LEGACY_ALIASES.get(op, op)
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
    """A chain -> the worker actions of one day, one per turn. The only expansion."""
    if "NO_ACT" in ops and len(ops) > 1:
        raise ValueError(f"NO_ACT is a whole-chain op, got {ops}")
    return [compile_op(op, entity) for op in ops]


# --- named views ------------------------------------------------------------ #

def _names(cls_name: str, values) -> Enum:
    """An `str` enum whose member names and values are the engine's own strings.

    `Good.WHEAT == "WHEAT"`, so a member is usable wherever the string is, and the
    member set is built from the engine table above — never typed. Members can be
    added to by the engine and removed only by it.
    """
    return Enum(cls_name, {str(v): str(v) for v in values}, type=str, module=__name__)


Good = _names("Good", GOODS)
Crop = _names("Crop", CROPS)
Species = _names("Species", ANIMALS)
Resource = _names("Resource", RESOURCES)
Product = _names("Product", PRODUCTS)
Vector = _names("Vector", VECTOR)
Action = _names("Action", sorted(ACTIONS))
WorkerOp = _names("WorkerOp", sorted(WORKER_OPS))
MarketAction = _names("MarketAction", sorted(MARKET_ACTIONS))
ChainOp = _names("ChainOp", sorted(CHAIN_OPS))
