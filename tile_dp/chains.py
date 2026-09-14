"""tile_dp chains: v13 - per-tile daily action chains.

A chain is the ordered tuple of WORKER ops applied on one tile in one day.

Cost model (contract, 2026-09-14):
  * labour = number of worker ops in the chain. Market buys are NOT worker
    actions and the PICKUPs are not modelled here: supplying the inputs
    (seed / fertilizer / wheat / animal) is the SECRETARY layer's job.
  * NO_ACT = the worker does nothing on this tile (0 hours), but the day still
    passes and the tile state advances by one day. The engine's PASS action
    costs 1 hour and is deliberately NOT used in chain definitions.
  * requirements per op: PLANT -> 1 seed of that crop, FERTILIZE -> 1
    fertilizer, FEED -> 1 wheat, PLACE / PLACE_ANIMAL -> 1 animal.

SECRETARY GAP: for a pure graph search every chain must also supply its
prerequisites (buy seed/fertilizer/wheat/animal and carry it to the tile). That
is the secretary layer, which does not exist yet, so the graph executor realises
the purchases inline as a stand-in. When the secretary appears, this is where
the split happens.
"""

from __future__ import annotations

from itertools import combinations

RES_LABOR = "LABOR_HOURS"
RES_FERTILIZER = "FERTILIZER"
RES_SEED_WHEAT = "SEED_WHEAT"
RES_SEED_CARROT = "SEED_CARROT"
RES_SEED_TOMATO = "SEED_TOMATO"
RES_SEED_STRAWBERRY = "SEED_STRAWBERRY"
RES_SEED_MELON = "SEED_MELON"
RES_WHEAT = "WHEAT_FOOD"   # 1 wheat per FEED (animal food, from the bag)
RES_ANIMAL = "ANIMAL"      # 1 animal per PLACE (bought by the secretary)

RESOURCE_NAMES: tuple[str, ...] = (RES_LABOR, RES_FERTILIZER,
                                   RES_SEED_WHEAT, RES_SEED_CARROT,
                                   RES_SEED_TOMATO, RES_SEED_STRAWBERRY,
                                   RES_SEED_MELON, RES_WHEAT, RES_ANIMAL)
RESOURCE_ID: dict[str, int] = {n: i for i, n in enumerate(RESOURCE_NAMES)}
N_RESOURCE = len(RESOURCE_NAMES)

# NO_ACT: day-pass op for a tile the worker does not touch (0 hours).
NO_ACT = "NO_ACT"
# Market ops: executed by the market, not by the worker (= 0 worker ops).
MARKET_OPS = ("BUY_SEED", "BUY_PRODUCT", "BUY_ANIMAL")

CROP_OPS = ["FERTILIZE", "WATER", "HARVEST"]
ANIMAL_OPS = ["FEED", "CARE", "HARVEST", "COLLECT_FERTILIZER"]


def _canonical_subsets(ops: list[str]) -> list[tuple[str, ...]]:
    """All subsets of `ops` in canonical order; the empty one becomes NO_ACT."""
    return [tuple(op for op in ops if op in combo) or (NO_ACT,)
            for r in range(len(ops) + 1)
            for combo in combinations(ops, r)]


_CROP_SUBSETS: list[tuple[str, ...]] = _canonical_subsets(CROP_OPS)
_ANIMAL_SUBSETS: list[tuple[str, ...]] = _canonical_subsets(ANIMAL_OPS)

# chains per node kind (before the graph search prunes them):
#  NONE            : NO_ACT | PLANT,WATER (plant + water same day, F002)
#  NONE (animal)   : NO_ACT | BUILD | BUILD,PLACE | BUILD,PLACE,FEED
#  WEED            : NO_ACT | DIG
#  PLANT young     : subsets of {FERTILIZE, WATER}      (no harvest yet, F026)
#  PLANT mature    : subsets of {FERTILIZE, WATER, HARVEST}
#  ANIMAL          : subsets of {FEED, CARE, HARVEST, COLLECT_FERTILIZER}
#  EMPTY_STRUCTURE : NO_ACT | PLACE_ANIMAL
_CROP_MATURE: list[tuple[str, ...]] = list(_CROP_SUBSETS)
_YOUNG: list[tuple[str, ...]] = [c for c in _CROP_SUBSETS
                                 if "HARVEST" not in c]

NONE_CHAINS: tuple[tuple[str, ...], ...] = ((NO_ACT,), ("PLANT", "WATER"))
NONE_CHAINS_ANIMAL: tuple[tuple[str, ...], ...] = (
    (NO_ACT,), ("BUILD",), ("BUILD", "PLACE"), ("BUILD", "PLACE", "FEED"))
WEED_CHAINS: tuple[tuple[str, ...], ...] = ((NO_ACT,), ("DIG",))
ANIMAL_CHAINS: tuple[tuple[str, ...], ...] = tuple(_ANIMAL_SUBSETS)
EMPTY_STRUCTURE_CHAINS: tuple[tuple[str, ...], ...] = (
    (NO_ACT,), ("PLACE_ANIMAL",))

_REGISTRY: list[tuple[str, ...]] = []
for c in ([("PLANT", "WATER"), ("BUILD",), ("BUILD", "PLACE"),
           ("BUILD", "PLACE", "FEED"), ("DIG",), ("PLACE_ANIMAL",)]
          + _CROP_SUBSETS + _ANIMAL_SUBSETS):
    if c not in _REGISTRY:
        _REGISTRY.append(c)

CHAIN_NAMES: tuple[tuple[str, ...], ...] = tuple(_REGISTRY)
CHAIN_ID_OF: dict[tuple[str, ...], int] = {c: i for i, c in enumerate(_REGISTRY)}


def chain_ops(chain_id: int) -> tuple[str, ...]:
    """Decode a chain id into its op-name tuple (boundary function)."""
    return CHAIN_NAMES[chain_id]


def chain_id_of(ops: tuple[str, ...]) -> int:
    """Encode an op-name tuple into its registry id (boundary function)."""
    return CHAIN_ID_OF[ops]


def chain_labor(ops: tuple[str, ...]) -> int:
    """Worker ops of a chain: market buys and NO_ACT cost no hours."""
    return sum(1 for op in ops if op not in MARKET_OPS and op != NO_ACT)


def chain_requirements(entity: str, ops: tuple[str, ...]) -> dict[str, int]:
    """Input requirements of a chain (the secretary's shopping list)."""
    req: dict[str, int] = {}
    for op in ops:
        if op == "PLANT":
            req["SEED_" + entity] = req.get("SEED_" + entity, 0) + 1
        elif op == "FERTILIZE":
            req[RES_FERTILIZER] = req.get(RES_FERTILIZER, 0) + 1
        elif op == "FEED":
            req[RES_WHEAT] = req.get(RES_WHEAT, 0) + 1
        elif op in ("PLACE", "PLACE_ANIMAL"):
            req[RES_ANIMAL] = req.get(RES_ANIMAL, 0) + 1
    return req


def chains_for(kind: str, age: int | None = None,
               animal_graph: bool = False) -> list[tuple[str, ...]]:
    """Applicable chains BEFORE pruning, by node kind.

    animal_graph=True selects the animal-graph NONE chains instead of the crop
    planting chain. age (crop states) prunes HARVEST chains when age < 0 (F026).
    """
    if kind == "NONE":
        if animal_graph:
            return list(NONE_CHAINS_ANIMAL)
        return list(NONE_CHAINS)
    if kind == "WEED":
        return list(WEED_CHAINS)
    if kind == "ANIMAL":
        return list(ANIMAL_CHAINS)
    if kind == "EMPTY_STRUCTURE":
        return list(EMPTY_STRUCTURE_CHAINS)
    if age is not None and age < 0:
        return [c for c in _YOUNG]
    return list(_CROP_MATURE)
