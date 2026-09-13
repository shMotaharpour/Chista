"""Daily action chains for all tile contents (crops + animals), canonical
order, with per-state applicability and resource costs.

Order rule (Hossein): within a day, ops run in canonical order:
  crops:   PLANT -> FERTILIZE -> WATER -> HARVEST   (+ DIG last)
  animals: FEED -> CARE -> HARVEST -> COLLECT_FERT  (+ none)
Only canonical sequences exist (same-day order doesn't change outcomes
except WATER-before-HARVEST which is physics — F026 — and is enforced by
the canonical order anyway).
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

RESOURCE_NAMES: tuple[str, ...] = (RES_LABOR, RES_FERTILIZER,
                                   RES_SEED_WHEAT, RES_SEED_CARROT,
                                   RES_SEED_TOMATO, RES_SEED_STRAWBERRY,
                                   RES_SEED_MELON, RES_WHEAT)
RESOURCE_ID: dict[str, int] = {n: i for i, n in enumerate(RESOURCE_NAMES)}
N_RESOURCE = len(RESOURCE_NAMES)

CROP_OPS = ["FERTILIZE", "WATER", "HARVEST"]
ANIMAL_OPS = ["FEED", "CARE", "HARVEST", "COLLECT_FERTILIZER"]


def _canonical_subsets(ops: list[str]) -> list[tuple[str, ...]]:
    return [tuple(op for op in ops if op in combo)
            for r in range(len(ops) + 1)
            for combo in combinations(ops, r)]


_CROP_SUBSETS: list[tuple[str, ...]] = _canonical_subsets(CROP_OPS)
_ANIMAL_SUBSETS: list[tuple[str, ...]] = _canonical_subsets(ANIMAL_OPS)

# chains_for: per node kind, before engine-pruning
#  NONE            : PASS | PLANT,WATER (buy seed + plant + water)
#  WEED            : PASS | DIG
#  PLANT young     : subsets of {FERTILIZE, WATER}      (no harvest yet)
#  PLANT mature    : subsets of {FERTILIZE, WATER, HARVEST}
#  ANIMAL          : subsets of {FEED, CARE, HARVEST, COLLECT_FERT}
#  EMPTY_STRUCTURE : PASS | PLACE (animal)              (no DIG needed)
_CROP_MATURE: list[tuple[str, ...]] = [c for c in _CROP_SUBSETS]
_YOUNG: list[tuple[str, ...]] = [c for c in _CROP_SUBSETS
                                 if "HARVEST" not in c]

NONE_CHAINS: tuple[tuple[str, ...], ...] = (("PASS",), ("PLANT", "WATER"))
WEED_CHAINS: tuple[tuple[str, ...], ...] = (("PASS",), ("DIG",))
ANIMAL_CHAINS: tuple[tuple[str, ...], ...] = tuple(_ANIMAL_SUBSETS)
EMPTY_STRUCTURE_CHAINS: tuple[tuple[str, ...], ...] = (
    ("PASS",), ("PLACE_ANIMAL",))

_REGISTRY: list[tuple[str, ...]] = []
for c in ([("PASS",), ("DIG",), ("PLANT", "WATER")]
          + _CROP_SUBSETS + _ANIMAL_SUBSETS + [("PLACE_ANIMAL",)]):
    if c not in _REGISTRY:
        _REGISTRY.append(c)

CHAIN_NAMES: tuple[tuple[str, ...], ...] = tuple(_REGISTRY)
CHAIN_ID_OF: dict[tuple[str, ...], int] = {c: i for i, c in enumerate(_REGISTRY)}


def chain_ops(chain_id: int) -> tuple[str, ...]:
    return CHAIN_NAMES[chain_id]


def chain_id_of(ops: tuple[str, ...]) -> int:
    return CHAIN_ID_OF[ops]


def chains_for(kind: str, age: int | None = None, ongoing: bool = False
               ) -> list[tuple[str, ...]]:
    """Applicable chains before pruning, by node kind."""
    if kind == "NONE":
        return list(NONE_CHAINS)
    if kind == "WEED":
        return list(WEED_CHAINS)
    if kind == "ANIMAL":
        return list(ANIMAL_CHAINS)
    if kind == "EMPTY_STRUCTURE":
        return list(EMPTY_STRUCTURE_CHAINS)
    # PLANT
    if age is not None and age < 0:
        return list(_YOUNG)
    return list(_CROP_MATURE)
