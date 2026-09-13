"""Daily action chains for the carrot tile, with canonical ordering.

Order rule (Hossein, this branch): within one day, a chain's ops appear in
  PLANT -> FERTILIZE -> WATER -> HARVEST   (+ DIG at the end)
Only chains in canonical order are generated — (WATER, FERTILIZE) is
NOT a distinct chain (engine-verified: same-day order of fert vs water
does not change the outcome), and (HARVEST before WATER) is forbidden.

Resources: LABOR_HOURS (1 per op, FERTILIZE costs 2 = PICKUP + FERT),
FERTILIZER (1 per FERTILIZE), SEED_CARROT (1 per PLANT). Purchases are
market orders — they land one turn before the op needs them (F030) and
cost money, not labor; the money price is the secretary's business.
"""

from __future__ import annotations

from itertools import combinations

RES_LABOR = "LABOR_HOURS"
RES_FERTILIZER = "FERTILIZER"
RES_SEED = "SEED_CARROT"

RESOURCE_NAMES: tuple[str, ...] = (RES_LABOR, RES_FERTILIZER, RES_SEED)
RESOURCE_ID: dict[str, int] = {n: i for i, n in enumerate(RESOURCE_NAMES)}
N_RESOURCE = len(RESOURCE_NAMES)

# canonical order index (DIG always last)
_OP_ORDER = {"PLANT": 0, "FERTILIZE": 1, "WATER": 2, "HARVEST": 3, "DIG": 4}

# every canonical subset of the three PLANT-TILE ops (PLANT itself is only
# reachable from the NONE state via its own planting chain)
_PLANT_OPS = ["FERTILIZE", "WATER", "HARVEST"]
_ALL_SUBSETS: tuple[tuple[str, ...], ...] = tuple(
    tuple(op for op in _PLANT_OPS if op in combo)
    for r in range(len(_PLANT_OPS) + 1)
    for combo in combinations(_PLANT_OPS, r)
)

# per-state-kind applicable chains (before pruning):
#   NONE: PASS or the planting chain (buy + plant + water — water on
#         planting day is mandatory practice, F002)
#   WEED: PASS or DIG (recover the tile)
#   PLANT age < 0 (not harvestable): subsets of {FERTILIZE, WATER}
#   PLANT age >= 0 (harvestable):     subsets of {FERTILIZE, WATER, HARVEST}
_NONE_CHAINS: tuple[tuple[str, ...], ...] = (("PASS",), ("PLANT", "WATER"))
_WEED_CHAINS: tuple[tuple[str, ...], ...] = (("PASS",), ("DIG",))
# normalize the empty subset to the explicit PASS chain
_YOUNG_CHAINS: tuple[tuple[str, ...], ...] = tuple(
    ('PASS',) if c == () else c
    for c in _ALL_SUBSETS if "HARVEST" not in c)
_MATURE_CHAINS: tuple[tuple[str, ...], ...] = tuple(
    ('PASS',) if c == () else c for c in _ALL_SUBSETS)

# the single registry — stable order:
#   0: PASS | 1: DIG | 2: PLANT,WATER | 3..: young subsets | then mature
_REGISTRY: list[tuple[str, ...]] = [("PASS",), ("DIG",), ("PLANT", "WATER")]
for c in _YOUNG_CHAINS:
    if c != ("PASS",) and c not in _REGISTRY:
        _REGISTRY.append(c)
for c in _MATURE_CHAINS:
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


def chains_for(state_kind: str, age: int | None = None) -> list[tuple[str, ...]]:
    """Applicable chains BEFORE pruning, by state kind (and plant age)."""
    if state_kind == "NONE":
        return list(_NONE_CHAINS)
    if state_kind == "WEED":
        return list(_WEED_CHAINS)
    if age is not None and age < 0:
        return list(_YOUNG_CHAINS)
    return list(_MATURE_CHAINS)
