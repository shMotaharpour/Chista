"""Daily action chains per species (v1: wheat + carrot).

A chain is the ordered tuple of tile-ops the units perform on the tile during
ONE day (each op = 1 turn = 1 labor hour). Ordering constraints (engine-
verified, see DESIGN.md):

- One-shot crops (wheat/carrot): WATER strictly before HARVEST — the watered
  unit lands immediately (F026), so harvesting before watering forfeits the
  unit. FERTILIZE must precede WATER to cover the watering day... actually
  F004/F019: coverage is by DAY, and CARE-style ordering does not apply to
  plants; FERTILIZE on a day makes that day fertilized for the night calcs,
  so FERTILIZE anywhere in the day counts. We keep chains canonical:
  FERTILIZE -> WATER -> HARVEST -> (DIG when clearing) — DIG last because it
  empties the tile (F007).
- PLANT is not a chain op: it is a *transition decision* from NONE (the
  graph builder models planting as the NONE->age=-origin edge with the seed
  resource). Kept here as the planting pseudo-op name for readability.

Chains are stored as integer ids; ops decode at the boundary only.
"""

from __future__ import annotations

# Canonical op order within a day for one-shot crops (v1).
_OP_ORDER = {"FERTILIZE": 0, "WATER": 1, "HARVEST": 2, "DIG": 3}

# Resource names (resource_id registry — DESIGN.md)
RES_LABOR = "LABOR_HOURS"
RES_SEED_WHEAT = "SEED_WHEAT"
RES_SEED_CARROT = "SEED_CARROT"
RES_FERTILIZER = "FERTILIZER"

RESOURCE_NAMES: tuple[str, ...] = (RES_LABOR, RES_SEED_WHEAT,
                                   RES_SEED_CARROT, RES_FERTILIZER)
RESOURCE_ID: dict[str, int] = {n: i for i, n in enumerate(RESOURCE_NAMES)}

# Precondition tags (symbolic; v1 assumes satisfied — explicit for later)
PRE_HAS_FERTILIZER = "HAS_FERTILIZER"
PRE_HAS_SEED_WHEAT = "HAS_SEED_WHEAT"
PRE_HAS_SEED_CARROT = "HAS_SEED_CARROT"
PRE_STANDS_ON_TILE = "STANDS_ON_TILE"


def _make_chains(allow_harvest: bool) -> tuple[tuple[str, ...], ...]:
    """Lattice of daily chains for a one-shot crop, canonical order, deduped.

    Subsets of {FERTILIZE, WATER, HARVEST, DIG} in canonical order, minus
    empty, minus DIG-without-anything (a bare DIG just clears the tile —
    that IS a valid chain for the WEED state), with WATER<HARVEST enforced
    when both present.
    """
    chains: list[tuple[str, ...]] = []
    ops = [op for op, _ in sorted(_OP_ORDER.items(), key=lambda kv: kv[1])]
    n = len(ops)
    for mask in range(1, 1 << n):
        seq = tuple(ops[i] for i in range(n) if mask & (1 << i))
        if not allow_harvest and "HARVEST" in seq:
            continue
        # WATER strictly before HARVEST (canonical order already guarantees
        # it, but keep the guard explicit for future op-set changes)
        if "WATER" in seq and "HARVEST" in seq \
                and seq.index("WATER") > seq.index("HARVEST"):
            continue
        chains.append(seq)
    return tuple(chains)


# Registry (v1: same lattice for wheat and carrot — both one-shot, both
# harvestable; DIG-only is included because weed clearing uses it).
CHAINS: tuple[tuple[str, ...], ...] = _make_chains(allow_harvest=True)

# Planting pseudo-chains (NONE -> plant transitions are graph edges with a
# seed resource; the actual op executed on day d hour 0 is PLANT).
PLANT_CHAIN_WHEAT: tuple[str, ...] = ("PLANT_WHEAT", "WATER")
PLANT_CHAIN_CARROT: tuple[str, ...] = ("PLANT_CARROT", "WATER")

# The single chain registry — ONE source of ids for the graph builder, the
# solver, and storage. Order: PASS (0), plant pseudo-chains, then the
# one-shot op lattice. The graph's _chain_id must use THIS registry.
_ALL_CHAINS: tuple[tuple[str, ...], ...] = (
    (("PASS",),) + (PLANT_CHAIN_WHEAT, PLANT_CHAIN_CARROT) + CHAINS)

CHAIN_NAME_TO_ID: dict[tuple[str, ...], int] = {
    c: i for i, c in enumerate(_ALL_CHAINS)}


def chain_ops(chain_id: int) -> tuple[str, ...]:
    """Decode a chain id into its op-name tuple (boundary function)."""
    return _ALL_CHAINS[chain_id]


def chain_id_of(ops: tuple[str, ...]) -> int:
    """Encode an op-name tuple into its registry id (boundary function)."""
    return CHAIN_NAME_TO_ID[ops]


def encode_chains() -> tuple[tuple[str, ...], ...]:
    """The full registry (used by the graph builder / storage)."""
    return _ALL_CHAINS
