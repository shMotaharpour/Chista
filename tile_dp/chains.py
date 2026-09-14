"""tile_dp chains: v14 - per-tile daily action chains (DIG-layered).

A chain is the ordered tuple of WORKER ops applied on one tile in one day.

Cost model (contract, 2026-09-14):
  * labour = number of ops that are in WORKER_OPS (allow-list below). Market
    buys are the market's action and a PICKUP is a carry of the secretary
    layer, so both cost 0 hours: supplying the inputs (seed / fertilizer /
    wheat / animal) is the SECRETARY layer's job.
  * NO_ACT = the worker does nothing on this tile (0 hours), but the day still
    passes and the tile state advances by one day. The engine's PASS action
    costs 1 hour and is deliberately NOT used in chain definitions.
  * requirements per op: PLANT -> 1 seed of that crop, FERTILIZE -> 1
    fertilizer, FEED -> 1 wheat, PLACE / PLACE_ANIMAL -> 1 animal of the
    entity's species (ANIMAL_GOOSE / ANIMAL_COW / ANIMAL_SHEEP).
  * `chain_requirements` is the single source of truth: it returns the labour
    hours (RES_LABOR) plus the inputs; `chain_labor` is a view of the same
    count and nothing outside counts ops again.

SECRETARY GAP: for a pure graph search every chain must also supply its
prerequisites (buy seed/fertilizer/wheat/animal and carry it to the tile). That
is the secretary layer, which does not exist yet, so the graph executor realises
the purchases inline as a stand-in. When the secretary appears, this is where
the split happens.

Applicability (2026-09-14): `chains_for` drops chains that can do nothing on the
node:
  * FERTILIZE for age < -2: the engine's fertilize effect covers the day itself
    plus two more (kaggriculture.py: fertilized_until_day = day + 2), so an
    earlier dose cannot reach the start of the effective window.
  * HARVEST for age < 0 (nothing produced before the first yield day) and when
    yield_units == 0 (harvesting zero units only burns the hour).

KNOWN FOLLOW-UP (deliberately left to the graph's turn, 2026-09-14): graph.py
still builds the cost vector inline and calls `chains_for` with an age only for
PLANT nodes, so these two filters change nothing in the built graphs until that
call also passes `yield_units=state.yield_units` and the age of ANIMAL nodes.
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
RES_ANIMAL = "ANIMAL"      # generic slot kept for the graph's inline use list;
                           # the secretary reads the species ids below
RES_ANIMAL_GOOSE = "ANIMAL_GOOSE"
RES_ANIMAL_COW = "ANIMAL_COW"
RES_ANIMAL_SHEEP = "ANIMAL_SHEEP"
# 1 animal per PLACE / PLACE_ANIMAL, species taken from the entity (2026-09-14)
ANIMAL_RES = {"GOOSE": RES_ANIMAL_GOOSE, "COW": RES_ANIMAL_COW,
              "SHEEP": RES_ANIMAL_SHEEP}
# 1 seed per PLANT, crop taken from the entity (same names as tile_state)
SEED_RES = {"WHEAT": RES_SEED_WHEAT, "CARROT": RES_SEED_CARROT,
            "TOMATO": RES_SEED_TOMATO, "STRAWBERRY": RES_SEED_STRAWBERRY,
            "MELON": RES_SEED_MELON}

RESOURCE_NAMES: tuple[str, ...] = (RES_LABOR, RES_FERTILIZER,
                                   RES_SEED_WHEAT, RES_SEED_CARROT,
                                   RES_SEED_TOMATO, RES_SEED_STRAWBERRY,
                                   RES_SEED_MELON, RES_WHEAT, RES_ANIMAL,
                                   RES_ANIMAL_GOOSE, RES_ANIMAL_COW,
                                   RES_ANIMAL_SHEEP)
RESOURCE_ID: dict[str, int] = {n: i for i, n in enumerate(RESOURCE_NAMES)}
N_RESOURCE = len(RESOURCE_NAMES)

# NO_ACT: day-pass op for a tile the worker does not touch (0 hours).
NO_ACT = "NO_ACT"
# Market ops: executed by the market, not by the worker (= 0 worker ops).
MARKET_OPS = ("BUY_SEED", "BUY_PRODUCT", "BUY_ANIMAL")
# Worker ops = the only ops that cost hours (contract 2026-09-14): market buys
# are the market's action and a PICKUP is a carry of the secretary layer, so
# both stay outside this set and cost 0 worker hours.
WORKER_OPS = frozenset(("PLANT", "WATER", "FERTILIZE", "HARVEST", "DIG",
                        "BUILD", "PLACE", "PLACE_ANIMAL", "FEED", "CARE",
                        "COLLECT_FERTILIZER"))

CROP_OPS = ("FERTILIZE", "WATER", "HARVEST")
ANIMAL_OPS = ("FEED", "CARE", "HARVEST", "COLLECT_FERTILIZER")


def _canonical_subsets(ops: tuple[str, ...]) -> list[tuple[str, ...]]:
    """All subsets of `ops` in canonical order; the empty one becomes NO_ACT."""
    return [tuple(op for op in ops if op in combo) or (NO_ACT,)
            for r in range(len(ops) + 1)
            for combo in combinations(ops, r)]


_CROP_SUBSETS: list[tuple[str, ...]] = _canonical_subsets(CROP_OPS)
_ANIMAL_SUBSETS: list[tuple[str, ...]] = _canonical_subsets(ANIMAL_OPS)

# chains per node kind, explicit (before the graph search prunes them). Every
# kind except NONE and ANIMAL is then DIG-layered (see below):
#  NONE            : one state shared by crops and animals - the caller (the
#                    graph) selects the subset it may run:
#                      crop   : NO_ACT | (PLANT, WATER)  (plant + water, F002)
#                      animal : NO_ACT | BUILD | BUILD,PLACE | BUILD,PLACE,FEED
#                               | BUILD,PLACE,FEED,CARE
#  WEED            : NO_ACT | DIG              (+ the DIG layering)
#  PLANT           : subsets of {FERTILIZE, WATER, HARVEST} (+ the layering);
#                    HARVEST is dropped for age < 0 (F026) by the filter
#  ANIMAL          : subsets of {FEED, CARE, HARVEST, COLLECT_FERTILIZER}
#  EMPTY_STRUCTURE : NO_ACT | PLACE_ANIMAL     (+ the DIG layering)
NONE_CHAINS_CROP: tuple[tuple[str, ...], ...] = (
    (NO_ACT,), ("PLANT", "WATER"))
NONE_CHAINS_ANIMAL: tuple[tuple[str, ...], ...] = (
    (NO_ACT,), ("BUILD",), ("BUILD", "PLACE"), ("BUILD", "PLACE", "FEED"),
    ("BUILD", "PLACE", "FEED", "CARE"))
# NONE is ONE state (2026-09-14): this is the single list of what may start on a
# bare tile; NONE_CHAINS_* are the two selections the graphs pick from.
NONE_CHAINS: tuple[tuple[str, ...], ...] = (NONE_CHAINS_CROP
                                            + NONE_CHAINS_ANIMAL[1:])

# --- DIG layering (2026-09-14) ----------------------------------------------
# A chain is closed over the tile's state space: it may START with DIG or put
# DIG right after HARVEST and may then run one full NONE chain, so one day can
# convert a tile from one kind to another (owner's rule). DIG never touches a
# tile that holds an animal (kaggriculture.py: DIG returns early then) and DIG
# on a bare tile is a no-op.
_KIND_AFTER: dict[str, dict[str, str]] = {
    "NONE": {"PLANT": "PLANT", "BUILD": "EMPTY_STRUCTURE"},
    "PLANT": {"DIG": "NONE"},
    "WEED": {"DIG": "NONE"},
    "EMPTY_STRUCTURE": {"DIG": "NONE", "PLACE": "ANIMAL",
                        "PLACE_ANIMAL": "ANIMAL"},
    "ANIMAL": {},
}


def _kind_after(kind: str, ops: tuple[str, ...]) -> str:
    """Tile kind after running `ops` on `kind` (known transitions only)."""
    for op in ops:
        kind = _KIND_AFTER.get(kind, {}).get(op, kind)
    return kind


def _dig_tail(kind: str, head: tuple[str, ...]) -> list[tuple[str, ...]]:
    """Variants that put DIG after `head` (empty head = DIG is the first op).

    Each variant ends with DIG (tile -> NONE) or with one full NONE chain; a
    follow-up that rebuilds the kind the tile had before DIG is dropped.
    """
    if "DIG" not in _KIND_AFTER.get(kind, {}):
        return []
    before = _kind_after(kind, head)
    if "DIG" not in _KIND_AFTER.get(before, {}):
        return []
    out: list[tuple[str, ...]] = [head + ("DIG",)]
    for follow in NONE_CHAINS:
        if follow == (NO_ACT,):
            continue
        # Same-kind rebuild is pointless (dig a plant, replant the same crop),
        # except an empty structure may change: COOP <-> PASTURE is legal
        # (owner 2026-09-14). The graph drops the identical-structure case.
        if (_kind_after("NONE", follow) == before
                and not (before == "EMPTY_STRUCTURE" and "BUILD" in follow)):
            continue                      # nothing changes: pointless variant
        out.append(head + ("DIG",) + follow)
    return out


def _layer(kind: str, base: tuple[tuple[str, ...], ...]
           ) -> tuple[tuple[str, ...], ...]:
    """Base chains plus every legal DIG layering of them (dedup, order kept)."""
    out: list[tuple[str, ...]] = list(base)
    out += _dig_tail(kind, ())                    # DIG as the first op
    for chain in base:
        if chain and chain[-1] == "HARVEST":      # DIG right after HARVEST
            out += _dig_tail(kind, chain)
    return tuple(dict.fromkeys(out))

# Every kind below is layered with DIG (`_layer`): base chains + DIG first +
# DIG right after HARVEST, with their NONE follow-ups.
WEED_CHAINS: tuple[tuple[str, ...], ...] = _layer(
    "WEED", ((NO_ACT,), ("DIG",)))
EMPTY_STRUCTURE_CHAINS: tuple[tuple[str, ...], ...] = _layer(
    "EMPTY_STRUCTURE", ((NO_ACT,), ("PLACE_ANIMAL",)))
CROP_CHAINS: tuple[tuple[str, ...], ...] = _layer(
    "PLANT", tuple(_CROP_SUBSETS))
CROP_CHAINS_YOUNG: tuple[tuple[str, ...], ...] = _layer(
    "PLANT", tuple(c for c in _CROP_SUBSETS if "HARVEST" not in c))
ANIMAL_CHAINS: tuple[tuple[str, ...], ...] = _layer(
    "ANIMAL", tuple(_ANIMAL_SUBSETS))

# The per-kind lists above are the contract; `chains_for` only selects from them
# and then filters. Registry ids stay internal (they can shift when a list
# changes) - use `chain_name` / `ops_of_name` for anything that outlives a run.
CHAINS_BY_KIND: dict[str, tuple[tuple[str, ...], ...]] = {
    "NONE": NONE_CHAINS,
    "WEED": WEED_CHAINS,
    "PLANT": CROP_CHAINS,
    "ANIMAL": ANIMAL_CHAINS,
    "EMPTY_STRUCTURE": EMPTY_STRUCTURE_CHAINS,
}

_REGISTRY: list[tuple[str, ...]] = []
for c in (NONE_CHAINS_CROP + NONE_CHAINS_ANIMAL + WEED_CHAINS
          + EMPTY_STRUCTURE_CHAINS + CROP_CHAINS + ANIMAL_CHAINS):
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


def _worker_hours(ops: tuple[str, ...]) -> int:
    """Worker hours of a chain: only ops in WORKER_OPS cost hours."""
    return sum(1 for op in ops if op in WORKER_OPS)


def chain_labor(ops: tuple[str, ...]) -> int:
    """Worker hours of a chain (thin view of `chain_requirements`)."""
    return _worker_hours(ops)


def chain_requirements(entity: str, ops: tuple[str, ...]) -> dict[str, int]:
    """Cost of a chain: labour hours + input requirements.

    Single source of truth for the cost model (2026-09-14): the labour hours
    (RES_LABOR) are produced here too, so nobody outside has to count ops
    again. PLANT -> 1 seed of the entity's crop, FERTILIZE -> 1 fertilizer,
    FEED -> 1 wheat, PLACE / PLACE_ANIMAL -> 1 animal of the entity's species.
    """
    req: dict[str, int] = {RES_LABOR: _worker_hours(ops)}
    for op in ops:
        if op == "PLANT":
            key = SEED_RES.get(entity, "SEED_" + entity)
            req[key] = req.get(key, 0) + 1
        elif op == "FERTILIZE":
            req[RES_FERTILIZER] = req.get(RES_FERTILIZER, 0) + 1
        elif op == "FEED":
            req[RES_WHEAT] = req.get(RES_WHEAT, 0) + 1
        elif op in ("PLACE", "PLACE_ANIMAL"):
            key = ANIMAL_RES.get(entity, RES_ANIMAL)
            req[key] = req.get(key, 0) + 1
    return req


def chain_name(ops: tuple[str, ...]) -> str:
    """Stable name of a chain ('FERTILIZE+WATER+HARVEST', 'NO_ACT')."""
    return "+".join(ops)


def ops_of_name(name: str) -> tuple[str, ...]:
    """Inverse of `chain_name` (use names for anything that outlives a run)."""
    return tuple(name.split("+"))


def _applicable(ops: tuple[str, ...], age: int | None,
                yield_units: int | None) -> bool:
    """False when the chain is a guaranteed no-op on the node (2026-09-14)."""
    if age is not None and age < -2 and "FERTILIZE" in ops:
        return False        # a 3-day fertilize effect cannot reach the window
    if "HARVEST" in ops:
        if age is not None and age < 0:
            return False    # nothing produced before the first yield day
        if yield_units is not None and yield_units <= 0:
            return False    # harvesting zero units only burns the hour
    return True


def chains_for(kind: str, age: int | None = None,
               animal_graph: bool = False,
               yield_units: int | None = None) -> list[tuple[str, ...]]:
    """Applicable chains of a node kind BEFORE pruning, after the filters.

    `animal_graph=True` only SELECTS the animal subset of the single NONE list
    (NONE is one state; the graph decides which start chains it may run).
    `age` / `yield_units` prune chains that would do nothing (see module doc).
    """
    if kind == "NONE":
        base = NONE_CHAINS_ANIMAL if animal_graph else NONE_CHAINS_CROP
    elif kind == "PLANT":
        base = CROP_CHAINS_YOUNG if (age is not None and age < 0) \
            else CROP_CHAINS
    elif kind == "ANIMAL":
        base = ANIMAL_CHAINS
    elif kind == "WEED":
        base = WEED_CHAINS
    elif kind == "EMPTY_STRUCTURE":
        base = EMPTY_STRUCTURE_CHAINS
    else:
        return []
    return [c for c in base if _applicable(c, age, yield_units)]
