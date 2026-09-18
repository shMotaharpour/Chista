"""The tile's daily chains: what one worker-day on one tile can be, and what it costs.

A chain is the ordered tuple of ops applied on one tile in one day. The vocabulary is
`agent.world`'s — there is no op here that the engine does not have — plus the three
market buys a chain may name as its input acquisition, which the market layer executes
and a worker never does.

    labour    every op in TILE_OPS costs one worker hour; a market buy is the market's
              action and a shed trip is the day layer's, so both cost none. PASS is the
              idle chain: no worker visits the tile, so it costs no hour and no turn —
              the tile simply ages a day.
    inputs    PLANT -> 1 seed of the entity's crop, FERTILIZE -> 1 fertilizer,
              FEED -> 1 wheat, PLACE -> 1 animal of the entity's species.
              `chain_requirements` is the single source of truth for both; nothing
              outside counts ops again.

Cost and produce are SEPARATE vectors over the 18 `world.model.Column` names, and they
are never netted: wheat is the FEED input and the WHEAT crop's product, so one column
carries both readings and the side says which.

The chains are generated, per node kind, and closed over the tile's state space: a chain
may DIG first or DIG right after HARVEST and then run one full bare-tile chain, so a
single day can convert a tile from one kind to another. `chains_for` selects from those
lists and then filters by the node's own age, yield and entity — the engine's windows,
not constants (HARVEST only from the crop's first yield day, FERTILIZE only while the
three-day dose can still reach an effective day, CARE only with FEED, and never a
structure the entity does not live in).
"""

from __future__ import annotations

from itertools import combinations

from agent.world.model import (ANIMALS, COLUMNS, CROPS, Animal, Crop, Column,
                               Structure, TileKind, UnitAction)
from agent.world.rules import ANIMAL_RULES, CROP_RULES, TURNS_PER_DAY
from agent.world.tile import crop_age_origin

# --- the columns a plan is priced over ---------------------------------------- #

RESOURCE_NAMES: tuple[str, ...] = COLUMNS
RESOURCE_ID: dict[str, int] = {name: i for i, name in enumerate(RESOURCE_NAMES)}
N_RESOURCE = len(RESOURCE_NAMES)

#: Which column a PLANT / PLACE spends, and which one a harvest fills.
SEED_RES: dict[str, str] = {crop: f"SEED_{crop}" for crop in CROPS}
ANIMAL_RES: dict[str, str] = {animal: f"ANIMAL_{animal}" for animal in ANIMALS}
PRODUCT_RES: dict[str, str] = {crop: crop for crop in CROPS}
PRODUCT_RES.update({a: ANIMAL_RULES[a]["product"] for a in ANIMALS})
RES_LABOR = Column.LABOR.value
RES_FERTILIZER = Column.FERTILIZER.value
RES_WHEAT = Column.WHEAT.value

# --- the vocabulary ------------------------------------------------------------ #

#: The ops a chain may name that the market executes. A chain names the input it needs;
#: the market layer decides the order and the price.
MARKET_OPS: tuple[str, ...] = ("BUY_SEED", "BUY_PRODUCT", "BUY_ANIMAL")

#: The ops that cost a worker hour: every unit op except PASS. A shed trip is not here
#: — the day layer schedules it, the contractor does not price it.
TILE_OPS: frozenset[str] = frozenset(
    op.value for op in UnitAction if op is not UnitAction.PASS)

#: The case with no action at all: no worker touches the tile that day and the tile
#: simply goes on to the next one. It is not `PASS` — PASS is a turn spent waiting, which
#: is the builder's business when it has to let a market purchase land, and a chain never
#: names it.
NO_ACTION: tuple[str, ...] = ()

#: Every op a chain may name.
ALL_OPS: frozenset[str] = TILE_OPS | set(MARKET_OPS)
BUILD_OF_STRUCTURE: dict[Structure, str] = {Structure.COOP: UnitAction.BUILD_COOP.value,
                                            Structure.PASTURE: UnitAction.BUILD_PASTURE.value}

#: The ops that create the entity a chain is about, so they are what its entity code is
#: read from.
CONSTRUCTIVE_OPS: tuple[str, ...] = (UnitAction.PLANT.value,
                                     UnitAction.BUILD_COOP.value,
                                     UnitAction.BUILD_PASTURE.value,
                                     UnitAction.PLACE.value)

#: The domain filter: a crop entity never runs an animal op and an animal entity never
#: plants; HARVEST is the one op both have.
ANIMAL_ONLY_OPS: frozenset[str] = frozenset((
    UnitAction.BUILD_COOP.value, UnitAction.BUILD_PASTURE.value, UnitAction.PLACE.value,
    UnitAction.FEED.value, UnitAction.CARE.value, UnitAction.COLLECT_FERTILIZER.value))
CROP_ONLY_OPS: frozenset[str] = frozenset((UnitAction.PLANT.value,))

#: The two chains' op pools, per node kind.
CROP_OPS: tuple[str, ...] = (UnitAction.FERTILIZE.value, UnitAction.WATER.value,
                             UnitAction.HARVEST.value)
ANIMAL_OPS: tuple[str, ...] = (UnitAction.FEED.value, UnitAction.CARE.value,
                               UnitAction.HARVEST.value,
                               UnitAction.COLLECT_FERTILIZER.value)

#: DIG turns exactly these kinds back into a bare tile: the engine removes a plant, a
#: weed or an EMPTY structure, refuses a tile that holds an animal, and does nothing on
#: a bare tile (kaggriculture.py:484-491).
DIGGABLE_KINDS: frozenset[TileKind] = frozenset(
    (TileKind.PLANT, TileKind.WEED, TileKind.EMPTY_COOP, TileKind.EMPTY_PASTURE))

# --- the chains, generated ------------------------------------------------------ #

def _subsets(ops: tuple[str, ...]) -> list[tuple[str, ...]]:
    """Every subset of `ops`; the empty one is the no-action case."""
    return [tuple(op for op in ops if op in combo) or NO_ACTION
            for r in range(len(ops) + 1)
            for combo in combinations(ops, r)]


_CROP_SUBSETS: list[tuple[str, ...]] = _subsets(CROP_OPS)
# CARE is a no-op without FEED on the same day: the engine spends `cared_today` only
# together with `fed_today` (kaggriculture.py:826-829), so CARE alone is never offered.
_ANIMAL_SUBSETS: list[tuple[str, ...]] = [
    c for c in _subsets(ANIMAL_OPS) if not ("CARE" in c and "FEED" not in c)]

#: What may start on a bare tile: a crop plants, an animal builds its own structure and
#: moves in. The two selections below are what the graph picks from — NONE is one state.
#: ONE list, with no crop/animal split: what a worker can do on a bare tile. The split
#: belongs to the process that builds the graph, never to the chains themselves — the
#: graph and the registry must not know that a tile was "meant" for a crop or an animal.
NONE_CHAINS: tuple[tuple[str, ...], ...] = (
    NO_ACTION,
    (UnitAction.PLANT.value, UnitAction.WATER.value),
) + tuple(
    chain for structure in (Structure.COOP, Structure.PASTURE)
    for chain in ((BUILD_OF_STRUCTURE[structure],),
                  (BUILD_OF_STRUCTURE[structure], UnitAction.PLACE.value),
                  (BUILD_OF_STRUCTURE[structure], UnitAction.PLACE.value,
                   UnitAction.FEED.value),
                  (BUILD_OF_STRUCTURE[structure], UnitAction.PLACE.value,
                   UnitAction.FEED.value, UnitAction.CARE.value)))


def _kind_after(kind: TileKind, ops: tuple[str, ...]) -> TileKind:
    """The tile kind after `ops` run on `kind` (the transitions the engine allows)."""
    for op in ops:
        if op == UnitAction.PLANT.value:
            kind = TileKind.PLANT
        elif op == UnitAction.BUILD_COOP.value:
            kind = TileKind.EMPTY_COOP
        elif op == UnitAction.BUILD_PASTURE.value:
            kind = TileKind.EMPTY_PASTURE
        elif op == UnitAction.PLACE.value:
            kind = TileKind.ANIMAL
        elif op == UnitAction.DIG.value:
            kind = TileKind.NONE
    return kind


def _dig_tail(kind: TileKind, head: tuple[str, ...]) -> list[tuple[str, ...]]:
    """Variants that put DIG after `head` (an empty head = DIG is the first op).

    Each variant ends with DIG (tile back to bare) or with one full bare-tile chain, so
    one day can change what a tile is. Nothing is dropped for "the kind would not
    change": digging a crop and planting it again restarts its lifecycle, which is a
    real option past its golden window. The variants that truly add nothing are removed
    by the graph's own Pareto rule, where the decision belongs.
    """
    if kind not in DIGGABLE_KINDS:
        return []
    if _kind_after(kind, head) not in DIGGABLE_KINDS:
        return []
    out = [head + (UnitAction.DIG.value,)]
    out += [head + (UnitAction.DIG.value,) + follow for follow in NONE_CHAINS
            if follow != NO_ACTION]
    return out


def _layer(kind: TileKind, base: tuple[tuple[str, ...], ...]
           ) -> tuple[tuple[str, ...], ...]:
    """Base chains plus every legal DIG layering of them (dedup, order kept)."""
    out: list[tuple[str, ...]] = list(base)
    out += _dig_tail(kind, ())                       # DIG as the first op
    for chain in base:
        if chain and chain[-1] == UnitAction.HARVEST.value:
            out += _dig_tail(kind, chain)            # DIG right after HARVEST
    return tuple(dict.fromkeys(out))


WEED_CHAINS: tuple[tuple[str, ...], ...] = _layer(
    TileKind.WEED, (NO_ACTION, (UnitAction.DIG.value,)))
#: An empty structure takes an animal of its own structure and can be dug back; BUILD is
#: absent here on purpose — the engine builds on a bare tile only.
EMPTY_STRUCTURE_BASE: tuple[tuple[str, ...], ...] = (
    NO_ACTION, (UnitAction.PLACE.value,))
EMPTY_COOP_CHAINS: tuple[tuple[str, ...], ...] = _layer(TileKind.EMPTY_COOP,
                                                        EMPTY_STRUCTURE_BASE)
EMPTY_PASTURE_CHAINS: tuple[tuple[str, ...], ...] = _layer(TileKind.EMPTY_PASTURE,
                                                           EMPTY_STRUCTURE_BASE)
CROP_CHAINS: tuple[tuple[str, ...], ...] = _layer(TileKind.PLANT, tuple(_CROP_SUBSETS))
CROP_CHAINS_YOUNG: tuple[tuple[str, ...], ...] = _layer(
    TileKind.PLANT, tuple(c for c in _CROP_SUBSETS
                          if UnitAction.HARVEST.value not in c))
ANIMAL_CHAINS: tuple[tuple[str, ...], ...] = _layer(TileKind.ANIMAL,
                                                    tuple(_ANIMAL_SUBSETS))

#: Per kind: the contract `chains_for` selects from.
CHAINS_BY_KIND: dict[TileKind, tuple[tuple[str, ...], ...]] = {
    TileKind.NONE: NONE_CHAINS,
    TileKind.WEED: WEED_CHAINS,
    TileKind.PLANT: CROP_CHAINS,
    TileKind.ANIMAL: ANIMAL_CHAINS,
    TileKind.EMPTY_COOP: EMPTY_COOP_CHAINS,
    TileKind.EMPTY_PASTURE: EMPTY_PASTURE_CHAINS,
}

_REGISTRY: list[tuple[str, ...]] = []
for _chain in (NONE_CHAINS + WEED_CHAINS + EMPTY_COOP_CHAINS + EMPTY_PASTURE_CHAINS
               + CROP_CHAINS + ANIMAL_CHAINS):
    if _chain not in _REGISTRY:
        _REGISTRY.append(_chain)

CHAIN_NAMES: tuple[tuple[str, ...], ...] = tuple(_REGISTRY)
CHAIN_ID_OF: dict[tuple[str, ...], int] = {c: i for i, c in enumerate(_REGISTRY)}


def chain_ops(chain_id: int) -> tuple[str, ...]:
    """A chain id -> its ops. An id outside the registry raises: ids are positions, so
    an artifact built with another registry would otherwise decode into a wrong chain."""
    if not 0 <= chain_id < len(CHAIN_NAMES):
        raise ValueError(f"chain id {chain_id} outside the registry "
                         f"(0..{len(CHAIN_NAMES) - 1}): artifact and registry disagree, "
                         "rebuild it")
    return CHAIN_NAMES[chain_id]


def chain_id_of(ops: tuple[str, ...]) -> int:
    """The registry id of a chain."""
    return CHAIN_ID_OF[ops]


def chain_name(ops: tuple[str, ...]) -> str:
    """Stable name of a chain ('FERTILIZE+WATER+HARVEST', 'NO_ACTION')."""
    return "+".join(ops) if ops else "NO_ACTION"


def ops_of_name(name: str) -> tuple[str, ...]:
    """Inverse of `chain_name` (use names for anything that outlives a run)."""
    return tuple(name.split("+"))


# --- what a chain costs and yields ---------------------------------------------- #

def chain_labor(ops: tuple[str, ...]) -> int:
    """The worker hours a chain costs: one per op in `TILE_OPS`."""
    return sum(1 for op in ops if op in TILE_OPS)


#: Engine steps one op fills: a market buy rides along with a PASS, so neither costs a
#: turn here; the ops absent from this table cost one.
OP_STEPS: dict[str, int] = {UnitAction.PLANT.value: 2, UnitAction.FERTILIZE.value: 3,
                            UnitAction.FEED.value: 3, UnitAction.PLACE.value: 3}


def chain_steps(ops: tuple[str, ...]) -> int:
    """The turns a chain fills within one day: the engine gives a unit `turns_per_day`
    turns, so a chain needing more can never run."""
    return sum(OP_STEPS.get(op, 1) for op in ops if op not in MARKET_OPS)


def chain_requirements(entity: str | None, ops: tuple[str, ...]) -> dict[str, int]:
    """The cost of a chain: labour hours plus the inputs it spends.

    `entity` may be None for a chain that names none (a bare tile's PASS or DIG); the
    PLANT and PLACE branches require it and raise without it, because a silent fallback
    would price a seed nobody bought.
    """
    req: dict[str, int] = {RES_LABOR: chain_labor(ops)}
    for op in ops:
        if op == UnitAction.PLANT.value:
            key = SEED_RES.get(entity)
            if key is None:
                raise ValueError(f"chain {ops} plants, but entity {entity!r} is not a "
                                 "crop")
            req[key] = req.get(key, 0) + 1
        elif op == UnitAction.FERTILIZE.value:
            req[RES_FERTILIZER] = req.get(RES_FERTILIZER, 0) + 1
        elif op == UnitAction.FEED.value:
            req[RES_WHEAT] = req.get(RES_WHEAT, 0) + 1
        elif op == UnitAction.PLACE.value:
            key = ANIMAL_RES.get(entity)
            if key is None:
                raise ValueError(f"chain {ops} places an animal, but entity {entity!r} "
                                 f"is not one of {sorted(ANIMAL_RES)}")
            req[key] = req.get(key, 0) + 1
    return req


def cost_vector(entity: str | None, ops: tuple[str, ...]) -> list[int]:
    """The cost side of an edge: LABOR plus every input the chain spends."""
    vec = [0] * N_RESOURCE
    for res, units in chain_requirements(entity, ops).items():
        vec[RESOURCE_ID[res]] = units
    return vec


def produce_vector(entity: str | None, harvest: int, fert_collect: int) -> list[int]:
    """The produce side of an edge: harvest units of the entity's product, plus the
    fertilizer a COLLECT_FERTILIZER picked up. Never netted with the cost side."""
    vec = [0] * N_RESOURCE
    if harvest:
        if entity not in PRODUCT_RES:
            raise ValueError(f"chain yields a harvest, but entity {entity!r} has no "
                             "product")
        vec[RESOURCE_ID[PRODUCT_RES[entity]]] = harvest
    if fert_collect:
        vec[RESOURCE_ID[RES_FERTILIZER]] = fert_collect
    return vec


# --- entities -------------------------------------------------------------------- #

#: Crops then animals, 1..8, 0 = none. The artifact stores this code.
ENTITY_NAMES: tuple[str, ...] = tuple(CROPS) + tuple(ANIMALS)
ENTITY_CODE: dict[str, int] = {n: i + 1 for i, n in enumerate(ENTITY_NAMES)}
ENTITY_OF_CODE: dict[int, str] = {code: n for n, code in ENTITY_CODE.items()}


def is_animal(entity: str) -> bool:
    """Whether `entity` is one of the three species."""
    return entity in ANIMALS


def entity_code_of(entity: str | None) -> int:
    """The entity's code (0 = none). An unknown entity raises rather than coding as 0."""
    if entity is None:
        return 0
    if entity not in ENTITY_CODE:
        raise ValueError(f"unknown entity {entity!r}: not one of {ENTITY_NAMES}")
    return ENTITY_CODE[entity]


def entity_of_code(code: int) -> str | None:
    """Inverse of `entity_code_of` (0 = none; an out-of-range code raises)."""
    if code == 0:
        return None
    if not 0 < code <= len(ENTITY_NAMES):
        raise ValueError(f"entity code {code} is out of range 0..{len(ENTITY_NAMES)}")
    return ENTITY_OF_CODE[code]


def domain_ok(ops: tuple[str, ...], entity: str) -> bool:
    """False when `entity` must not run `ops`: a crop never runs an animal op, an animal
    never plants, and a structure must be the one the species lives in."""
    ops_set = set(ops)
    if is_animal(entity):
        if ops_set & CROP_ONLY_OPS:
            return False
        structure = ANIMAL_RULES[entity]["structure"]
        return not (ops_set & (set(BUILD_OF_STRUCTURE.values())
                               - {BUILD_OF_STRUCTURE[structure]}))
    return not (ops_set & ANIMAL_ONLY_OPS)


# --- what can run on a node ------------------------------------------------------ #

def _fert_window(crop: str) -> tuple[int, int]:
    """The ages at which a dose can still reach an effective day.

    The dose covers its own day plus two (kaggriculture.py:481). A one-shot crop earns
    it only through WATER inside its golden window, so the useful ages start at -2 and
    run to the window's end. An ongoing crop earns it on the production nights, which
    sit at ages `k * interval - 1`; the first is the night of age -1, so -3 is still
    useful and the last is the last such night.
    """
    spec = CROP_RULES[crop]
    if spec["ongoing"]:
        return -3, (int(spec["max_yield"]) - 1) * int(spec["interval"]) - 1
    return -2, int(spec["max_yield_day"]) - crop_age_origin(crop)


def _harvest_min_age(crop: str) -> int:
    """The first age at which the engine lets HARVEST succeed (:453).

    The engine refuses HARVEST while `day - planted_day < first_yield_day`, and our age
    counts from `crop_age_origin` — the start of the golden window for a one-shot crop.
    MELON's window opens on day 6 while its first yield day is 10, so without this the
    model would call ages 0..3 harvestable and the engine would refuse them.
    """
    return int(CROP_RULES[crop]["first_yield_day"]) - crop_age_origin(crop)


def _applicable(ops: tuple[str, ...], age: int | None, yield_units: int | None,
                entity: str | None) -> bool:
    """False when the chain is a guaranteed no-op on this node."""
    if "CARE" in ops and "FEED" not in ops:
        return False
    if entity is not None and not domain_ok(ops, entity):
        return False
    crop = entity if entity in CROPS else None
    if "FERTILIZE" in ops and age is not None and crop is not None:
        low, high = _fert_window(crop)
        if not low <= age <= high:
            return False
    if "HARVEST" in ops:
        if yield_units is not None and yield_units <= 0:
            return False                    # harvesting zero units burns the hour
        if age is not None and crop is not None and age < _harvest_min_age(crop):
            return False                    # nothing is produced before the first day
        if age is not None and crop is None and age < 0:
            return False                    # an animal produces from cycle age 0 on
    return True


def chains_for(kind: TileKind, age: int | None = None,
               yield_units: int | None = None, entity: str | None = None
               ) -> list[tuple[str, ...]]:
    """The applicable chains of a node, before pruning.

    The selection is by the tile's KIND only; `age`, `yield_units` and `entity` are the
    node's own, so the windows come from the engine's tables and a chain the entity
    cannot run is filtered out rather than never offered.
    """
    if kind is TileKind.NONE:
        base = NONE_CHAINS
    elif kind is TileKind.PLANT:
        base = CROP_CHAINS_YOUNG if (age is not None and age < 0) else CROP_CHAINS
    elif kind in CHAINS_BY_KIND:
        base = CHAINS_BY_KIND[kind]
    else:
        return []
    return [c for c in base if _applicable(c, age, yield_units, entity)]


# --- the artifact's identity ----------------------------------------------------- #

def registry_fingerprint() -> str:
    """Fingerprint of the chain registry.

    Chain ids are positions in `CHAIN_NAMES`, so they shift whenever a list changes, and
    an artifact stores ids. The graph writes this fingerprint and refuses to load an
    artifact built from another registry instead of decoding ids into the wrong chains.
    """
    from hashlib import sha256
    return sha256("\n".join(chain_name(c) for c in CHAIN_NAMES).encode()).hexdigest()[:16]


def engine_fingerprint() -> str:
    """Fingerprint of the engine source the graph decodes against: the rules the graph
    encodes (windows, dose, day length) live there, so a changed engine must invalidate
    an artifact built from the old ones."""
    from hashlib import sha1
    from pathlib import Path

    from kaggle_environments.envs.kaggriculture import kaggriculture as K
    return sha1(Path(K.__file__).read_bytes()).hexdigest()[:8]


def contract_id() -> str:
    """What an artifact IS, computed from everything it depends on: the registry, the
    engine, the day length, the key layout, and the builder's own source (a pruning or
    filter change must invalidate an old artifact exactly as an engine change does)."""
    from hashlib import sha1
    from pathlib import Path

    from agent.tile_dp.tile_state import KEY_BITS
    builder = Path(__file__).resolve().parents[2] / "offline_lab" / "build" / "graph.py"
    return (f"tile-dp/reg={registry_fingerprint()}"
            f"+eng={engine_fingerprint()}+tpd={TURNS_PER_DAY}+pb={KEY_BITS}"
            f"+bld={sha1(builder.read_bytes()).hexdigest()[:8]}")


# Contract: every chain fits one engine day (the engine gives a unit `turns_per_day`
# turns, so this is the limit that can actually fire — a worker-hour count could not).
_OVERLONG = tuple(c for c in CHAIN_NAMES if chain_steps(c) > TURNS_PER_DAY)
if _OVERLONG:
    raise AssertionError(f"chains needing more than {TURNS_PER_DAY} steps (one day): "
                         f"{_OVERLONG}")
