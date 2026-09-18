"""tile_dp chains: v16 - per-tile daily action chains (DIG-layered).

A chain is the ordered tuple of WORKER ops applied on one tile in one day.

Resource vocabulary (v16, contract 2026-09-14): 18 names, exactly one id per
physical item - LABOR_HOURS, FERTILIZER, WHEAT, SEED_WHEAT, SEED_CARROT,
SEED_TOMATO, SEED_STRAWBERRY, SEED_MELON, CARROT, TOMATO, STRAWBERRY, MELON,
EGG, MILK, WOOL, ANIMAL_GOOSE, ANIMAL_COW, ANIMAL_SHEEP. RES_WHEAT is the
engine's own product id "WHEAT" (it used to be `WHEAT_FOOD`) and the generic
RES_ANIMAL is gone: PLACE always name the species. PRODUCT_RES
maps an entity to the resource it yields (K.CROPS[name].get("product", name),
K.ANIMALS[name]["product"]).

Cost and produce are SEPARATE vectors (contract 2026-09-14): `chain_requirements`
is the cost side (LABOR_HOURS + the inputs) and `produce_vector` the produce side
(harvest units of the entity's product + collected fertilizer). A resource id may
appear on both sides - wheat is the FEED input and the WHEAT crop's product - and
the two sides are never netted nor collapsed into one vector.

Cost model (contract, 2026-09-14):
  * labour = number of ops that are in WORKER_OPS (allow-list below). Market
    buys are the market's action and a PICKUP is a carry of the day
    layer, so both cost 0 hours: supplying the inputs (seed / fertilizer /
    wheat / animal) is the SECRETARY layer's job.
  * NO_ACT = the worker does nothing on this tile (0 hours), but the day still
    passes and the tile state advances by one day. The engine's PASS action
    costs 1 hour and is deliberately NOT used in chain definitions.
  * requirements per op: PLANT -> 1 seed of that crop, FERTILIZE -> 1
    fertilizer, FEED -> 1 wheat, PLACE -> 1 animal of the
    entity's species (ANIMAL_GOOSE / ANIMAL_COW / ANIMAL_SHEEP).
  * `chain_requirements` is the single source of truth: it returns the labour
    hours (RES_LABOR) plus the inputs; `chain_labor` is a view of the same
    count and nothing outside counts ops again.

SECRETARY GAP: for a pure graph search every chain must also supply its
prerequisites (buy seed/fertilizer/wheat/animal and carry it to the tile). That
is the day layer, which does not exist yet, so the graph executor realises
the purchases inline as a stand-in. When the day appears, this is where
the split happens.

Applicability (2026-09-14): `chains_for` drops chains that can do nothing on the
node, and the age windows come from the engine's own crop tables (owner's items
2 and 8), not from a constant:
  * HARVEST only from `first_yield_day - crop_age_origin` on: the engine refuses
    it before the first yield day, and for MELON the golden window opens four
    days earlier than that first yield day.
  * FERTILIZE only inside `_fert_window`: the dose covers its own day plus two
    (kaggriculture.py:481) and must be able to reach an effective day - the
    golden window for a one-shot crop, a production night for an ongoing one
    (whose first night is the night of age -1, so age -3 is still useful).
  * HARVEST is dropped when `yield_units == 0` (it would only burn the hour).
  * CARE only together with FEED (the engine consumes `cared_today` with
    `fed_today`: a CARE-only chain is a one-hour no-op).

graph.py now does both ends of this: it prices every chain through
`cost_vector` / `produce_vector` and passes each node's own age and yield_units
to `chains_for`.

Entity (v16, decision 10): `ENTITY_NAMES` / `ENTITY_CODE` give each entity a small
int code (1..8, 0 = none) for the artifact's `entity_code` field and
`CONSTRUCTIVE_OPS` names the ops that parameterise a chain by entity (PLANT /
BUILD / PLACE). `domain_ok` is the domain filter: a crop entity
never runs an animal op, an animal entity never plants.
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

from kaggle_environments.envs.kaggriculture import kaggriculture as K

from tile_dp.tile_state import (EMPTY_KIND_OF_STRUCTURE, EMPTY_KINDS,
                                KIND_ANIMAL, KIND_EMPTY_COOP,
                                KIND_EMPTY_PASTURE, KIND_NONE, KIND_PLANT,
                                KIND_WEED, KEY_BITS, TURNS_PER_DAY,
                                crop_age_origin)

RES_LABOR = "LABOR_HOURS"
RES_FERTILIZER = "FERTILIZER"
RES_SEED_WHEAT = "SEED_WHEAT"
RES_SEED_CARROT = "SEED_CARROT"
RES_SEED_TOMATO = "SEED_TOMATO"
RES_SEED_STRAWBERRY = "SEED_STRAWBERRY"
RES_SEED_MELON = "SEED_MELON"
RES_WHEAT = "WHEAT"        # the engine's own product id: 1 wheat per FEED AND
                           # the WHEAT crop's harvested product (cost and
                           # produce keep their own vectors, never netted)
RES_CARROT = "CARROT"
RES_TOMATO = "TOMATO"
RES_STRAWBERRY = "STRAWBERRY"
RES_MELON = "MELON"
RES_EGG = "EGG"
RES_MILK = "MILK"
RES_WOOL = "WOOL"
RES_ANIMAL_GOOSE = "ANIMAL_GOOSE"
RES_ANIMAL_COW = "ANIMAL_COW"
RES_ANIMAL_SHEEP = "ANIMAL_SHEEP"
# 1 animal per PLACE, species taken from the entity (2026-09-14)
ANIMAL_RES = {"GOOSE": RES_ANIMAL_GOOSE, "COW": RES_ANIMAL_COW,
              "SHEEP": RES_ANIMAL_SHEEP}
# 1 seed per PLANT, crop taken from the entity (same names as tile_state)
SEED_RES = {"WHEAT": RES_SEED_WHEAT, "CARROT": RES_SEED_CARROT,
            "TOMATO": RES_SEED_TOMATO, "STRAWBERRY": RES_SEED_STRAWBERRY,
            "MELON": RES_SEED_MELON}

# The one product name per entity (v16, decision 3): a crop keeps its own name,
# an animal yields the engine's product id.
PRODUCT_RES: dict[str, str] = {c: K.CROPS[c].get("product", c) for c in K.CROPS}
PRODUCT_RES.update({a: K.ANIMALS[a]["product"] for a in K.ANIMALS})

# The one entity vocabulary of the artifact (v16, decision 10): crops then
# animals, 1..8 in that order, 0 = no entity.
ENTITY_NAMES: tuple[str, ...] = tuple(K.CROPS) + tuple(K.ANIMALS)
ENTITY_CODE: dict[str, int] = {n: i + 1 for i, n in enumerate(ENTITY_NAMES)}
ENTITY_OF_CODE: dict[int, str] = {code: n for n, code in ENTITY_CODE.items()}

# 18 names, exactly one id per physical item (v16, decision 3).
RESOURCE_NAMES: tuple[str, ...] = (RES_LABOR, RES_FERTILIZER, RES_WHEAT,
                                   RES_SEED_WHEAT, RES_SEED_CARROT,
                                   RES_SEED_TOMATO, RES_SEED_STRAWBERRY,
                                   RES_SEED_MELON, RES_CARROT, RES_TOMATO,
                                   RES_STRAWBERRY, RES_MELON, RES_EGG,
                                   RES_MILK, RES_WOOL, RES_ANIMAL_GOOSE,
                                   RES_ANIMAL_COW, RES_ANIMAL_SHEEP)
RESOURCE_ID: dict[str, int] = {n: i for i, n in enumerate(RESOURCE_NAMES)}
N_RESOURCE = len(RESOURCE_NAMES)

# The op vocabulary is `world/model.py`'s; these are views of it (ARCHITECTURE §5
# step 6), so a chain cannot name an op the model does not have.
from world.model import CHAIN_OPS as _CHAIN_OPS
from world.model import MARKET_ACTIONS as _MARKET_ACTIONS
from world.model import WORKER_OPS as _WORKER_OPS

NO_ACT = "NO_ACT"
# Market ops a chain may name: the buys it needs. The market's other actions
# (SELL, HIRE, BUY_LAND) are the market layer's, never a chain's.
MARKET_OPS: tuple[str, ...] = tuple(sorted(set(_MARKET_ACTIONS) & set(_CHAIN_OPS)))
# Worker ops = the only ops that cost hours: market buys are the market's action
# and a PICKUP is a carry of the day layer, so both cost 0 worker hours.
WORKER_OPS: frozenset[str] = frozenset(_WORKER_OPS)
# Every op a chain may name.
ALL_OPS: frozenset[str] = frozenset(_CHAIN_OPS)

CROP_OPS = ("FERTILIZE", "WATER", "HARVEST")
ANIMAL_OPS = ("FEED", "CARE", "HARVEST", "COLLECT_FERTILIZER")
# Ops the two domains do NOT share (v16, decision 9 domain filter): HARVEST is
# the only op both kinds have, so these are exactly the ones one side must never
# run. PLANT is the only crop-only op (an animal tile holds its animal).
ANIMAL_ONLY_OPS = frozenset(("BUILD", "PLACE", "FEED", "CARE",
                            "COLLECT_FERTILIZER"))
CROP_ONLY_OPS = frozenset(("PLANT",))
# Constructive ops (v16, decision 10): the ops that name the entity a chain
# creates, so they are what `entity_code` is read from. A chain has at most one
# domain's worth of them (the domain filter guarantees the match).
CONSTRUCTIVE_OPS = ("PLANT", "BUILD", "PLACE")


def _canonical_subsets(ops: tuple[str, ...]) -> list[tuple[str, ...]]:
    """All subsets of `ops` in canonical order; the empty one becomes NO_ACT."""
    return [tuple(op for op in ops if op in combo) or (NO_ACT,)
            for r in range(len(ops) + 1)
            for combo in combinations(ops, r)]


_CROP_SUBSETS: list[tuple[str, ...]] = _canonical_subsets(CROP_OPS)
# CARE is a no-op without FEED on the same day: the engine consumes
# `cared_today` only together with `fed_today` (kaggriculture.py:826-829), so the
# subset list never offers CARE alone (2026-09-14).
_ANIMAL_SUBSETS: list[tuple[str, ...]] = [
    c for c in _canonical_subsets(ANIMAL_OPS)
    if not ("CARE" in c and "FEED" not in c)]

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
#  ANIMAL          : subsets of {FEED, CARE, HARVEST, COLLECT_FERTILIZER};
#                    CARE only together with FEED (no-op otherwise)
#  EMPTY_COOP      : NO_ACT | PLACE     (+ the DIG layering); only the
#                    GOOSE candidates are tried on it
#  EMPTY_PASTURE   : NO_ACT | PLACE     (+ the DIG layering); only the
#                    COW / SHEEP candidates are tried on it
# BUILD_* need a NONE tile and DIG frees any of these back to NONE
# (kaggriculture.py:484-503, probed 2026-09-14): a structure appears only on a
# bare tile and disappears into one.
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
# DIG turns exactly these kinds back into NONE: the engine's DIG removes a
# plant, a weed or an EMPTY coop/pasture, refuses a tile that holds an animal
# and does nothing on a NONE tile (kaggriculture.py:484-491).
DIGGABLE_KINDS: frozenset[str] = frozenset((KIND_PLANT, KIND_WEED,
                                           KIND_EMPTY_COOP,
                                           KIND_EMPTY_PASTURE))


def _kind_after(kind: str, ops: tuple[str, ...],
                entity: str | None = None) -> str:
    """Tile kind after running `ops` on `kind` (known transitions only).

    A structure's kind follows the entity (BUILD_COOP vs BUILD_PASTURE), so the
    transition needs the entity that a constructive op names.
    """
    for op in ops:
        if op == "PLANT":
            kind = KIND_PLANT
        elif op == "BUILD":
            if entity in K.ANIMALS:
                kind = EMPTY_KIND_OF_STRUCTURE[K.ANIMALS[entity]["structure"]]
        elif op == "PLACE":
            kind = KIND_ANIMAL
        elif op == "DIG":
            kind = KIND_NONE
    return kind


def _dig_tail(kind: str, head: tuple[str, ...],
              entity: str | None = None) -> list[tuple[str, ...]]:
    """Variants that put DIG after `head` (empty head = DIG is the first op).

    Each variant ends with DIG (tile -> NONE) or with one full NONE chain.
    Nothing is dropped here for "the kind does not change" any more (owner
    2026-09-14): digging a crop and planting the same crop again is a real
    option (it restarts the lifecycle, which matters for a crop past its golden
    window), and digging an empty COOP to build a COOP again is a real cost. The
    variants that truly add nothing are removed by the graph's Pareto rule
    instead - same next state, componentwise no more cost - which is where the
    decision belongs.
    """
    if kind not in DIGGABLE_KINDS:
        return []
    before = _kind_after(kind, head, entity)
    if before not in DIGGABLE_KINDS:
        return []
    out: list[tuple[str, ...]] = [head + ("DIG",)]
    for follow in NONE_CHAINS:
        if follow == (NO_ACT,):
            continue
        out.append(head + ("DIG",) + follow)
    return out


def _layer(kind: str, base: tuple[tuple[str, ...], ...],
           entity: str | None = None) -> tuple[tuple[str, ...], ...]:
    """Base chains plus every legal DIG layering of them (dedup, order kept)."""
    out: list[tuple[str, ...]] = list(base)
    out += _dig_tail(kind, (), entity)            # DIG as the first op
    for chain in base:
        if chain and chain[-1] == "HARVEST":      # DIG right after HARVEST
            out += _dig_tail(kind, chain, entity)
    return tuple(dict.fromkeys(out))

# Every kind below is layered with DIG (`_layer`): base chains + DIG first +
# DIG right after HARVEST, with their NONE follow-ups.
WEED_CHAINS: tuple[tuple[str, ...], ...] = _layer(
    KIND_WEED, ((NO_ACT,), ("DIG",)))
# An empty structure takes an animal of ITS OWN structure only (the graph's
# candidates do that filter) and can be dug back to NONE; BUILD is absent here
# on purpose, the engine builds on a NONE tile only.
EMPTY_STRUCTURE_BASE: tuple[tuple[str, ...], ...] = ((NO_ACT,),
                                                    ("PLACE",))
EMPTY_COOP_CHAINS: tuple[tuple[str, ...], ...] = _layer(KIND_EMPTY_COOP,
                                                        EMPTY_STRUCTURE_BASE)
EMPTY_PASTURE_CHAINS: tuple[tuple[str, ...], ...] = _layer(
    KIND_EMPTY_PASTURE, EMPTY_STRUCTURE_BASE)
CROP_CHAINS: tuple[tuple[str, ...], ...] = _layer(
    KIND_PLANT, tuple(_CROP_SUBSETS))
CROP_CHAINS_YOUNG: tuple[tuple[str, ...], ...] = _layer(
    KIND_PLANT, tuple(c for c in _CROP_SUBSETS if "HARVEST" not in c))
ANIMAL_CHAINS: tuple[tuple[str, ...], ...] = _layer(
    KIND_ANIMAL, tuple(_ANIMAL_SUBSETS))

# The per-kind lists above are the contract; `chains_for` only selects from them
# and then filters. Registry ids stay internal (they can shift when a list
# changes) - use `chain_name` / `ops_of_name` for anything that outlives a run.
CHAINS_BY_KIND: dict[str, tuple[tuple[str, ...], ...]] = {
    KIND_NONE: NONE_CHAINS,
    KIND_WEED: WEED_CHAINS,
    KIND_PLANT: CROP_CHAINS,
    KIND_ANIMAL: ANIMAL_CHAINS,
    KIND_EMPTY_COOP: EMPTY_COOP_CHAINS,
    KIND_EMPTY_PASTURE: EMPTY_PASTURE_CHAINS,
}

_REGISTRY: list[tuple[str, ...]] = []
for c in (NONE_CHAINS_CROP + NONE_CHAINS_ANIMAL + WEED_CHAINS
          + EMPTY_COOP_CHAINS + EMPTY_PASTURE_CHAINS + CROP_CHAINS
          + ANIMAL_CHAINS):
    if c not in _REGISTRY:
        _REGISTRY.append(c)

CHAIN_NAMES: tuple[tuple[str, ...], ...] = tuple(_REGISTRY)
CHAIN_ID_OF: dict[tuple[str, ...], int] = {c: i for i, c in enumerate(_REGISTRY)}


def chain_ops(chain_id: int) -> tuple[str, ...]:
    """Decode a chain id into its op-name tuple (boundary function).

    An id outside the registry RAISES with the range in the message (owner's
    item 7): the id is a position in `CHAIN_NAMES`, so an artifact built with
    another registry would otherwise decode into a wrong chain or die with a
    bare IndexError far from the cause.
    """
    if not 0 <= chain_id < len(CHAIN_NAMES):
        raise ValueError(f"chain id {chain_id} outside the registry "
                         f"(0..{len(CHAIN_NAMES) - 1}): artifact and registry "
                         "disagree, rebuild it")
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


# Engine steps one op costs the executor: a market buy rides along with a PASS,
# a PICKUP is its own step and the act itself is the last one. Ops absent here
# cost exactly one step (WATER, CARE, DIG, HARVEST, COLLECT_FERTILIZER); NO_ACT
# is the idle chain and asks no worker for anything, so it costs none.
OP_STEPS: dict[str, int] = {"PLANT": 2, "FERTILIZE": 3, "FEED": 3,
                            "PLACE": 3, "PLACE": 3, NO_ACT: 0}


def chain_steps(ops: tuple[str, ...]) -> int:
    """Engine steps a chain needs, i.e. the turns it fills within one day.

    Owner's item 14: the registry contract has to bound the DAY the chain
    occupies, not only the worker-hour count - the engine gives a farmer
    `turns_per_day` turns and a chain that needs more steps than that can never
    be run in one day.
    """
    return sum(OP_STEPS.get(op, 1) for op in ops if op not in MARKET_OPS)


def registry_fingerprint() -> str:
    """Fingerprint of the chain registry (owner's item 7).

    Chain ids are positions in `CHAIN_NAMES`, so they shift whenever a list
    changes (they did, twice, while this file was being fixed). An artifact
    stores ids, so it is only readable together with the exact registry that
    produced it: `TileGraph.save` writes this fingerprint and `load` refuses a
    mismatch instead of decoding the ids into the wrong chains.
    """
    from hashlib import sha256
    joined = "\n".join(chain_name(c) for c in CHAIN_NAMES)
    return sha256(joined.encode()).hexdigest()[:16]


def engine_fingerprint() -> str:
    """Fingerprint of the engine source this graph decodes against.

    The rules the graph encodes (harvest windows, fertilizer dose, day length,
    dry limit) live in kaggriculture.py: if that file changes, an artifact built
    from the old rules must not be reused. Hashing it turns that into a
    mechanical check instead of a promise.
    """
    from hashlib import sha1
    return sha1(Path(K.__file__).read_bytes()).hexdigest()[:8]


def contract_id() -> str:
    """What an artifact IS, computed - never a hand-typed version.

    v15/v16/v17 were labels bumped by hand at every edit: they said nothing
    about the content and could not detect a mismatch by themselves (owner,
    2026-09-14: why mint a version while the first state is still unfinished - a
    version is a property of the product, not of an iteration).
    Instead, an artifact carries the fingerprint of everything it depends on -
    the chain registry, the engine source, the day length, the key layout AND
    the builder's own logic (graph.py: the pruning rules, the applicability
    filters and the executor; review 2026-09-14: `3c8733a` moved the edge count
    by 1286 without touching a chain name, and a logic change must invalidate
    the old artifact the same way an engine change does) - so a changed input
    makes an old artifact unusable by construction. The first product (this
    tile lifecycle graph) is unfinished, so nothing is stamped with a version
    number yet.
    """
    from hashlib import sha1
    graph_src = (Path(__file__).resolve().parent / "graph.py").read_bytes()
    return (f"tile-dp/reg={registry_fingerprint()}"
            f"+eng={engine_fingerprint()}+tpd={TURNS_PER_DAY}+pb={KEY_BITS}"
            f"+bld={sha1(graph_src).hexdigest()[:8]}")


# Contract: every chain fits one engine day (owner's item 14). The guard that
# only looked at worker hours (8 at most, `chain_labor`) could never fire: the
# engine's own limit is the number of turns a day has.
_OVERLONG_CHAINS = tuple(c for c in CHAIN_NAMES
                         if chain_steps(c) > TURNS_PER_DAY)
if _OVERLONG_CHAINS:
    raise AssertionError(f"chains needing more than {TURNS_PER_DAY} steps "
                         f"(one day): {_OVERLONG_CHAINS}")


def chain_requirements(entity: str | None,
                       ops: tuple[str, ...]) -> dict[str, int]:
    """Cost of a chain: labour hours + input requirements.

    `entity` may be None for a chain that names no entity (a bare tile's NO_ACT
    or DIG); the PLANT / PLACE branches require it and raise without it.

    Single source of truth for the cost model (2026-09-14): the labour hours
    (RES_LABOR) are produced here too, so nobody outside has to count ops
    again. PLANT -> 1 seed of the entity's crop, FERTILIZE -> 1 fertilizer,
    FEED -> 1 wheat, PLACE -> 1 animal of the entity's species.
    """
    req: dict[str, int] = {RES_LABOR: _worker_hours(ops)}
    for op in ops:
        if op == "PLANT":
            key = SEED_RES.get(entity)
            if key is None:
                raise ValueError(f"chain {ops} plants, but entity {entity!r} "
                                 "is not a crop (v16: no silent fallback)")
            req[key] = req.get(key, 0) + 1
        elif op == "FERTILIZE":
            req[RES_FERTILIZER] = req.get(RES_FERTILIZER, 0) + 1
        elif op == "FEED":
            req[RES_WHEAT] = req.get(RES_WHEAT, 0) + 1
        elif op == "PLACE":
            key = ANIMAL_RES.get(entity)
            if key is None:
                raise ValueError(f"chain {ops} places an animal, but entity "
                                 f"{entity!r} is not one of {sorted(ANIMAL_RES)}")
            req[key] = req.get(key, 0) + 1
    return req


def is_animal(entity: str) -> bool:
    """True when `entity` is one of the three animal species (v16)."""
    return entity in K.ANIMALS


def domain_ok(ops: tuple[str, ...], entity: str) -> bool:
    """False when `entity` must not run `ops` (v16, decision 9 domain filter).

    A crop entity never builds / places / feeds / cares for an animal; an animal
    entity never plants. HARVEST belongs to both domains.
    """
    if is_animal(entity):
        return not (set(ops) & CROP_ONLY_OPS)
    return not (set(ops) & ANIMAL_ONLY_OPS)


def entity_code_of(entity: str | None) -> int:
    """Small int code of an entity name (0 = none).

    An unknown entity RAISES (owner, 2026-09-14): silently coding it as 0 mixed
    it up with "no entity at all".
    """
    if entity is None:
        return 0
    if entity not in ENTITY_CODE:
        raise ValueError(f"unknown entity {entity!r}: not one of {ENTITY_NAMES}")
    return ENTITY_CODE[entity]


def entity_of_code(code: int) -> str | None:
    """Inverse of `entity_code_of` (0 = none; an unknown code raises)."""
    if code == 0:
        return None
    if not 0 < code <= len(ENTITY_NAMES):
        raise ValueError(f"entity code {code} is out of range 0.."
                         f"{len(ENTITY_NAMES)}")
    return ENTITY_OF_CODE[code]


def cost_vector(entity: str | None, ops: tuple[str, ...]) -> list[int]:
    """`chain_requirements` as an N_RESOURCE cost vector (v16).

    The cost side of an edge: LABOR_HOURS plus every input the chain consumes.
    """
    vec = [0] * N_RESOURCE
    for res, units in chain_requirements(entity, ops).items():
        vec[RESOURCE_ID[res]] = units
    return vec


def produce_vector(entity: str | None, harvest: int,
                   fert_collect: int) -> list[int]:
    """The produce side of a chain as an N_RESOURCE vector (v16).

    Harvest units of the entity's product (PRODUCT_RES) plus the fertilizer a
    COLLECT_FERTILIZER op picked up. Kept separate from the cost vector on
    purpose: wheat, for one, is both the FEED input and the WHEAT crop's product.
    """
    vec = [0] * N_RESOURCE
    if harvest:
        vec[RESOURCE_ID[PRODUCT_RES[entity]]] = harvest
    if fert_collect:
        vec[RESOURCE_ID[RES_FERTILIZER]] = fert_collect
    return vec


def chain_name(ops: tuple[str, ...]) -> str:
    """Stable name of a chain ('FERTILIZE+WATER+HARVEST', 'NO_ACT')."""
    return "+".join(ops)


def ops_of_name(name: str) -> tuple[str, ...]:
    """Inverse of `chain_name` (use names for anything that outlives a run)."""
    return tuple(name.split("+"))


def _fert_window(spec: dict) -> tuple[int, int]:
    """Ages where a FERTILIZE can still reach an effective day (engine tables).

    The dose covers its own day plus two (kaggriculture.py:481). A one-shot crop
    earns it only through WATER inside its golden window, so the useful ages run
    from -2 up to the window's end. An ongoing crop earns it on the production
    nights, which sit at ages k * interval - 1 for k = 1..max_yield: the first
    one is the night of age -1, so the first useful age is -3 (owner's item 8:
    the old -2 bound dropped a real dose) and the last is the last such night.
    """
    if spec.get("ongoing"):
        last_night = (int(spec["max_yield"]) - 1) * int(spec["interval"]) - 1
        return -3, last_night
    return -2, int(spec["max_yield_day"]) - crop_age_origin(spec)


def _harvest_min_age(spec: dict) -> int:
    """First age at which the engine lets HARVEST succeed (kaggriculture.py:453).

    The engine refuses HARVEST while `day - planted_day < first_yield_day`, but
    the model's age counts from `crop_age_origin`, which for a one-shot crop is
    the START OF THE GOLDEN WINDOW: MELON's window opens on day 6 while its
    first yield day is 10, so ages 0..3 were "harvestable" in the model and
    refused by the engine (owner's item 2: 278 of the 377 MELON harvest edges
    were phantom).
    """
    return int(spec["first_yield_day"]) - crop_age_origin(spec)


def _applicable(ops: tuple[str, ...], age: int | None, yield_units: int | None,
                entity: str | None = None) -> bool:
    """False when the chain is a guaranteed no-op on the node (2026-09-14)."""
    if "CARE" in ops and "FEED" not in ops:
        return False    # CARE only counts with FEED (fed_today and cared_today)
    spec = K.CROPS.get(entity) if entity is not None else None
    if "FERTILIZE" in ops and age is not None and spec is not None:
        lo, hi = _fert_window(spec)
        if not lo <= age <= hi:
            return False    # the 3-day dose cannot reach an effective day
    if "HARVEST" in ops:
        if age is not None and spec is not None and age < _harvest_min_age(spec):
            return False    # nothing produced before the first yield day
        if age is not None and spec is None and age < 0:
            return False    # an animal produces from cycle age 0 on
        if yield_units is not None and yield_units <= 0:
            return False    # harvesting zero units only burns the hour
    return True


def chains_for(kind: str, age: int | None = None,
               animal_graph: bool = False,
               yield_units: int | None = None,
               entity: str | None = None) -> list[tuple[str, ...]]:
    """Applicable chains of a node kind BEFORE pruning, after the filters.

    `animal_graph=True` only SELECTS the animal subset of the single NONE list
    (NONE is one state; the graph decides which start chains it may run).
    `entity` picks the crop / animal spec the age windows come from, so a MELON
    node is not harvested before its own first yield day (owner's items 2, 8).
    """
    if kind == KIND_NONE:
        base = NONE_CHAINS_ANIMAL if animal_graph else NONE_CHAINS_CROP
    elif kind == KIND_PLANT:
        base = CROP_CHAINS_YOUNG if (age is not None and age < 0) \
            else CROP_CHAINS
    elif kind == KIND_ANIMAL:
        base = ANIMAL_CHAINS
    elif kind == KIND_WEED:
        base = WEED_CHAINS
    elif kind in EMPTY_KINDS:
        base = CHAINS_BY_KIND[kind]
    else:
        return []
    return [c for c in base if _applicable(c, age, yield_units, entity)]
