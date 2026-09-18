"""The tile chains: what a chain IS, how the built ones are loaded, and what they cost.

A chain is one worker-day on one tile: an ordered tuple of ops. It is not `PASS` — PASS is
a turn spent waiting, which the builder uses when it has to let a market purchase land —
and an idle day is its own case, `NO_ACTION`: the empty chain, no worker on the tile, the
tile simply goes on to the next day.

Which chains exist is NOT decided here. That is a build, and it lives in
`offline_lab/build/chains.py`, which writes its result as the artifact `tile_chains`. This
file holds the base definition, the loader, and the ledger — the things the agent needs at
runtime, and nothing that chooses.

The entity (a crop or a species) travels BESIDE a chain, not inside it: the same tuple
`("PLACE",)` is a goose on a coop or a cow on a pasture, and the edge that carries the
chain also carries `entity_code`. `actions_of` is where the two meet and an argument is
attached.
"""

from __future__ import annotations

import json
from pathlib import Path

from agent.artifact import artifact_path, info_path
from agent.world.action import WorkerAction
from agent.world.model import (ANIMALS, COLUMNS, CROPS, Animal, Column, Crop,
                               Structure, UnitAction)
from agent.world.rules import ANIMAL_RULES, TURNS_PER_DAY

# --- the base definition -------------------------------------------------------- #

#: The ops a chain may name, as the artifact stores them.
TileChain = tuple[str, ...]

#: One action of a chain: a unit op with its item. A chain is what happens ON A TILE, so
#: no market op is in one — buying is the market layer's decision, and it is the day layer
#: that turns a need into an order. The type is world's, so a chain action can never name
#: something the engine does not have.
TileChainAction = WorkerAction

#: The case with no action at all.
NO_ACTION: TileChain = ()

#: The ops that cost a worker hour: every unit op except PASS.
TILE_OPS: frozenset[str] = frozenset(
    op.value for op in UnitAction if op is not UnitAction.PASS)

#: The op that builds each structure: a chain names the structure its species lives in.
BUILD_OF_STRUCTURE: dict[Structure, str] = {Structure.COOP: UnitAction.BUILD_COOP.value,
                                            Structure.PASTURE: UnitAction.BUILD_PASTURE.value}

#: The ops that create the entity a chain is about, so they are what its code is read
#: from.
CONSTRUCTIVE_OPS: tuple[str, ...] = (UnitAction.PLANT.value,
                                     UnitAction.BUILD_COOP.value,
                                     UnitAction.BUILD_PASTURE.value,
                                     UnitAction.PLACE.value)

# --- the columns a plan is priced over ------------------------------------------ #

RESOURCE_NAMES: tuple[str, ...] = COLUMNS
RESOURCE_ID: dict[str, int] = {name: i for i, name in enumerate(RESOURCE_NAMES)}
N_RESOURCE = len(RESOURCE_NAMES)
RES_LABOR = Column.LABOR.value
RES_FERTILIZER = Column.FERTILIZER.value
RES_WHEAT = Column.WHEAT.value

#: Which column a PLANT / PLACE spends, and which one a harvest fills.
SEED_RES: dict[str, str] = {crop: f"SEED_{crop}" for crop in CROPS}
ANIMAL_RES: dict[str, str] = {animal: f"ANIMAL_{animal}" for animal in ANIMALS}
PRODUCT_RES: dict[str, str] = {crop: crop for crop in CROPS}
PRODUCT_RES.update({a: ANIMAL_RULES[a]["product"] for a in ANIMALS})

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


# --- the chains, loaded ---------------------------------------------------------- #

_ARTIFACT = "tile_chains"


def load_chains(name: str = _ARTIFACT) -> tuple[TileChain, ...]:
    """The built chains, from `agent/artifact/<name>.json`.

    The artifact is trusted: everything that checks it - the ops, the registry
    fingerprint, the engine, the day length - is a BUILD step's job, where it can afford
    the time. This is the runtime, so it reads the file and hands over the chains.
    """
    info = json.loads(info_path(name).read_text())
    data = json.loads((info_path(name).parent / info["file"]).read_text())
    return tuple(tuple(chain) for chain in data["chains"])


#: The registry, loaded on first use: the chains builder imports this module to write
#: the artifact, so the load cannot happen at import time.
_LOADED: tuple[tuple[TileChain, ...], dict[TileChain, int]] | None = None


def _registry() -> tuple[tuple[TileChain, ...], dict[TileChain, int]]:
    global _LOADED
    if _LOADED is None:
        chains = load_chains()
        _LOADED = (chains, {c: i for i, c in enumerate(chains)})
    return _LOADED


def chain_ops(chain_id: int) -> TileChain:
    """A chain id -> its ops. An id outside the registry raises: ids are positions, so an
    artifact built with another registry would decode into a wrong chain."""
    chains = _registry()[0]
    if not 0 <= chain_id < len(chains):
        raise ValueError(f"chain id {chain_id} outside the registry "
                         f"(0..{len(chains) - 1}): artifact and registry disagree, "
                         "rebuild it")
    return chains[chain_id]


def chain_id_of(ops: TileChain) -> int:
    """The registry id of a chain."""
    return _registry()[1][ops]


def chain_name(ops: TileChain) -> str:
    """Stable name of a chain ('FERTILIZE+WATER+HARVEST', 'NO_ACTION')."""
    return "+".join(ops) if ops else "NO_ACTION"


def ops_of_name(name: str) -> TileChain:
    """Inverse of `chain_name` (use names for anything that outlives a run)."""
    return tuple(name.split("+")) if name != "NO_ACTION" else NO_ACTION


def actions_of(ops: TileChain, entity: str | None = None) -> tuple[TileChainAction, ...]:
    """A chain's actions, each carrying its argument.

    The entity is what fills the argument of the ops that take one: PLANT names its crop
    and PLACE its species. The structure is already in the op (BUILD_COOP /
    BUILD_PASTURE), which is why an animal chain needs no more than its species here.
    """
    out: list[TileChainAction] = []
    for op in ops:
        if op == UnitAction.PLANT.value:
            out.append(WorkerAction(UnitAction.PLANT, Crop(entity) if entity else None))
        elif op == UnitAction.PLACE.value:
            out.append(WorkerAction(UnitAction.PLACE, Animal(entity) if entity else None))
        else:
            out.append(WorkerAction(UnitAction(op)))
    return tuple(out)


# --- what a chain costs and yields ------------------------------------------------ #

def chain_labor(ops: TileChain) -> int:
    """The worker hours a chain costs: one per op in `TILE_OPS`."""
    return sum(1 for op in ops if op in TILE_OPS)


#: Engine steps one op fills: the ops absent from this table cost one, and a market buy
#: rides along with a turn it does not spend.
OP_STEPS: dict[str, int] = {UnitAction.PLANT.value: 2, UnitAction.FERTILIZE.value: 3,
                            UnitAction.FEED.value: 3, UnitAction.PLACE.value: 3}


def chain_steps(ops: TileChain) -> int:
    """The turns a chain fills within one day: the engine gives a unit `turns_per_day`
    turns, so a chain needing more can never run."""
    return sum(OP_STEPS.get(op, 1) for op in ops)


def chain_requirements(entity: str | None, ops: TileChain) -> dict[str, int]:
    """The cost of a chain: labour hours plus the inputs it spends.

    `entity` may be None for a chain that names none (a bare tile's NO_ACTION or DIG); the
    PLANT and PLACE branches require it and raise without it, because a silent fallback
    would price a seed nobody bought.
    """
    req: dict[str, int] = {RES_LABOR: chain_labor(ops)}
    for op in ops:
        if op == UnitAction.PLANT.value:
            key = SEED_RES.get(entity)
            if key is None:
                raise ValueError(f"chain {ops} plants, but entity {entity!r} is not a crop")
            req[key] = req.get(key, 0) + 1
        elif op == UnitAction.FERTILIZE.value:
            req[RES_FERTILIZER] = req.get(RES_FERTILIZER, 0) + 1
        elif op == UnitAction.FEED.value:
            req[RES_WHEAT] = req.get(RES_WHEAT, 0) + 1
        elif op == UnitAction.PLACE.value:
            key = ANIMAL_RES.get(entity)
            if key is None:
                raise ValueError(f"chain {ops} places an animal, but entity {entity!r} is "
                                 f"not one of {sorted(ANIMAL_RES)}")
            req[key] = req.get(key, 0) + 1
    return req


def cost_vector(entity: str | None, ops: TileChain) -> list[int]:
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
            raise ValueError(f"chain yields a harvest, but entity {entity!r} has no product")
        vec[RESOURCE_ID[PRODUCT_RES[entity]]] = harvest
    if fert_collect:
        vec[RESOURCE_ID[RES_FERTILIZER]] = fert_collect
    return vec


# --- the artifact's identity ------------------------------------------------------ #

def __getattr__(name: str):
    """`CHAIN_NAMES` / `CHAIN_ID_OF`, loaded when they are first asked for.

    Defined last, and it refuses every private name: a module's own lookups during import
    must never come back through here.
    """
    if name.startswith("_"):
        raise AttributeError(name)
    if name in ("CHAIN_NAMES", "CHAIN_ID_OF"):
        return _registry()[0 if name == "CHAIN_NAMES" else 1]
    raise AttributeError(name)


def _fingerprint(chains: tuple[TileChain, ...]) -> str:
    """The registry fingerprint of a chain list: ids are positions, so this is what an
    artifact's info carries and what a build compares."""
    from hashlib import sha256
    return sha256("\n".join(chain_name(c) for c in chains).encode()).hexdigest()[:16]


def registry_fingerprint() -> str:
    """Fingerprint of the loaded registry: ids are positions, so an artifact that stores
    ids is only readable together with the exact registry that produced it."""
    from hashlib import sha256
    return _fingerprint(_registry()[0])


def engine_fingerprint() -> str:
    """Fingerprint of the engine source the graph decodes against."""
    from hashlib import sha1

    from kaggle_environments.envs.kaggriculture import kaggriculture as K
    return sha1(Path(K.__file__).read_bytes()).hexdigest()[:8]


def contract_id() -> str:
    """What an artifact IS, computed from everything it depends on: the registry, the
    engine, the day length, the key layout, and the builder's own source."""
    from hashlib import sha1
    from agent.tile_dp.tile_state import KEY_BITS
    builder = Path(__file__).resolve().parents[2] / "offline_lab" / "build" / "graph.py"
    chains_builder = (Path(__file__).resolve().parents[2] / "offline_lab" / "build"
                      / "chains.py")
    return (f"tile-dp/reg={registry_fingerprint()}"
            f"+eng={engine_fingerprint()}+tpd={TURNS_PER_DAY}+pb={KEY_BITS}"
            f"+bld={sha1(builder.read_bytes()).hexdigest()[:8]}"
            f"+chn={sha1(chains_builder.read_bytes()).hexdigest()[:8]}")
