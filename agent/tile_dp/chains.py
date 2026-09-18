"""The tile chains: what a chain IS, and how the built ones are loaded.

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
from agent.world.model import (ANIMALS, CROPS, Animal, Crop, Structure,
                               UnitAction)
from agent.world.rules import ANIMAL_RULES

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


#: A chain is stored as ONE integer, 4 bits per op, and a zero nibble ends it. The ops are
#: named once in the artifact's op table, so hundreds of chains cost a few KB instead of a
#: list of op-name strings each.
OP_BITS = 4
OP_MASK = (1 << OP_BITS) - 1


def pack_chain(ops: tuple[str, ...], codes: dict[str, int]) -> int:
    """A chain as one integer. Codes start at 1, so a zero nibble can only be the end.

    The op table is 1-based on purpose: 0 is the END sentinel, so the empty chain
    (NO_ACTION) is the only chain whose code is 0 and no real chain can be mistaken for it.
    A vocabulary-wide table would break this - 17 ops do not fit 15 codes in one nibble -
    which is why the table names only the ops the chains actually use.
    """
    if len(ops) * OP_BITS >= 64:
        raise ValueError(f"chain too long to pack: {ops}")
    for op in ops:
        if not 0 < codes[op] <= OP_MASK:
            raise ValueError(f"op {op!r} has code {codes[op]} outside one nibble")
    value = 0
    for i, op in enumerate(ops):
        value |= codes[op] << (OP_BITS * i)
    return value


def unpack_chain(value: int, names: tuple[str, ...]) -> TileChain:
    """The ops of a packed chain, stopping at the END sentinel (a zero nibble)."""
    ops = []
    while value:
        code = value & OP_MASK
        if code == 0:
            break
        ops.append(names[code - 1])
        value >>= OP_BITS
    return tuple(ops)


def load_chains(name: str = _ARTIFACT) -> tuple[TileChain, ...]:
    """The built chains, from `agent/artifact/<name>.json`.

    The artifact is trusted: everything that checks it - the ops, the registry
    fingerprint, the engine, the day length - is a BUILD step's job, where it can afford
    the time. This is the runtime, so it reads the file and hands over the chains.
    """
    info = json.loads(info_path(name).read_text())
    data = json.loads((info_path(name).parent / info["file"]).read_text())
    names = tuple(data["ops"])
    return tuple(unpack_chain(int(v), names) for v in data["chains"])


#: The registry, loaded on first use: the chains builder imports this module to write
#: the artifact, so the load cannot happen at import time.
_LOADED: tuple[tuple[TileChain, ...], dict[TileChain, int]] | None = None
_LOADED_MTIME: float = -1.0


def _registry() -> tuple[tuple[TileChain, ...], dict[TileChain, int]]:
    global _LOADED, _LOADED_MTIME
    stamp = info_path(_ARTIFACT).stat().st_mtime
    if _LOADED is None or stamp != _LOADED_MTIME:
        _LOADED_MTIME = stamp
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


def loaded_chains() -> tuple[TileChain, ...]:
    """The chains the artifact holds (loaded once)."""
    return _registry()[0]


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
