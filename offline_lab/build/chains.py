"""Build the tile chains and write them as an artifact. Offline only.

A chain is one worker-day on one tile, and what a day can be is a state machine over the
tile's KIND, because every action can change it:

    NONE           -> BUILD_COOP, BUILD_PASTURE, PLANT <crop>
    EMPTY_COOP     -> PLACE <goose>, DIG
    EMPTY_PASTURE  -> PLACE <cow|sheep>, DIG
    WEED           -> DIG
    PLANT          -> FERTILIZE, WATER, HARVEST          (DIG first, or after HARVEST)
    ANIMAL         -> FEED, CARE, COLLECT_FERTILIZER, HARVEST

So PLANT makes the tile a PLANT and the rest of the day may water and harvest it; BUILD_*
makes it a structure and PLACE may fill it; DIG makes it bare and a bare-tile action may
follow. Two rules keep the set canonical instead of combinatorial noise:

  * the ORDER inside a day is fixed where it matters and used everywhere so a chain is
    never listed twice: PLANT > FERTILIZE > WATER > HARVEST, and for animals
    FEED > CARE > COLLECT_FERTILIZER > HARVEST (the engine does not care, but the
    canonical order is what stops duplicate chains);
  * DIG is either the first op or comes right after HARVEST - the engine refuses DIG on a
    tile that holds an animal, and building a structure needs a bare tile
    (kaggriculture.py:484-503).

The RESULT is `agent/artifact/tile_chains`, which is all the agent loads: the agent
defines what a chain IS, and never how one was chosen.

The table is written by `offline_lab.build.graph`, in the same run as the graph:

    .venv/bin/python -m offline_lab.build.graph
"""

from __future__ import annotations

import json
from pathlib import Path

from agent.artifact import artifact_path, write_info
from agent.tile_dp import chains as base
from agent.tile_dp.chains import (BUILD_OF_STRUCTURE, NO_ACTION, TILE_OPS, chain_name, entity_code_of)
from offline_lab.build.ledger import (ANIMAL_RES, PRODUCT_RES, SEED_RES)
from agent.tile_dp.contract import (engine_fingerprint)
from agent.world.model import (ANIMALS, CROPS, RESOURCE_NAMES, Structure, TileKind,
                               UnitAction)
from agent.world.rules import ANIMAL_RULES, CROP_RULES, TURNS_PER_DAY
from agent.world.tile import crop_age_origin

NAME = "tile_chains"
TABLE_PATH = artifact_path(NAME, ".data.json")


def registry_fingerprint(chains: tuple[tuple[str, ...], ...]) -> str:
    """Fingerprint of the chains THIS build produced (the agent's own one hashes the
    registry it loaded, which does not exist yet while this runs)."""
    return base.fingerprint_chains(chains)


def contract_of(chains: tuple[tuple[str, ...], ...]) -> str:
    """The contract this build stamps: the registry it made, the engine, and its own
    source."""
    from hashlib import sha1
    from agent.tile_dp.tile_state import KEY_BITS
    return (f"tile-dp/reg={registry_fingerprint(chains)}"
            f"+eng={engine_fingerprint()}+tpd={TURNS_PER_DAY}+pb={KEY_BITS}"
            f"+chn={sha1(Path(__file__).read_bytes()).hexdigest()[:8]}")


# --- what a day can be ------------------------------------------------------------ #

#: The single actions each KIND allows, in the canonical order.
ACTIONS_BY_KIND: dict[TileKind, tuple[str, ...]] = {
    TileKind.NONE: (UnitAction.BUILD_COOP.value, UnitAction.BUILD_PASTURE.value,
                    UnitAction.PLANT.value),
    TileKind.EMPTY_COOP: (UnitAction.PLACE.value, UnitAction.DIG.value),
    TileKind.EMPTY_PASTURE: (UnitAction.PLACE.value, UnitAction.DIG.value),
    TileKind.WEED: (UnitAction.DIG.value,),
    TileKind.PLANT: (UnitAction.FERTILIZE.value, UnitAction.WATER.value,
                     UnitAction.HARVEST.value, UnitAction.DIG.value),
    TileKind.ANIMAL: (UnitAction.FEED.value, UnitAction.CARE.value,
                      UnitAction.COLLECT_FERTILIZER.value, UnitAction.HARVEST.value),
}

#: The order a chain must keep: an op may only follow one of a lower or equal rank. The
#: engine does not need it; it is what keeps a chain from being listed twice.
RANK: dict[str, int] = {
    UnitAction.PLANT.value: 0, UnitAction.BUILD_COOP.value: 0,
    UnitAction.BUILD_PASTURE.value: 0,
    UnitAction.PLACE.value: 1, UnitAction.FERTILIZE.value: 1, UnitAction.FEED.value: 1,
    UnitAction.CARE.value: 2, UnitAction.WATER.value: 2,
    UnitAction.COLLECT_FERTILIZER.value: 3,
    UnitAction.HARVEST.value: 4,
    UnitAction.DIG.value: 5,
}

#: The kinds a structure or an animal belongs to, per structure.
_KIND_OF_STRUCTURE = {Structure.COOP: TileKind.EMPTY_COOP,
                      Structure.PASTURE: TileKind.EMPTY_PASTURE}


def _kinds_after(kind: TileKind, op: str, ongoing: bool) -> tuple[TileKind, ...]:
    """The kind(s) the tile can be in once `op` has run on `kind`.

    HARVEST has two: a ONE-SHOT crop is cleared by it, so the tile becomes bare and the
    same day may plant again (kaggriculture.py:464-468), while an ongoing crop and an
    animal keep their tile. The chains do not know which crop they are about - the entity
    is applied later - so the walk takes both branches and the filter keeps the one the
    entity allows.
    """
    if op == UnitAction.PLANT.value:
        return (TileKind.PLANT,)
    if op == UnitAction.BUILD_COOP.value:
        return (TileKind.EMPTY_COOP,)
    if op == UnitAction.BUILD_PASTURE.value:
        return (TileKind.EMPTY_PASTURE,)
    if op == UnitAction.PLACE.value:
        return (TileKind.ANIMAL,)
    if op == UnitAction.DIG.value:
        return (TileKind.NONE,)
    if op == UnitAction.HARVEST.value and kind is TileKind.PLANT:
        # A ONE-SHOT crop is cleared by its harvest, so the tile is bare and the same day may
        # plant again (kaggriculture.py:464-468). An ONGOING crop keeps its tile: it is not
        # bare, so planting after it is refused and the tile turns to WEED.
        return (kind,) if ongoing else (TileKind.NONE,)
    return (kind,)


def _day_chains(kind: TileKind, ops: tuple[str, ...] = (), rank: int = -1,
                steps: int = 0, used: frozenset[str] = frozenset(),
                restarted: bool = False, ongoing: bool = False) -> list[tuple[str, ...]]:
    """Every day this kind can have, as the sequences the rules allow.

    Recursive over the day: at each step the tile is in some kind, one of that kind's
    actions may run, and the kind moves on. The canonical ORDER is kept WITHIN a
    kind-episode - the rank may not drop while the tile stays the same - because a change
    of kind starts a new episode: after a DIG the tile is bare, so PLANT (rank 0) may
    follow DIG (rank 5), and after a one-shot HARVEST the tile is bare the same way.

    DIG is either the first op or right after HARVEST, and an op is done once in a day. A
    chain is any prefix of such a walk, which is what makes an incomplete day - the worker
    stops early - a chain of its own.
    """
    out: list[tuple[str, ...]] = [ops]
    if steps >= TURNS_PER_DAY:
        return out
    for op in ACTIONS_BY_KIND.get(kind, ()):
        if op in used:
            continue          # once per tile-LIFE: a new plant may be watered again
        if RANK[op] < rank:
            continue          # the order, inside this kind-episode
        if op == UnitAction.DIG.value:
            if ops and ops[-1] != UnitAction.HARVEST.value:
                continue      # DIG first, or right after a HARVEST
        for nxt in _kinds_after(kind, op, ongoing):
            # A change of kind replaces the tile, so its daily flags start clean: the order
            # and the once-per-life set begin again. Only ONE such restart is allowed in a
            # day - a tile may be dug and replanted, but a chain that replants and harvests
            # and replants and harvests is not a plan, it is a loop, and the registry must
            # not carry thousands of them.
            restart = nxt is TileKind.NONE and kind is not TileKind.NONE
            if restart and restarted:
                continue
            new_rank = RANK[op] if nxt is kind else -1
            new_used = (used | {op}) if nxt is kind else frozenset({op})
            out += _day_chains(nxt, ops + (op,), new_rank, steps + _steps(op), new_used,
                               restarted or restart, ongoing)
    return out


#: Engine steps one op fills (a chain may not need more than a day has turns).
_STEPS_PER_OP: dict[str, int] = {UnitAction.PLANT.value: 2, UnitAction.FERTILIZE.value: 3,
                                 UnitAction.FEED.value: 3, UnitAction.PLACE.value: 3}


def _steps(op: str) -> int:
    return _STEPS_PER_OP.get(op, 1)


def chains_of_kind(kind: TileKind, ongoing: bool = False) -> tuple[tuple[str, ...], ...]:
    """The chains a tile of this kind can have, `NO_ACTION` first (dedup, order kept).

    A day that PLANTS is a day that WATERS - after the planting, not before it: a crop left
    dry on its planting day turns to WEED by the next day.
    """
    out = [NO_ACTION]
    for chain in _day_chains(kind, ongoing=ongoing):
        if not chain or chain in out:
            continue
        # A day that plants WATERS the new plant: the watering must come AFTER the PLANT,
        # because a crop left dry on its planting day is lost - measured: plant, fertilise,
        # walk away, and the tile is WEED by the next day. Watering before the plant does
        # not count.
        if UnitAction.PLANT.value in chain:
            after = chain[chain.index(UnitAction.PLANT.value) + 1:]
            if UnitAction.WATER.value not in after:
                continue
        out.append(chain)
    return tuple(out)


def registry() -> tuple[tuple[str, ...], ...]:
    """The chains, in registry order (ids are positions in this list)."""
    out: list[tuple[str, ...]] = []
    for kind in (TileKind.NONE, TileKind.WEED, TileKind.EMPTY_COOP,
                 TileKind.EMPTY_PASTURE, TileKind.ANIMAL):
        for chain in chains_of_kind(kind):
            if chain not in out:
                out.append(chain)
    for chains in CROP_CHAINS.values():
        for chain in chains:
            if chain not in out:
                out.append(chain)
    return tuple(out)


#: The kinds whose days do not depend on which crop is standing there.
CHAINS_BY_KIND: dict[TileKind, tuple[tuple[str, ...], ...]] = {
    kind: chains_of_kind(kind) for kind in
    (TileKind.NONE, TileKind.WEED, TileKind.EMPTY_COOP, TileKind.EMPTY_PASTURE,
     TileKind.ANIMAL)}

#: A PLANT tile's days, per crop nature: what may follow a harvest differs, so these are two
#: lists and never one (owner's finding).
CROP_CHAINS: dict[bool, tuple[tuple[str, ...], ...]] = {
    True: chains_of_kind(TileKind.PLANT, ongoing=True),
    False: chains_of_kind(TileKind.PLANT, ongoing=False)}

#: A young plant cannot be harvested yet: the same lists, without HARVEST.
CROP_CHAINS_YOUNG: dict[bool, tuple[tuple[str, ...], ...]] = {
    key: tuple(c for c in chains if UnitAction.HARVEST.value not in c)
    for key, chains in CROP_CHAINS.items()}


# --- the filters ------------------------------------------------------------------ #

ANIMAL_ONLY_OPS: frozenset[str] = frozenset((
    UnitAction.BUILD_COOP.value, UnitAction.BUILD_PASTURE.value, UnitAction.PLACE.value,
    UnitAction.FEED.value, UnitAction.CARE.value, UnitAction.COLLECT_FERTILIZER.value))
CROP_ONLY_OPS: frozenset[str] = frozenset((UnitAction.PLANT.value,))


def _fert_window(crop: str) -> tuple[int, int]:
    """The ages at which a dose can still reach an effective day (kaggriculture.py:481)."""
    spec = CROP_RULES[crop]
    if spec["ongoing"]:
        return -3, (int(spec["max_yield"]) - 1) * int(spec["interval"]) - 1
    return -2, int(spec["max_yield_day"]) - crop_age_origin(crop)


def _harvest_min_age(crop: str) -> int:
    """The first age at which the engine lets HARVEST succeed (:453)."""
    return int(CROP_RULES[crop]["first_yield_day"]) - crop_age_origin(crop)


def domain_ok(ops: tuple[str, ...], entity: str,
              structure: Structure | None = None) -> bool:
    """False when `entity` must not run `ops`.

    A crop never runs an animal op and an animal never plants. A structure must be the one
    the species lives in - with the engine's own exception: BUILD_COOP needs a bare tile,
    so changing a barn into a coop is only possible by DIGGING it first (:493-503).
    """
    ops_set = set(ops)
    if entity in ANIMALS:
        if ops_set & CROP_ONLY_OPS:
            return False
        # Compare STRUCTURES, never op names: the state carries "COOP", the chain carries
        # "BUILD_COOP", and the animal's rule carries the structure itself.
        as_name = lambda v: getattr(v, "value", v)
        own_struct = as_name(ANIMAL_RULES[entity]["structure"])
        builds = {op for op in ops_set if op.startswith("BUILD_")}
        built = {op[len("BUILD_"):] for op in builds}
        here = as_name(structure)
        # Where the animal would land: the structure the chain builds, else the one the
        # state already has. PLACE only works in the species' own structure - the engine
        # refuses a goose in a pasture, so a chain that promises one is a lie.
        lands = own_struct if own_struct in built else next(iter(built), here)
        if UnitAction.PLACE.value in ops_set and lands is not None and lands != own_struct:
            return False
        other = builds - {f"BUILD_{own_struct}"}
        if not other:
            return True
        return (UnitAction.DIG.value in ops
                and ops.index(UnitAction.DIG.value) < min(ops.index(op) for op in other))
    return not (ops_set & ANIMAL_ONLY_OPS)


def applicable(ops: tuple[str, ...], age: int | None, yield_units: int | None,
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
            return False
        if age is not None and crop is not None and age < _harvest_min_age(crop):
            return False
        if age is not None and crop is None and age < 0:
            return False
    return True


def chains_for(kind: TileKind, age: int | None = None, yield_units: int | None = None,
               entity: str | None = None) -> list[tuple[str, ...]]:
    """The applicable chains of a node: selected by KIND, then filtered by the node."""
    if kind is TileKind.PLANT:
        # Which crop is standing there decides both lists: its nature picks the harvest
        # branch, its age whether a harvest is possible at all.
        ongoing = True if entity is None else bool(CROP_RULES[entity]["ongoing"])
        young = age is not None and age < 0
        base_chains = (CROP_CHAINS_YOUNG if young else CROP_CHAINS)[ongoing]
    elif kind in CHAINS_BY_KIND:
        base_chains = CHAINS_BY_KIND[kind]
    else:
        return []
    return [c for c in base_chains if applicable(c, age, yield_units, entity)]


# --- write the artifact ------------------------------------------------------------ #

_INDEX: dict[tuple[str, ...], int] | None = None


def chain_id_of(ops: tuple[str, ...]) -> int:
    """The chain's index in the table THIS build writes.

    The builder must index by the registry it is about to write, not by the artifact on
    disk: the two are the same table only inside one run, and the agent's loader reads the
    file.
    """
    global _INDEX
    if _INDEX is None:
        _INDEX = {chain: i for i, chain in enumerate(registry())}
    return _INDEX[ops]


def chain_ops(chain_id: int) -> tuple[str, ...]:
    """The ops of an id in the table THIS build writes (the agent's loader reads the file)."""
    return registry()[chain_id]


def write_table(contract: str, chains=None) -> Path:
    """Write the chain table the agent loads, stamped with the GRAPH's contract.

    Called by the graph builder, never on its own: the graph's action indices ARE these
    chains, so the two artifacts are one build.
    """
    chains = tuple(registry() if chains is None else chains)
    overlong = [c for c in chains if base.chain_steps(c) > TURNS_PER_DAY]
    if overlong:
        raise AssertionError(f"chains needing more than {TURNS_PER_DAY} steps: {overlong}")
    TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    # Only the chains that survived pruning reach the artifact (the owner's rule: the
    # artifact carries the USEFUL actions), and they are packed: one integer per chain.
    # The op table is the ops these chains ACTUALLY use, not the whole vocabulary: one
    # nibble holds 15 codes and TILE_OPS has more, so a vocabulary-wide table collided
    # codes and corrupted every chain on the way back in.
    names = tuple(sorted({op for chain in chains for op in chain}))
    if len(names) > (1 << base.OP_BITS) - 1:
        raise AssertionError(f"{len(names)} ops do not fit {base.OP_BITS} bits")
    codes = {op: i + 1 for i, op in enumerate(names)}
    TABLE_PATH.write_text(json.dumps({"ops": list(names),
                                      "chains": [base.pack_chain(c, codes)
                                                 for c in chains]}) + "\n")
    return write_info(NAME, kind="tile_chains", file=TABLE_PATH.name,
                      contract=contract,
                      engine=base.engine_fingerprint(),
                      registry=registry_fingerprint(chains),
                      stats={"chains": len(chains),
                             "no_action": chain_name(NO_ACTION),
                             "ops": list(names),
                             "op_bits": base.OP_BITS,
                             "columns": RESOURCE_NAMES,
                             "per_kind": {**{kind.value: len(chains_of_kind(kind))
                                             for kind in CHAINS_BY_KIND},
                                          "PLANT": len(CROP_CHAINS[True]) + len(CROP_CHAINS[False]),
                                          "PLANT_ongoing": len(CROP_CHAINS[True]),
                                          "PLANT_oneshot": len(CROP_CHAINS[False])}},
                      source="offline_lab.build.graph (the graph's action index)")


if __name__ == "__main__":
    raise SystemExit(main())
