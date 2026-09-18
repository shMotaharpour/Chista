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

Run from the repo root:

    .venv/bin/python -m offline_lab.build.chains
"""

from __future__ import annotations

import json
from pathlib import Path

from agent.artifact import artifact_path, write_info
from agent.tile_dp import chains as base
from agent.tile_dp.chains import (ANIMAL_RES, BUILD_OF_STRUCTURE, NO_ACTION, PRODUCT_RES,
                                  SEED_RES, TILE_OPS, chain_name, entity_code_of,
                                  engine_fingerprint)
from agent.world.model import ANIMALS, CROPS, Structure, TileKind, UnitAction
from agent.world.rules import ANIMAL_RULES, CROP_RULES, TURNS_PER_DAY
from agent.world.tile import crop_age_origin

NAME = "tile_chains"
PATH = artifact_path(NAME, ".data.json")


def registry_fingerprint(chains: tuple[tuple[str, ...], ...]) -> str:
    """Fingerprint of the chains THIS build produced (the agent's own one hashes the
    registry it loaded, which does not exist yet while this runs)."""
    from hashlib import sha256
    return sha256("\n".join(chain_name(c) for c in chains).encode()).hexdigest()[:16]


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
                     UnitAction.HARVEST.value),
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


def _kind_after(kind: TileKind, op: str) -> TileKind:
    """The tile's kind once `op` has run on `kind`."""
    if op == UnitAction.PLANT.value:
        return TileKind.PLANT
    if op == UnitAction.BUILD_COOP.value:
        return TileKind.EMPTY_COOP
    if op == UnitAction.BUILD_PASTURE.value:
        return TileKind.EMPTY_PASTURE
    if op == UnitAction.PLACE.value:
        return TileKind.ANIMAL
    if op == UnitAction.DIG.value:
        return TileKind.NONE
    return kind


def _day_chains(kind: TileKind, ops: tuple[str, ...] = (), rank: int = -1,
                steps: int = 0) -> list[tuple[str, ...]]:
    """Every day this kind can have, as the sequences the rules allow.

    Recursive over the day: at each step the tile is in some kind, one of that kind's
    actions may run (in canonical order, and DIG only first or right after HARVEST), and
    the kind moves on. A chain is any prefix of such a walk, which is what makes an
    incomplete day - the worker stops early - a chain of its own.
    """
    out: list[tuple[str, ...]] = [ops]
    if steps >= TURNS_PER_DAY:
        return out
    for op in ACTIONS_BY_KIND.get(kind, ()):
        if RANK[op] < rank:
            continue
        if op in ops:
            continue          # an op is done once in a day (water once, feed once, ...)
        if op == UnitAction.DIG.value:
            # first op of the chain, or right after a HARVEST
            if ops and ops[-1] != UnitAction.HARVEST.value:
                continue
        nxt = _kind_after(kind, op)
        out += _day_chains(nxt, ops + (op,), RANK[op], steps + _steps(op))
    return out


#: Engine steps one op fills (a chain may not need more than a day has turns).
_STEPS_PER_OP: dict[str, int] = {UnitAction.PLANT.value: 2, UnitAction.FERTILIZE.value: 3,
                                 UnitAction.FEED.value: 3, UnitAction.PLACE.value: 3}


def _steps(op: str) -> int:
    return _STEPS_PER_OP.get(op, 1)


def chains_of_kind(kind: TileKind) -> tuple[tuple[str, ...], ...]:
    """The chains a tile of this kind can have, `NO_ACTION` first (dedup, order kept)."""
    out = [NO_ACTION]
    for chain in _day_chains(kind):
        if chain and chain not in out:
            out.append(chain)
    return tuple(out)


def registry() -> tuple[tuple[str, ...], ...]:
    """The chains, in registry order (ids are positions in this list)."""
    out: list[tuple[str, ...]] = []
    for kind in (TileKind.NONE, TileKind.WEED, TileKind.EMPTY_COOP,
                 TileKind.EMPTY_PASTURE, TileKind.PLANT, TileKind.ANIMAL):
        for chain in chains_of_kind(kind):
            if chain not in out:
                out.append(chain)
    return tuple(out)


CHAINS_BY_KIND: dict[TileKind, tuple[tuple[str, ...], ...]] = {
    kind: chains_of_kind(kind) for kind in
    (TileKind.NONE, TileKind.WEED, TileKind.EMPTY_COOP, TileKind.EMPTY_PASTURE,
     TileKind.PLANT, TileKind.ANIMAL)}

#: A young plant cannot be harvested yet: the same list, without HARVEST.
CROP_CHAINS_YOUNG: tuple[tuple[str, ...], ...] = tuple(
    c for c in CHAINS_BY_KIND[TileKind.PLANT] if UnitAction.HARVEST.value not in c)


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


def domain_ok(ops: tuple[str, ...], entity: str) -> bool:
    """False when `entity` must not run `ops`.

    A crop never runs an animal op and an animal never plants. A structure must be the one
    the species lives in - with the engine's own exception: BUILD_COOP needs a bare tile,
    so changing a barn into a coop is only possible by DIGGING it first (:493-503).
    """
    ops_set = set(ops)
    if entity in ANIMALS:
        if ops_set & CROP_ONLY_OPS:
            return False
        structure = ANIMAL_RULES[entity]["structure"]
        other = set(BUILD_OF_STRUCTURE.values()) - {BUILD_OF_STRUCTURE[structure]}
        if not ops_set & other:
            return True
        return (UnitAction.DIG.value in ops
                and ops.index(UnitAction.DIG.value)
                < min(ops.index(op) for op in ops_set & other))
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
        base_chains = CROP_CHAINS_YOUNG if (age is not None and age < 0) \
            else CHAINS_BY_KIND[kind]
    elif kind in CHAINS_BY_KIND:
        base_chains = CHAINS_BY_KIND[kind]
    else:
        return []
    return [c for c in base_chains if applicable(c, age, yield_units, entity)]


# --- write the artifact ------------------------------------------------------------ #

def main() -> int:
    chains = registry()
    overlong = [c for c in chains if base.chain_steps(c) > TURNS_PER_DAY]
    if overlong:
        raise AssertionError(f"chains needing more than {TURNS_PER_DAY} steps: {overlong}")
    PATH.parent.mkdir(parents=True, exist_ok=True)
    PATH.write_text(json.dumps({"chains": [list(c) for c in chains]}, indent=1) + "\n")
    info = write_info(NAME, kind="tile_chains", file=PATH.name,
                      contract=contract_of(chains),
                      engine=base.engine_fingerprint(),
                      registry=registry_fingerprint(chains),
                      stats={"chains": len(chains),
                             "no_action": chain_name(NO_ACTION),
                             "ops": sorted(TILE_OPS),
                             "columns": base.RESOURCE_NAMES,
                             "per_kind": {kind.value: len(chains_of_kind(kind))
                                          for kind in CHAINS_BY_KIND},
                             "steps_per_chain": {chain_name(c): base.chain_steps(c)
                                                 for c in chains}},
                      source="offline_lab.build.chains:registry")
    print(f"chains {len(chains)} -> {PATH.name}, info -> {info.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
