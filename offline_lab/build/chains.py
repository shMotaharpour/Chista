"""Build the tile chains and write them as an artifact. Offline only.

The rules that decide what a chain may be - the op pools, the subsets, the DIG layering,
the applicability windows - are the process. The RESULT is `agent/artifact/tile_chains`,
which is all the agent loads: the agent defines what a chain IS (`agent.tile_dp.chains`),
and never how one was chosen.

Run from the repo root:

    .venv/bin/python -m offline_lab.build.chains
"""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

from agent.artifact import artifact_path, write_info
from agent.tile_dp import chains as base
from agent.tile_dp.chains import (ALL_OPS, ANIMAL_RES, BUILD_OF_STRUCTURE, MARKET_OPS,
                                  NO_ACTION, PRODUCT_RES, SEED_RES, TILE_OPS,
                                  chain_name, entity_code_of, registry_fingerprint)
from agent.world.model import (ANIMALS, CROPS, Column, Structure, TileKind, UnitAction)
from agent.world.rules import ANIMAL_RULES, CROP_RULES, TURNS_PER_DAY
from agent.world.tile import crop_age_origin

NAME = "tile_chains"
PATH = artifact_path(NAME, ".json")

# --- the pools a chain is drawn from -------------------------------------------- #

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

ANIMAL_ONLY_OPS: frozenset[str] = frozenset((
    UnitAction.BUILD_COOP.value, UnitAction.BUILD_PASTURE.value, UnitAction.PLACE.value,
    UnitAction.FEED.value, UnitAction.CARE.value, UnitAction.COLLECT_FERTILIZER.value))
CROP_ONLY_OPS: frozenset[str] = frozenset((UnitAction.PLANT.value,))


def _subsets(ops: tuple[str, ...]) -> list[tuple[str, ...]]:
    """Every subset of `ops`; the empty one is the no-action case."""
    return [tuple(op for op in ops if op in combo) or NO_ACTION
            for r in range(len(ops) + 1)
            for combo in combinations(ops, r)]


_CROP_SUBSETS = _subsets(CROP_OPS)
# CARE is a no-op without FEED on the same day: the engine spends `cared_today` only
# together with `fed_today` (kaggriculture.py:826-829).
_ANIMAL_SUBSETS = [c for c in _subsets(ANIMAL_OPS)
                   if not ("CARE" in c and "FEED" not in c)]

#: ONE bare-tile list, with no crop/animal split: the split belongs to the filters, not
#: to the chains.
NONE_CHAINS: tuple[tuple[str, ...], ...] = (
    NO_ACTION, (UnitAction.PLANT.value, UnitAction.WATER.value),
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
    """Variants that put DIG after `head` (an empty head = DIG is the first op)."""
    if kind not in DIGGABLE_KINDS:
        return []
    if _kind_after(kind, head) not in DIGGABLE_KINDS:
        return []
    out = [head + (UnitAction.DIG.value,)]
    out += [head + (UnitAction.DIG.value,) + follow for follow in NONE_CHAINS
            if follow != NO_ACTION]
    return out


def _layer(kind: TileKind, base_chains: tuple[tuple[str, ...], ...]
           ) -> tuple[tuple[str, ...], ...]:
    """Base chains plus every legal DIG layering of them (dedup, order kept)."""
    out: list[tuple[str, ...]] = list(base_chains)
    out += _dig_tail(kind, ())
    for chain in base_chains:
        if chain and chain[-1] == UnitAction.HARVEST.value:
            out += _dig_tail(kind, chain)
    return tuple(dict.fromkeys(out))


CHAINS_BY_KIND: dict[TileKind, tuple[tuple[str, ...], ...]] = {
    TileKind.NONE: NONE_CHAINS,
    TileKind.WEED: _layer(TileKind.WEED, (NO_ACTION, (UnitAction.DIG.value,))),
    TileKind.PLANT: _layer(TileKind.PLANT, tuple(_CROP_SUBSETS)),
    TileKind.ANIMAL: _layer(TileKind.ANIMAL, tuple(_ANIMAL_SUBSETS)),
    TileKind.EMPTY_COOP: _layer(TileKind.EMPTY_COOP,
                                (NO_ACTION, (UnitAction.PLACE.value,))),
    TileKind.EMPTY_PASTURE: _layer(TileKind.EMPTY_PASTURE,
                                   (NO_ACTION, (UnitAction.PLACE.value,))),
}
CROP_CHAINS_YOUNG: tuple[tuple[str, ...], ...] = _layer(
    TileKind.PLANT, tuple(c for c in _CROP_SUBSETS
                          if UnitAction.HARVEST.value not in c))


def registry() -> tuple[tuple[str, ...], ...]:
    """The chains, in registry order (ids are positions in this list)."""
    out: list[tuple[str, ...]] = []
    for kind in (TileKind.NONE, TileKind.WEED, TileKind.EMPTY_COOP,
                 TileKind.EMPTY_PASTURE, TileKind.PLANT, TileKind.ANIMAL):
        for chain in CHAINS_BY_KIND[kind]:
            if chain not in out:
                out.append(chain)
    return tuple(out)


# --- the filters ------------------------------------------------------------------ #

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
    """False when `entity` must not run `ops`: a crop never runs an animal op, an animal
    never plants, and a structure must be the one the species lives in."""
    ops_set = set(ops)
    if entity in ANIMALS:
        if ops_set & CROP_ONLY_OPS:
            return False
        structure = ANIMAL_RULES[entity]["structure"]
        return not (ops_set & (set(BUILD_OF_STRUCTURE.values())
                               - {BUILD_OF_STRUCTURE[structure]}))
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
                      contract=base.contract_id(),
                      engine=base.engine_fingerprint(),
                      registry=registry_fingerprint(),
                      stats={"chains": len(chains),
                             "no_action": chain_name(NO_ACTION),
                             "ops": sorted(ALL_OPS),
                             "market_ops": sorted(MARKET_OPS),
                             "columns": base.RESOURCE_NAMES,
                             "steps_per_chain": {chain_name(c): base.chain_steps(c)
                                                 for c in chains}},
                      source="offline_lab.build.chains:registry")
    print(f"chains {len(chains)} -> {PATH.name}, info -> {info.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
