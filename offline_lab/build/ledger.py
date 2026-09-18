"""What a chain costs and yields: the ledger the BUILDER prices edges with.

Cost is labour hours plus the inputs a chain spends (a seed, an animal, fertiliser, feed);
produce is the harvest plus the fertiliser a collection picked up. The two are never netted,
and both are written into the graph as the per-edge `cost` and `produce` columns - which is
why the runtime needs none of this: it reads the matrices.
"""

from __future__ import annotations

from agent.world.model import ANIMALS, CROPS, N_RESOURCE, RESOURCE_ID, UnitAction
from agent.world.model import RES_LABOR, RES_FERTILIZER, RES_WHEAT
from agent.world.rules import ANIMAL_RULES

from agent.tile_dp.chains import TILE_OPS

#: Which column a PLANT / PLACE spends, and which one a harvest fills.
SEED_RES: dict[str, str] = {crop: f"SEED_{crop}" for crop in CROPS}
ANIMAL_RES: dict[str, str] = {animal: f"ANIMAL_{animal}" for animal in ANIMALS}
PRODUCT_RES: dict[str, str] = {crop: crop for crop in CROPS}
PRODUCT_RES.update({a: ANIMAL_RULES[a]["product"] for a in ANIMALS})


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
