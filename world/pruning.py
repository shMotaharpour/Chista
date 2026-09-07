"""Action pruning harness — rules extracted VERBATIM from the engine's
`_apply_unit_action` (kaggriculture.py L313-540) and `_process_market`.

Every `prune` below mirrors an explicit `if ...: return` guard in the engine:
if the engine's guard would reject, we don't offer the op. If the engine
would EXECUTE and change state, we keep the op. Nothing else.

Two categories only (docs/research/008 §L0):
  1. ILLEGAL — engine guard rejects (silent no-op)
  2. INERT   — engine executes but provably no state change is possible

Verified per-rule in tests/test_pruning.py: apply action on live engine,
diff state before/after == zero change for pruned ops.

Economic judgments are NOT pruned here — L1 constraints / L2 objectives.
"""
from __future__ import annotations

from world import mechanics as M
from world.tilegraph import at_shed


# =================================================================
# FARMER/HAND ops — mirrors of _apply_unit_action guards (engine L313+)
# =================================================================

def farmer_candidates(state, unit_idx: int = 0):
    """Ops legal+effective for a unit standing on its current tile.

    Guards mirrored (engine -> here):
      tile == "LOCKED" -> return                     (L397)
      PLANT: crop not in CROPS / tile not None / seeds<=0   (L401-407)
      WATER: tile not PLANT / watered_today          (L410-412)
      HARVEST: yield_units<=0 / (PLANT and age<first_yield_day)  (L446-456)
      FERTILIZE: tile not PLANT / no fertilizer in hand  (L472-475)
      DIG: tile None / placed animal                 (L483-486)
      BUILD_COOP|PASTURE: tile not None              (L492-500)
      FEED: tile not animal / fed_today / no wheat in hand  (L503-507)
      COLLECT_FERTILIZER: not animal / not available  (L510-513)
      CARE: not animal / cared_today                 (L517-520)
    """
    fx, fy = state.farmer_xy if unit_idx == 0 else tuple(state.hands[unit_idx - 1])
    tile = state.tile_at(fx, fy)
    inv = state.inventories()[unit_idx]
    out = []

    locked = tile == "LOCKED"

    if tile is None and not locked:
        # PLANT guard (L401-407)
        for crop, n in state.seeds.items():
            if n > 0:
                out.append(("PLANT", crop, fx, fy))
        # BUILD guards (L492-500): tile must be None — structure choice is
        # still L2's (which animal to house), but BOTH builds are legal ops
        # here since the engine only checks `tile is not None`.
        if any(state.shed.get(a, 0) > 0 or inv.get(a, 0) > 0 for a in M.ANIMALS):
            out.append(("BUILD_COOP", fx, fy))
            out.append(("BUILD_PASTURE", fx, fy))
        # DIG on empty tile: engine L483-486 — tile is None -> return (inert)
        # so not offered.

    if M.is_plant(tile):
        p = state.plant_at(fx, fy)
        # WATER guard (L410-412): not PLANT -> return; watered_today -> return
        if not p.watered_today:
            out.append(("WATER", fx, fy))
        # HARVEST guards (L446-457): yield_units<=0 -> return;
        # age < first_yield_day -> return (even with yield_units>0, which
        # one-time crops START with — _new_plant gives yield=1 at planting!)
        cd = M.CROPS[tile["crop"]]
        if tile.get("yield_units", 0) > 0 and p.age >= cd["first_yield_day"]:
            out.append(("HARVEST", fx, fy))
        # FERTILIZE guard (L472-475): needs fertilizer IN HAND
        if inv.get("FERTILIZER", 0) > 0:
            out.append(("FERTILIZE", fx, fy))
        # DIG on a plant: engine executes (removes plant) — keep
        out.append(("DIG", fx, fy))

    if M.is_animal_tile(tile):
        a = state.animal_at(fx, fy)
        # FEED guards (L503-507): fed_today -> return; no wheat in hand -> return
        if not a.fed_today and inv.get("WHEAT", 0) > 0:
            out.append(("FEED", fx, fy))
        # COLLECT_FERTILIZER guards (L510-513): not available -> return
        if a.fertilizer_available:
            out.append(("COLLECT_FERTILIZER", fx, fy))
        # CARE guard (L517-520): cared_today -> return
        if not a.cared_today:
            out.append(("CARE", fx, fy))
        # HARVEST on animal product: yield_units>0 -> executes
        if tile.get("yield_units", 0) > 0:
            out.append(("HARVEST", fx, fy))
        # DIG on structure WITH animal: engine L485 returns — inert, not offered.
        # DIG on EMPTY structure (no animal): engine executes — keep.
        if "animal" not in tile:
            out.append(("DIG", fx, fy))

    return out


# =================================================================
# Shed ops — guards at L343 (DROP) / L359 (PICKUP)
# =================================================================

def farmer_pickup_candidates(state, unit_idx: int = 0) -> list:
    """ENGINE RULE (L359-360, L364-368): PICKUP not at a shed-adjacent tile
    returns; item must exist in shed with n>0."""
    xy = state.farmer_xy if unit_idx == 0 else tuple(state.hands[unit_idx - 1])
    if not at_shed(xy):
        return []
    out = []
    for item, n in state.shed.items():
        if n > 0:
            out.append(("PICKUP", item, n))
    return out


def drop_candidates(state, unit_idx: int = 0) -> list:
    """ENGINE RULE (L343-357): DROP not at a shed-adjacent tile returns;
    with a non-empty hand it always executes (dumps everything; overflow
    discarded). Empty hand -> no state change -> inert, not offered."""
    xy = state.farmer_xy if unit_idx == 0 else tuple(state.hands[unit_idx - 1])
    inv = state.inventories()[unit_idx]
    if not at_shed(xy):
        return []
    if any(v > 0 for v in inv.values()):
        return [("DROP", xy)]
    return []


# =================================================================
# Movement — guards at L328-339
# =================================================================

def move_candidates(state, unit_idx: int = 0) -> list:
    """ENGINE RULE (L328-339): moves execute unless the target is outside
    the 10x10 board (locked tiles are passable). Border-aware pruning only."""
    x, y = state.farmer_xy if unit_idx == 0 else tuple(state.hands[unit_idx - 1])
    out = [("PASS",)]
    if y > 0:
        out.append(("NORTH",))
    if y < 9:
        out.append(("SOUTH",))
    if x > 0:
        out.append(("WEST",))
    if x < 9:
        out.append(("EAST",))
    return out


# =================================================================
# Market — mirrors of _parse_order/_commit_unit guards (engine L640+, L663+)
# =================================================================

def market_candidates(state) -> list:
    """Guards mirrored:
    - _parse_order (L640-656): n<=0 or malformed -> order dropped (ILLEGAL)
    - HIRE (L702-708): money < fib(hires_today) -> return
    - BUY_LAND (L712-721): n_unlocked_extra >= 3 -> return; money < price -> return
    - SELL (L664-672): shed[item] <= 0 -> commit fails (ILLEGAL)
    - BUY_PRODUCT (L674-684): money < price -> fails; shed full -> fails
    - BUY_SEED (L686-690): money < seed price -> fails
    - BUY_ANIMAL (L692-699): money < cost -> fails; shed full -> fails
    Prices at quote time depend on both players' orders — L1 decides amounts;
    here we only offer the op with n=1 as the vocabulary entry.
    """
    out = []
    if state.money >= M.hire_cost(state.hires_today):
        out.append(["HIRE"])
    if len(state.unlocked) < 4:
        land_idx = len(state.unlocked) - 1
        cost = M.LAND_PRICES[land_idx]
        if state.money >= cost:
            out.append(["BUY_LAND"])
    for item, n in state.shed.items():
        if n > 0:
            out.append(["SELL", item, n])
    for crop, cd in M.CROPS.items():
        if state.money >= cd["seed"]:
            out.append(["BUY_SEED", crop, 1])
    for animal, ad in M.ANIMALS.items():
        if state.money >= ad["cost"] and sum(state.shed.values()) < M.SHED_CAP:
            out.append(["BUY_ANIMAL", animal, 1])
    if state.money >= 1 and sum(state.shed.values()) < M.SHED_CAP:
        # BUY_PRODUCT exists for WHEAT and FERTILIZER only (engine L597-598)
        for item in ("WHEAT", "FERTILIZER"):
            if state.money >= M.market_price(item, state.market_inventory[item]):
                out.append(["BUY_PRODUCT", item, 1])
    return out


# =================================================================
# Top-level
# =================================================================

def prune_farmer(state, unit_idx: int = 0) -> list:
    """Aggregate of the ENGINE RULE mirrors above (L313-540 _apply_unit_action)
    for one unit: tile ops + shed ops."""
    return (farmer_candidates(state, unit_idx)
            + farmer_pickup_candidates(state, unit_idx)
            + drop_candidates(state, unit_idx))


def prune_all(state) -> dict:
    """Complete pruned action vocabulary for this turn, per unit. Aggregate
    of the ENGINE RULE mirrors above; economic choice is L1/L2's."""
    return {
        "farmer": prune_farmer(state, 0),
        "hands": [prune_farmer(state, i + 1) for i in range(len(state.hands))],
        "farmer_moves": move_candidates(state, 0),
        "market": market_candidates(state),
    }
