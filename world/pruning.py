"""Action pruning harness — remove engine-rejected and provably-inert actions.

Two categories only (docs/research/008 §L0):
  1. ILLEGAL — the engine silently rejects (no-op): must trace to engine source
  2. INERT   — engine executes but provably cannot change state

Each rule is verified by an in-engine experiment: apply the action, diff
state before/after == zero change (tests/test_pruning.py).

Economic judgments (sell at floor, buy seeds without tiles, ...) are NOT
pruned here — they are L1 constraints / L2 objectives.
"""
from __future__ import annotations

from world import mechanics as M
from world.tilegraph import at_shed


# ---------------------------------------------------------------- farmer ops

def farmer_candidates(state):
    """Ops the farmer can do ON ITS CURRENT TILE right now (non-illegal,
    non-inert). Movement and destination choice are L2's job."""
    fx, fy = state.farmer_xy
    tile = state.tile_at(fx, fy)
    out = []

    if M.is_plant(tile):
        p = state.plant_at(fx, fy)
        if not p.watered_today and p.consecutive_unwatered < 2:
            out.append(("WATER", fx, fy))               # inert if watered
        if M.plant_mature(tile, state.day):
            out.append(("HARVEST", fx, fy))             # inert if yield 0
        # FERTILIZE: legal with fertilizer in hand; value judgment is L2's

    if M.is_animal_tile(tile):
        a = state.animal_at(fx, fy)
        if not a.fed_today:
            out.append(("FEED", fx, fy))
        if not a.cared_today:
            out.append(("CARE", fx, fy))
        if a.yield_units > 0:
            out.append(("HARVEST", fx, fy))
        if a.fertilizer_available:
            out.append(("COLLECT_FERTILIZER", fx, fy))

    if tile is None and state.is_unlocked_tile(fx, fy):
        for crop, n in state.seeds.items():
            if n > 0:
                out.append(("PLANT", crop, fx, fy))
        # BUILD_*: legal on empty unlocked tiles when the matching animal
        # sits in shed/hand; the "which animal/structure" choice is L2's.
        for animal, structure in M.ANIMAL_STRUCTURE.items():
            in_hand = state.inventories()[0].get(animal, 0)
            in_shed = state.shed.get(animal, 0)
            if in_hand or in_shed:
                out.append(("BUILD_" + structure, fx, fy))
    return out


def farmer_pickup_candidates(state) -> list:
    """PICKUP: only at shed tiles (illegal elsewhere), only items the shed
    has."""
    if not at_shed(state.farmer_xy):
        return []
    out = []
    shed = state.shed
    if shed.get("WHEAT", 0) > 0:
        out.append(("PICKUP", "WHEAT", shed["WHEAT"]))
    for animal in M.ANIMALS:
        if shed.get(animal, 0) > 0:
            out.append(("PICKUP", animal, shed[animal]))
    return out


def farmer_move_candidates(state) -> list:
    """Movement is always legal on the 10x10 grid (locked tiles passable).
    Pruning does NOT judge destinations — that's L2's routing job."""
    return [("NORTH",), ("SOUTH",), ("EAST",), ("WEST",), ("PASS",)]


def market_candidates(state) -> list:
    """Market orders — prune only engine-level no-ops, not economics.
    - HIRE: pruned when money < next fib cost (engine rejects unpaid hire)
    - SELL: pruned for zero-quantity items
    - BUY_*: legal for any n >= 1 (economic sense is L1's job)
    - BUY_LAND: pruned when all quadrants already unlocked
    """
    out = []
    if state.money >= M.hire_cost(state.hires_today):
        out.append(["HIRE"])
    for item, n in state.shed.items():
        if n > 0:
            out.append(["SELL", item, n])
    for crop in M.CROPS:
        out.append(["BUY_SEED", crop, 1])
    for animal in M.ANIMALS:
        out.append(["BUY_ANIMAL", animal, 1])
    if len(state.unlocked) < 4:
        out.append(["BUY_LAND"])
    return out


# ---------------------------------------------------------------- top-level

def prune_farmer(state) -> list:
    """All non-illegal, non-inert farmer ops for the CURRENT tile."""
    return farmer_candidates(state) + farmer_pickup_candidates(state)


def prune_market(state) -> list:
    return market_candidates(state)


def prune_all(state) -> dict:
    """The complete pruned action vocabulary for this turn."""
    return {
        "farmer": prune_farmer(state),
        "moves": farmer_move_candidates(state),
        "market": prune_market(state),
        "hands": len(state.hands),
    }
