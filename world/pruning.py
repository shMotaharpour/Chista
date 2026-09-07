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


# ---------------------------------------------------------------- helpers

def _on_unlocked_tile(state, x: int, y: int) -> bool:
    lim = state.unlocked_limit()
    return 0 <= x < lim and 0 <= y < lim


# ---------------------------------------------------------------- farmer ops

def farmer_candidate_tiles(state):
    """(x, y) tiles where the farmer stands that admit at least one op."""
    fx, fy = state.farmer_xy
    tile = state.tiles[fy][fx]
    out = []
    if M.is_plant(tile):
        p = next(p for p in state.plants() if (p.x, p.y) == (fx, fy))
        if not p.watered_today and p.consecutive_unwatered < 2:
            out.append(("WATER", fx, fy))                       # inert if watered
        tile_raw = state.tiles[fy][fx]
        if M.plant_mature(tile_raw, state.day):
            out.append(("HARVEST", fx, fy))                     # inert if yield 0 & ongoing
        if state.day <= tile_raw.get("fertilized_until_day", -1) is False and False:
            pass  # FERTILIZE handled by L2 priority (needs fertilizer in hand)
    if M.is_animal_tile(tile):
        a = next(a for a in state.animal_tiles() if (a.x, a.y) == (fx, fy))
        if a.fed_today is False:
            out.append(("FEED", fx, fy))
        if a.cared_today is False:
            out.append(("CARE", fx, fy))
        if a.yield_units > 0:
            out.append(("HARVEST", fx, fy))
        if a.fertilizer_available:
            out.append(("COLLECT_FERTILIZER", fx, fy))
    if tile is None:
        for crop in M.CROPS:
            if state.seeds.get(crop, 0) > 0:
                out.append(("PLANT", crop, fx, fy))
        if state.player_has_structure_materials() if hasattr(state, "player_has_structure_materials") else False:
            pass  # BUILD_* legality depends on engine cost rules — left to L2/L3
    return out


def farmer_pickup_candidates(state) -> list:
    """PICKUP candidates: only at shed tiles, only items the shed has,
    only if the unit's hand can use them (wheat -> feed needed, animal ->
    empty structure exists)."""
    if not at_shed(state.farmer_xy):
        return []                                                # illegal elsewhere
    out = []
    shed = state.shed
    wheat = shed.get("WHEAT", 0)
    if wheat > 0 and state.animals_needing_feed():
        out.append(("PICKUP", "WHEAT", wheat))
    for animal in M.ANIMALS:
        if shed.get(animal, 0) > 0 and any(
                t.get("kind") == M.ANIMAL_STRUCTURE[animal] and not t.get("animal")
                for row in state.tiles for t in row if isinstance(t, dict)):
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
    - BUY_SEED / BUY_ANIMAL / BUY_PRODUCT: legal for any n >= 1 the money
      can cover (economic sense is L1's job)
    - BUY_LAND: pruned when all quadrants already unlocked
    """
    from world import mechanics as M
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
    return farmer_candidate_tiles(state) + farmer_pickup_candidates(state)


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
