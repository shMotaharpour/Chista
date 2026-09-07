"""TileGraph — travel-time atoms over the 10x10 grid.

This module ONLY provides cost atoms: pure, stateless, route-free.
- dist(a, b): shortest-move count (Manhattan; engine has no obstacles and
  locked tiles are passable — engine source: "Locked tiles are passable")
- dist_to_shed(xy): distance to the nearest shed-adjacent tile
- spawn_position(occupancy): where the next unit spawns (engine L161/L533)
- action_turns(action, item_types): turns consumed by one op

Route composition (ordering tasks, batching pickups) is L2's VRP job —
NOT here. No function in this file may sum over a task sequence.
"""
from __future__ import annotations

BOARD = 10
# shed-adjacent tiles: PICKUP/DROP only work here (engine docs, README §ops),
# in NWSE order (engine L132-135 _shed_access_tiles)
SHED_ADJACENT = [(4, 4), (5, 4), (4, 5), (5, 5)]
DEFAULT_SPAWN = (4, 4)   # farmer spawns at first NWSE-free shed tile => (4,4)
                         # every morning (engine L161-166, L879). Hands use the
                         # min-occupancy rule — see spawn_position().

# one op on a tile always costs one turn.
ONE_OP = 1


def dist(a: tuple[int, int], b: tuple[int, int]) -> int:
    """ENGINE RULE (movement, L327-339): 4-directional moves, nothing blocks
    movement (locked tiles are passable — L334-337) => shortest move count
    equals Manhattan distance."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def dist_to_shed(xy: tuple[int, int]) -> int:
    """ENGINE RULE (L132-135): shed access = min Manhattan distance to the
    four NWSE-ordered shed-adjacent tiles."""
    return min(dist(xy, s) for s in SHED_ADJACENT)


def at_shed(xy: tuple[int, int]) -> bool:
    """ENGINE RULE (L138 _is_shed_adjacent): xy is one of the 4 shed tiles."""
    return xy in SHED_ADJACENT


def action_turns(action: str, item_types: int = 1) -> int:
    """Turns for one op.
    - PICKUP: 1 turn per DISTINCT item type; count is free within one type
      (["PICKUP", item, n] — engine L359: one op, one type, any n).
    - DROP: always 1 turn — dumps the ENTIRE hand inventory in one op
      (engine L343-357: iterates all items, no arguments).
    - everything else: 1 turn."""
    if action == "PICKUP":
        return item_types
    return ONE_OP


def spawn_position(occupancy: dict[tuple[int, int], int]) -> tuple[int, int]:
    """ENGINE RULE (L533-541 _spawn_hand): a NEW unit spawns on the
    shed-adjacent tile with MINIMUM current occupancy; ties broken by NWSE
    order of SHED_ADJACENT. Farmer spawns first at (4,4) (L161-166)."""
    best = min(SHED_ADJACENT, key=lambda t: (occupancy.get(t, 0),
                                             SHED_ADJACENT.index(t)))
    return best


def units_spawn_positions(n_hands: int) -> list[tuple[int, int]]:
    """ENGINE RULE derived (L879 farmer first, then L533 per hand): where the
    farmer + n_hands stand at morning spawn."""
    occ: dict[tuple[int, int], int] = {t: 0 for t in SHED_ADJACENT}
    positions: list[tuple[int, int]] = [DEFAULT_SPAWN]
    occ[DEFAULT_SPAWN] += 1
    for _ in range(n_hands):
        pos = spawn_position(occ)
        positions.append(pos)
        occ[pos] += 1
    return positions
