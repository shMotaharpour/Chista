"""TileGraph — travel-time atoms over the 10x10 grid.

This module ONLY provides cost atoms: pure, stateless, route-free.
- dist(a, b): shortest-move count (Manhattan; engine has no obstacles and
  locked tiles are passable — engine source: "Locked tiles are passable")
- dist_to_shed(xy): distance to the nearest shed-adjacent tile
- action_turns(action, n_items_of_one_type): turns consumed by one op

Route composition (ordering tasks, batching pickups) is L2's VRP job —
NOT here. No function in this file may sum over a task sequence.
"""
from __future__ import annotations

BOARD = 10
# shed-adjacent tiles: PICKUP/DROP only work here (engine docs, README §ops)
SHED_ADJACENT = {(4, 4), (5, 4), (4, 5), (5, 5)}
DEFAULT_SPAWN = (4, 4)   # farmer & hands respawn here every morning (engine L877)

# one op on a tile always costs one turn; PICKUP/DROP cost one turn per
# distinct item type (n of the same type is free — engine: PICKUP <item> [n])
ONE_OP = 1


def dist(a: tuple[int, int], b: tuple[int, int]) -> int:
    """Manhattan distance == shortest move count (4-directional movement,
    nothing blocks movement)."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def dist_to_shed(xy: tuple[int, int]) -> int:
    return min(dist(xy, s) for s in SHED_ADJACENT)


def at_shed(xy: tuple[int, int]) -> bool:
    return xy in SHED_ADJACENT


def action_turns(action: str, item_types: int = 1) -> int:
    """Turns for one op. PICKUP/DROP: 1 turn per distinct item type."""
    if action in ("PICKUP", "DROP"):
        return item_types
    return ONE_OP


def bfs_dist(start: tuple[int, int], goal: tuple[int, int]) -> int:
    """Reference BFS — must equal Manhattan everywhere (guard against the
    engine ever adding obstacles)."""
    if start == goal:
        return 0
    frontier, seen = [start], {start}
    d = 0
    while frontier:
        d += 1
        nxt = []
        for (x, y) in frontier:
            for nx, ny in ((x+1, y), (x-1, y), (x, y+1), (x, y-1)):
                if 0 <= nx < BOARD and 0 <= ny < BOARD and (nx, ny) not in seen:
                    if (nx, ny) == goal:
                        return d
                    seen.add((nx, ny))
                    nxt.append((nx, ny))
        frontier = nxt
    raise ValueError(f"unreachable: {goal}")
