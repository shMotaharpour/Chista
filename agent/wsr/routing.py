"""The walk: the one place a path between two tiles is spelled.

The search prices a walk by its length; the compiler writes the moves. Both go through here, so the
distance a day was priced at and the moves it is written with can never be two different paths.
"""

from __future__ import annotations

from agent.world.board import MOVE_DELTA
from agent.world.rules import BOARD_SIZE

Cell = tuple[int, int]


def walk(start: Cell, goal: Cell, board: int = BOARD_SIZE) -> list[tuple[str, ...]]:
    """The move ops that carry a unit from `start` to `goal`, x first then y.

    Shortest by construction and always legal: the engine only refuses a move that leaves the
    board. Ties break x-then-y, so a plan is a function of its inputs rather than of a hash order.
    """
    x, y = int(start[0]), int(start[1])
    gx, gy = int(goal[0]), int(goal[1])
    out: list[tuple[str, ...]] = []
    while x != gx:
        step = "EAST" if gx > x else "WEST"
        x += MOVE_DELTA[step][0]
        out.append((step,))
    while y != gy:
        step = "SOUTH" if gy > y else "NORTH"
        y += MOVE_DELTA[step][1]
        out.append((step,))
    if not (0 <= x < board and 0 <= y < board):
        raise ValueError(f"walk({start}, {goal}) leaves the board")
    return out
