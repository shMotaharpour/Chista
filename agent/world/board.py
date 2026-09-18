"""The board: cells, quadrants, and the shed's four doors.

Numbers are in `rules.py`; this is the vocabulary a layer uses to talk about a place on
the farm. A `Cell` is `(x, y)` with y growing downward, exactly as the engine indexes
`tiles[y][x]` (kaggriculture.py:145-149).
"""

from __future__ import annotations

from agent.world.rules import BOARD_SIZE, HALF, SHED_ACCESS

#: A place on the farm: `(x, y)`, y grows downward.
Cell = tuple[int, int]

#: The four 5x5 quadrants, NW first (kaggriculture.py:96, :127-129). NW is free.
QUADRANTS: tuple[str, ...] = ("NW", "NE", "SW", "SE")


def quadrant_of(cell: Cell, board_size: int = BOARD_SIZE) -> str:
    """Which quadrant a cell is in (kaggriculture.py:127-129)."""
    x, y = cell
    half = board_size // 2
    return ("N" if y < half else "S") + ("W" if x < half else "E")


def in_board(cell: Cell, board_size: int = BOARD_SIZE) -> bool:
    """Whether a cell is on the board at all (:326)."""
    x, y = cell
    return 0 <= x < board_size and 0 <= y < board_size


#: The four tiles from which the shed can be used, in the engine's NWSE order (:132-135).
#: PICKUP, DROP and PLACE-into-shed work from these and nowhere else.
SHED_DOORS: tuple[Cell, ...] = SHED_ACCESS

#: Where the main farmer stands at the start of every day: the first shed door in NW
#: order, `(4, 4)` at board 10 (:161-166, :879).
SPAWN: Cell = SHED_ACCESS[0]
