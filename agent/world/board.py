"""The board: cells, quadrants, and the shed's four doors.

Numbers are in `rules.py`; this is the vocabulary a layer uses to talk about a place on
the farm. A `Cell` is `(x, y)` with y growing downward, exactly as the engine indexes
`tiles[y][x]` (kaggriculture.py:145-149).
"""

from __future__ import annotations

from agent.world.rules import BOARD_SIZE, HALF, SHED_ACCESS

#: A place on the farm: `(x, y)`, y grows downward.
Cell = tuple[int, int]

#: A move's delta is `(dx, dy)`, y growing downward (kaggriculture.py:87-93): NORTH is
#: `(0, -1)` — x does not change, y decreases. That is the MOVE's delta and not the
#: action: the action for north is the one-element list `["NORTH"]`, and a cell is
#: `(x, y)` while the board array is indexed `[y][x]`. Both are true at once, and
#: `tile_at` is the only place the array order is written.
MOVE_DELTA: dict[str, Cell] = {
    "NORTH": (0, -1), "SOUTH": (0, 1), "EAST": (1, 0), "WEST": (-1, 0)}

#: The four 5x5 quadrants, NW first (kaggriculture.py:96, :127-129). NW is free.
QUADRANTS: tuple[str, ...] = ("NW", "NE", "SW", "SE")


def tile_at(tiles: list, cell: Cell) -> object:
    """The engine stores the board as `tiles[y][x]`: the OUTER list is y, the inner one
    x (kaggriculture.py:145-149, and the observation format in the environment's README
    says the same). Every read goes through here so the order is written down once."""
    x, y = cell
    return tiles[y][x]


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


def manhattan(a: Cell, b: Cell) -> int:
    """Turns between two cells: the board has no obstacles and a unit moves one tile a turn.

    Coordinates are cast because the engine's observations carry them as strings.
    """
    return abs(int(a[0]) - int(b[0])) + abs(int(a[1]) - int(b[1]))
