"""The day as arrays: the vectorised twin of `Instance`.

`Instance` is a Python structure - dicts, lists, `MinorTask` - which suits a loop that walks one
task at a time. A beam search asks the same questions about every task at once, so it wants the
same facts in arrays: a precedence matrix instead of a list of pairs, an item column instead of a
dict of needs, and one distance matrix for the whole board instead of a call per hop.

The board is fixed at ten by ten, so the distance matrix is built once per process and indexed
after that. It is symmetric with a zero diagonal, and no cell is further than eighteen apart, so
it fits in a byte per entry.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from agent.world.model import UnitAction
from agent.world.rules import BOARD_SIZE, SHED_ACCESS

NO_ITEM = -1


def distance_matrix(size: int = BOARD_SIZE) -> np.ndarray:
    """Manhattan distance between every pair of cells, built once.

    Row and column index a cell as `y * size + x`, which is the order the game's tile rows use.
    """
    coords = np.arange(size)
    ys, xs = np.meshgrid(coords, coords, indexing="ij")
    flat = np.stack([ys.ravel(), xs.ravel()], axis=1).astype(np.int16)
    delta = np.abs(flat[:, None, :] - flat[None, :, :])
    return delta.sum(axis=-1).astype(np.int8)


DISTANCE: np.ndarray = distance_matrix()
CELL_INDEX: dict[tuple[int, int], int] = {
    (int(y), int(x)): y * BOARD_SIZE + x
    for y in range(BOARD_SIZE)
    for x in range(BOARD_SIZE)
}


def index_of(cell: tuple[int, int]) -> int:
    return CELL_INDEX[(int(cell[0]), int(cell[1]))]


@dataclass
class TaskArray:
    """Every task of the day, one per row, with its constraints as columns.

    `pred[i, j]` is True when task `j` must finish before task `i` starts. The sum down a column is
    therefore the number of predecessors a task waits for, and `done @ pred` the number it already
    has - so "can this start" is one matrix product, not a loop.
    """

    ids: list[str] = field(default_factory=list)
    actions: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int8))
    items: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int8))
    cells: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), dtype=np.int16))
    columns: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int8))
    pred: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=bool))
    earliest: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int8))
    latest: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int8))

    # Two columns bound every task in time, and both are given to the search rather than looked up
    # by it: the timetable the planner hands over is the only source.
    #
    #   earliest  a fetch cannot happen before its good is in the shed, and a planting cannot
    #             happen before its seed is bought - same rule, different actor.
    #   latest    the last hour the task may run, the horizon by default.
    #
    # Not yet carried: a ceiling on a tile that harvests, saying the crop must reach the shed by a
    # given hour, and the harvest aggregation that lets a worker take wheat or fertilizer off the
    # tile instead of the shed. The columns are here so those land without reshaping the array.

    @property
    def n(self) -> int:
        return len(self.ids)

    @property
    def cell_index(self) -> np.ndarray:
        """The distance-matrix row of each task's tile."""
        return (self.cells[:, 0].astype(np.int32) * BOARD_SIZE
                + self.cells[:, 1].astype(np.int32))

    @property
    def needs(self) -> np.ndarray:
        """What each task must fetch first: an item id, or NO_ITEM."""
        return self.items

    @property
    def pred_count(self) -> np.ndarray:
        return self.pred.sum(axis=0)

    def window(self, hour: np.ndarray) -> np.ndarray:
        """Which tasks may run at `hour`, per state - both columns at once.

        `hour` is (batch, n): the hour each state would reach each task. A task outside its window
        is not a candidate, which is how an action the clock forbids gets dropped rather than
        scheduled and discarded later.
        """
        return (hour >= self.earliest) & (hour <= self.latest)

    def ready(self, done: np.ndarray) -> np.ndarray:
        """Which tasks have all their predecessors done - one matrix product.

        `done` is (batch, n) booleans. The result is (batch, n): True where a task may start.
        """
        return (done.astype(np.int8) @ self.pred.astype(np.int8)) == self.pred_count

    def done_by_column(self, done: np.ndarray) -> np.ndarray:
        """How many of each column's tasks are done, per state - for the tile-block preference."""
        n_col = int(self.columns.max()) + 1 if self.n else 0
        out = np.zeros((done.shape[0], n_col), dtype=np.int16)
        for c in range(n_col):
            mask = self.columns == c
            if mask.any():
                out[:, c] = done[:, mask].sum(axis=1)
        return out


def columns_of(cells: np.ndarray) -> np.ndarray:
    """Number the tiles, so tasks can be grouped by the tile they happen on."""
    seen: dict[tuple[int, int], int] = {}
    out = np.zeros(len(cells), dtype=np.int8)
    for i, (y, x) in enumerate(cells):
        key = (int(y), int(x))
        if key not in seen:
            seen[key] = len(seen)
        out[i] = seen[key]
    return out


def action_codes() -> dict[UnitAction, int]:
    return {a: i for i, a in enumerate(UnitAction)}


SHED_INDEX = np.asarray([index_of(d) for d in SHED_ACCESS], dtype=np.int32)
