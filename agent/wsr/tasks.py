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


def _item_table() -> dict:
    from agent.world.action import Animal, Crop, Product
    return {item: i for i, item in enumerate(list(Crop) + list(Animal) + list(Product))}


ITEM_CODE: dict = _item_table()


def build(chains, *, available: dict[str, int] | None = None, horizon: int = 24) -> TaskArray:
    """The planner's chains -> the arrays a beam search reads.

    The tasks themselves come from `expand_chain`, the same builder the greedy solver uses, so the
    two solvers see the same work and the same precedence - the only difference here is the shape.
    The time columns are filled from the timetable that is handed in: a fetch waits for its good,
    a planting waits for its seed, and everything else may run from the first hour. Nothing is read
    from the world.
    """
    from agent.wsr.models import expand_chain

    available = available or {}
    tasks = []
    order: list[tuple[str, str]] = []
    column_of: dict[str, int] = {}
    for index, (cell, ops, entity) in enumerate(chains):
        expansion = expand_chain(ops, entity=entity, cell=cell, prefix=f"d{index}_")
        tasks.extend(expansion.tasks)
        order.extend(expansion.order)
        for task in expansion.tasks:
            column_of[task.id] = index

    ids = [t.id for t in tasks]
    row_of = {tid: i for i, tid in enumerate(ids)}
    n = len(tasks)

    pred = np.zeros((n, n), dtype=bool)
    for before, after in order:
        if before in row_of and after in row_of:
            pred[row_of[after], row_of[before]] = True

    earliest = np.zeros(n, dtype=np.int8)
    for i, task in enumerate(tasks):
        good = _good_for(task)
        if good is not None:
            earliest[i] = int(available.get(str(getattr(good, "value", good)), 0))

    cells = np.asarray([t.cell if t.cell else (0, 0) for t in tasks], dtype=np.int16)
    return TaskArray(
        ids=ids,
        actions=np.asarray([_action_code(t.action) for t in tasks], dtype=np.int8),
        items=np.asarray([_item_code(t.item) for t in tasks], dtype=np.int8),
        cells=cells,
        columns=columns_of(cells),
        pred=pred,
        earliest=earliest,
        latest=np.full(n, horizon, dtype=np.int8),
    )


def _good_for(task) -> object | None:
    """What a task must already have: a fetch waits for its good, a planting waits for its seed.

    A fetch and a planting are the same rule with different actors - one takes the good out of the
    shed, the other puts the seed in the ground - so both are bounded by the hour the planner says
    the thing is available.
    """
    action = str(getattr(task.action, "value", task.action))
    if action == "PICKUP":
        return task.item
    if action == "PLANT":
        return task.crop
    return None


def _action_code(action) -> int:
    codes = action_codes()
    return codes.get(action, -1) if action in codes else -1


def _item_code(item) -> int:
    """A good's row in the item table. `Item` is a union of enums, so the table is the three."""
    if item is None:
        return NO_ITEM
    return ITEM_CODE.get(item, NO_ITEM)
