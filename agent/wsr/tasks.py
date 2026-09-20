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

from agent.world.action_rules import CARRIES
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
    #: The engine's spelling of each task - ("PLANT", "WHEAT"), ("WATER",), ("PICKUP", "GOOSE", 1).
    #: Built once with the arrays rather than derived per turn: it is what the compiler emits, and
    #: a task list that cannot spell itself is not a task list.
    ops: list[tuple] = field(default_factory=list)
    actions: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int8))
    #: What each task must have in its worker's bag - the world's own `CARRIES` plus PLACE, which
    #: carries the animal. NO_ITEM for everything else. This is what a trip to a shed door is for.
    items: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int8))
    #: What each task puts in the bag - the world's own `YIELDS`. A HARVEST yields the crop, a
    #: COLLECT_FERTILIZER yields fertilizer, and neither is a good the task needs: the same op can
    #: appear in both columns for no op, and reading one as the other is a day that waits for what
    #: it is about to produce.
    yields: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int8))
    #: How many units `yields` puts in the bag - a harvest's `yield_units`, read off the tile.
    yield_n: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int8))
    #: For a DROP, the harvest it banks - and -1 for every other row. A drop hands over the whole
    #: bag, so what it banks is decided by which harvests happened since the drop before it; this
    #: column is what lets the search see that a second drop with nothing new in the bag is free.
    banks: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int16))
    cells: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), dtype=np.int16))
    columns: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int8))
    pred: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=bool))
    #: No fetch column: a fetch is not a task the day schedules. `items` says what good each task
    #: consumes, and the trip to a door is priced on the task itself - once per worker per good,
    #: because a bag is per worker and the second feeding of a day is already in it.
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
    def is_drop(self) -> np.ndarray:
        """The rows that hand the bag over."""
        return self.actions == ACTION_CODE[UnitAction.DROP]

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
    def edges(self) -> list[tuple[int, int]]:
        """The precedence edges as (after, before) row pairs.

        The matrix is the shape a broadcast wants, but the graph itself is a handful of edges per
        task, and walking them costs the number of edges instead of their square.
        """
        return self._edges

    @property
    def edge_after(self) -> np.ndarray:
        """The successors of every edge, as an array - the shape a scatter wants."""
        return self._edge_after

    @property
    def edge_before(self) -> np.ndarray:
        """The predecessors of every edge, as an array."""
        return self._edge_before

    @property
    def edge_groups(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The edges grouped by successor: their order, where each group starts, and its successor.

        `np.nonzero` walks the matrix in row order, so the edges arrive already sorted by the task
        they point at - the grouping is a shift and a comparison, not a sort. What it buys is that
        the answer for every successor becomes one segment reduction over the edges that reach it,
        instead of a scatter that writes each edge on its own.
        """
        return self._edge_order, self._edge_starts, self._edge_targets

    @property
    def pred_count(self) -> np.ndarray:
        """How many predecessors each task waits for.

        `pred[i, j]` means j precedes i, so the count for a task is its ROW sum. Summing the
        column would count the tasks a task precedes, which reads as a satisfied constraint the
        moment any descendant is done - and that is how a feed gets scheduled before its wheat.
        """
        return self.pred.sum(axis=1)

    def window(self, hour: np.ndarray) -> np.ndarray:
        """Which tasks may run at `hour`, per state - both columns at once.

        `hour` is (batch, n): the hour each state would reach each task. A task outside its window
        is not a candidate, which is how an action the clock forbids gets dropped rather than
        scheduled and discarded later.
        """
        return (hour >= self.earliest) & (hour <= self.latest)

    def __post_init__(self) -> None:
        # The transposed product, cast once. `ready` runs once per step and rebuilding this on
        # every call would cost more than the product it feeds.
        self._pred_f32 = self.pred.T.astype(np.float32)
        # The edges, read once. They are a function of the graph, and the walk that needs them runs
        # once per step: `np.nonzero` over the whole matrix on every call was a third of the time
        # that walk took, for the same answer every time.
        after, before = np.nonzero(self.pred)
        self._edge_after = after.astype(np.int32)
        self._edge_before = before.astype(np.int32)
        self._edges = list(zip(after.tolist(), before.tolist()))
        # Grouped by successor, which is the order they are already in. A day whose chains have no
        # precedence at all - one drop on its own - has no edges, and an empty group list has no
        # first index to read.
        self._edge_order = np.argsort(after, kind="stable").astype(np.int32)
        grouped = after[self._edge_order]
        if grouped.size:
            self._edge_starts = np.flatnonzero(
                np.r_[True, grouped[1:] != grouped[:-1]]).astype(np.int32)
            self._edge_targets = grouped[self._edge_starts].astype(np.int32)
        else:
            self._edge_starts = np.zeros(0, dtype=np.int32)
            self._edge_targets = np.zeros(0, dtype=np.int32)

    def ready(self, done: np.ndarray) -> np.ndarray:
        """Which tasks have all their predecessors done - one matrix product.

        `done` is (batch, n) booleans, and the result is (batch, n): True where a task may start.
        The matrix is transposed because `pred[i, j]` means j precedes i, so reading task i's row
        means asking for column i of the product - `done @ pred` would count the tasks i precedes,
        which is the opposite question and reads as satisfied the moment any descendant is done.

        Float32, not int8: numpy has no BLAS kernel for int8 and runs this product in its own loop,
        which is most of the cost of a step on a hundred tiles. A count fits a float32's 24-bit
        mantissa exactly, so this is the same comparison, not an approximation of it.
        """
        return (done.astype(np.float32) @ self._pred_f32) == self.pred_count

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


ACTION_CODE: dict[UnitAction, int] = action_codes()


SHED_INDEX = np.asarray([index_of(d) for d in SHED_ACCESS], dtype=np.int32)


def _item_table() -> dict:
    from agent.world.action import Animal, Crop, Product
    return {item: i for i, item in enumerate(list(Crop) + list(Animal) + list(Product))}


ITEM_CODE: dict = _item_table()


def build(chains, *, available: dict[str, int] | None = None, horizon: int = 24,
          drop_by=None) -> TaskArray:
    """The planner's chains -> the arrays a beam search reads.

    The tasks themselves come from `expand_chain`, which is the layer's own expansion of the chain
    the caller hands over - the registry's id is the caller's business, not this one's. The time
    columns are filled from the timetable that is handed in: a consumer waits for its good, a
    planting waits for its seed, and everything else may run from the first hour. Nothing is read
    from the world.
    """
    from agent.world.model import UnitAction
    from agent.wsr.models import MinorTask, expand_chain
    from agent.wsr.routing import nearest_shed

    available = available or {}
    #: One deadline per chain, aligned with `chains`: the latest hour that chain's harvest must be in
    #: the shed, or None to leave it for the night. A DROP is derived from it, never declared.
    drop_by = list(drop_by) if drop_by is not None else []
    tasks = []
    order: list[tuple[str, str]] = []
    column_of: dict[str, int] = {}
    for index, (cell, ops, entity) in enumerate(chains):
        # The chain's entity is passed twice, exactly as the instance compiler does: once as the
        # structure it builds and once as the good it places. Dropping the second leaves the
        # animals' feed unfetched, because the fetch is derived from what the chain places.
        expansion = expand_chain(ops, entity=entity, cell=cell, item=entity,
                                 prefix=f"d{index}_")
        tasks.extend(expansion.tasks)
        order.extend(expansion.order)
        for task in expansion.tasks:
            column_of[task.id] = index

    # A fetch is not a task: it is the trip a consumer makes when its good is not in the bag yet.
    # Leaving it in the list made the beam choose it before the consumer it serves, which locked
    # that consumer to a worker that may never go near the tile.
    from agent.world.model import UnitAction

    fetches = {t.id for t in tasks if t.action == UnitAction.PICKUP}
    tasks = [t for t in tasks if t.id not in fetches]
    order = [(b, a) for b, a in order if b not in fetches and a not in fetches]
    column_of = {tid: col for tid, col in column_of.items() if tid not in fetches}

    # A DROP is derived: a harvest the caller wants banked gets a drop of its own, and the drop's
    # cell is the door it hands the bag over at - so the walk to it is priced by the same rule as
    # any other task and nothing in the search has to know what a drop is.
    deadline_of: dict[str, int] = {}
    banks_of: dict[str, str] = {}
    for index in range(len(chains)):
        deadline = drop_by[index] if index < len(drop_by) else None
        if deadline is None:
            continue
        cell = chains[index][0]
        for task in list(tasks):
            if column_of.get(task.id) != index or task.action != UnitAction.HARVEST:
                continue
            drop_id = f"{task.id}_drop"
            tasks.append(MinorTask(id=drop_id, cell=nearest_shed(cell), action=UnitAction.DROP))
            order.append((task.id, drop_id))              # bank it after you take it
            column_of[drop_id] = index
            deadline_of[drop_id] = int(deadline)
            banks_of[drop_id] = task.id

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

    # The drop's deadline is the caller's, and it is the only bound on it: a drop that cannot land
    # by then is a sale the day cannot make, and the search is told so rather than discovering it.
    latest = np.full(n, horizon, dtype=np.int8)
    banks = np.full(n, -1, dtype=np.int16)
    for i, task in enumerate(tasks):
        if task.id in deadline_of:
            latest[i] = min(int(deadline_of[task.id]), horizon)
        if task.id in banks_of:
            banks[i] = row_of[banks_of[task.id]]

    # A fetch carries no cell - it happens at the shed door, and the door depends on where the
    # worker is, not on the tile the good is for. The tile numbering therefore comes from the
    # chain the task was built for, never from the coordinates: a fetch belongs to the tile it
    # serves, and numbering by cell would put it on a tile of its own.
    shed_door = SHED_ACCESS[0]
    cells = np.asarray([t.cell if t.cell else shed_door for t in tasks], dtype=np.int16)
    columns = np.asarray([column_of[t.id] for t in tasks], dtype=np.int8)

    item_codes = np.asarray([_item_code(t.item) for t in tasks], dtype=np.int16)
    # Two columns, from the world layer's two tables rather than from one guess. A PICKUP would be
    # the fourth kind of need and there are none left: a fetch is the trip a consumer makes.
    need_codes = np.full(n, NO_ITEM, dtype=np.int16)
    yield_codes = np.full(n, NO_ITEM, dtype=np.int16)
    yield_units = np.zeros(n, dtype=np.int16)
    for i, task in enumerate(tasks):
        name = str(getattr(task.action, "value", task.action))
        if name in NEED_OPS:
            need_codes[i] = item_codes[i]
        elif name == "HARVEST":
            yield_codes[i] = item_codes[i]              # the crop the tile hands over
            yield_units[i] = int(getattr(task, "n", 1) or 1)
        elif name == "COLLECT_FERTILIZER":
            yield_codes[i] = _item_code("FERTILIZER")
            yield_units[i] = 1
    return TaskArray(
        ids=ids,
        ops=[_engine_op(t) for t in tasks],
        actions=np.asarray([_action_code(t.action) for t in tasks], dtype=np.int8),
        items=need_codes.astype(np.int8),
        yields=yield_codes.astype(np.int8),
        yield_n=yield_units.astype(np.int8),
        cells=cells,
        columns=columns,
        pred=pred,
        earliest=earliest,
        latest=latest,
        banks=banks,
    )


def _engine_op(task) -> tuple:
    """A task as the engine spells it.

    The argument of an op is the thing it names: PLANT names its crop and PLACE its species, which
    for a chain is the chain's entity. Everything else takes no argument - the ops that consume a
    carried good name nothing, because the good is in the bag and not in the op.
    """
    from agent.world.model import UnitAction

    name = str(getattr(task.action, "value", task.action))
    if name == UnitAction.PICKUP.value:
        return (name, str(getattr(task.item, "value", task.item)), int(task.n))
    if name == UnitAction.PLANT.value:
        return (name, str(getattr(task.crop, "value", task.crop)))
    if name == UnitAction.PLACE.value:
        return (name, str(getattr(task.item, "value", task.item)))
    return (name,)


#: The ops that must have a good in the worker's bag, from the world's own table plus PLACE.
NEED_OPS: frozenset[str] = frozenset(CARRIES) | {"PLACE"}


def _good_for(task) -> object | None:
    """What a task must already have: a fetch waits for its good, a planting waits for its seed.

    A fetch and a planting are the same rule with different actors - one takes the good out of the
    shed, the other puts the seed in the ground - so both are bounded by the hour the planner says
    the thing is available.
    """
    action = str(getattr(task.action, "value", task.action))
    if action == "PLANT":
        return task.crop
    # A task that consumes a good - PLACE, FEED, FERTILIZE - waits for the same hour, because the
    # trip that brings the good cannot happen before the shed holds it. Without this the search
    # prices a feeding at the turn the day opens, when the animal it feeds is still in the market.
    #
    # A HARVEST is NOT here. Its item is what it yields, and waiting for the crop to be in the shed
    # before the tile can be harvested is the wait that never ends.
    if action not in NEED_OPS:
        return None
    item = getattr(task, "item", None)
    return item if _item_code(item) >= 0 else None


def _action_code(action) -> int:
    codes = action_codes()
    return codes.get(action, -1) if action in codes else -1


def _item_code(item) -> int:
    """A good's row in the item table. `Item` is a union of enums, so the table is the three."""
    if item is None:
        return NO_ITEM
    return ITEM_CODE.get(item, NO_ITEM)
