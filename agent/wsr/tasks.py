"""The day as arrays: every task of the day, one per row, with its constraints as columns.

`expand_chain` (`models.py`) gives a chain's tasks as `MinorTask`s, which suits a loop that walks
one task at a time. A beam search asks the same questions about every task at once, so it wants the
same facts in arrays: a precedence matrix instead of a list of pairs, an item column instead of a
dict of needs, and one distance matrix for the whole board instead of a call per hop.

The board is fixed at ten by ten, so the distance matrix is built once per process and indexed
after that. It is symmetric with a zero diagonal, and no cell is further than eighteen apart, so
it fits in a byte per entry.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from agent.world.action_rules import CARRIES, YIELDS
from agent.world.model import UnitAction
from agent.world.rules import BOARD_SIZE, SHED_ACCESS

NO_ITEM = -1

#: The ops that put a good in a worker's bag off a TILE - what a drop has to bank. The world's own
#: `YIELDS` is the source; PICKUP is in that table too, but what it takes is already in the shed, so
#: a day that picked up and dropped would be walking in a circle.
BANKS_A_DROP: tuple[str, ...] = tuple(op for op in YIELDS if op != "PICKUP")


def distance_matrix(size: int = BOARD_SIZE) -> np.ndarray:
    """Manhattan distance between every pair of cells, built once.

    Row and column index a cell as `y * size + x`, which is the order the game's tile rows use.
    """
    coords = np.arange(size)
    ys, xs = np.meshgrid(coords, coords, indexing="ij")
    flat = np.stack([ys.ravel(), xs.ravel()], axis=1).astype(np.int16)
    delta = np.abs(flat[:, None, :] - flat[None, :, :])
    # int16, the width the search's arithmetic runs in: the walk is gathered once per step and cast
    # to this width every time, and a table in the answer's own width skips that pass.
    return delta.sum(axis=-1).astype(np.int16)


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
    #: The tasks each task must share a WORKER with, as row indices, padded with -1: one row per
    #: group-mate slot. A fetch and the op that consumes it are one group (the good is in one
    #: worker's bag), and a drop is in the group of the task it banks (a drop by anybody else hands
    #: over nothing). Empty when the day has no groups.
    ties: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=np.int16))
    cells: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), dtype=np.int16))
    columns: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int8))
    pred: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=bool))
    #: How many tasks each task's miss kills: the size of its downstream closure on `pred`,
    #: excluding the task itself. The rules (`ORDER_MATTERS`) own the order and `pred` carries it;
    #: the closure is its consequence, and a ranking that cannot see it reads a missed WATER as
    #: worth one task when it rots the HARVEST->PLANT->WATER chain behind it. Filled in
    #: `__post_init__` from `pred` - one source, the rules' own edges, never a second table.
    chain_weight: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int16))
    #: No fetch column: a fetch is not a task the day schedules. `items` says what good each task
    #: consumes; the worker's door load and any refetch after a DROP are priced by the search
    #: (`beam.door_load`, `beam.legs`) - once per worker per good, because a bag is per worker.
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
    def successor_groups(self) -> tuple[np.ndarray, np.ndarray]:
        """Every task's successors, flat, with where each task's slice of it starts.

        The counter that answers `ready` advances a task's successors when it is placed, so it needs
        them gathered the other way round from `edge_groups`: grouped by the PREDECESSOR, as one flat
        list and one start per task.
        """
        return self._succ_flat, self._succ_start

    @property
    def pred_count16(self) -> np.ndarray:
        """How many predecessors each task waits for, in the width the counter is kept in."""
        return self._pred_count

    @property
    def drop_rows(self) -> np.ndarray:
        """Which rows are drops. A function of the list, and read once per step."""
        return self._drop_rows

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
        self._drop_rows = np.flatnonzero(self.is_drop).astype(np.int32)
        # The successors, grouped by the task they follow, for the counter that answers `ready`.
        self._succ_flat = after[np.argsort(before, kind="stable")].astype(np.int32)
        self._succ_start = np.r_[
            0, np.cumsum(np.bincount(before, minlength=self.n))].astype(np.int32)
        self._pred_count = self.pred.sum(axis=1).astype(np.int16)
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
        # The closure, once per array: the consequence of the rules' own order. `succ[i, k]` is
        # "k directly after i", the sweep squares the reach matrix until it stops growing - a dag
        # converges, and the day's tasks bound it - and the weight is the closure's size. Booleans
        # squared, not counted: a count of paths, not of tasks, is what a second sweep would make.
        succ = self.pred.T                            # succ[i, k]: k directly after i
        reach = succ.copy()
        while True:
            grown = reach | (reach @ reach)
            if np.array_equal(grown, reach):
                break
            reach = grown
        self.chain_weight = reach.sum(axis=1).astype(np.int16)

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


#: The layers of `land_image`, in order, and what each one counts.
IMAGE_LAYERS: tuple[str, ...] = (
    "tasks",        # how many tasks the tile carries
    "wheat",        # the tile needs wheat: a FEED eats one
    "fertilizer",   # the tile needs fertilizer: a FERTILIZE spreads one
    "animal",       # the tile takes an animal: a PLACE puts one down
    "drop",         # the tile's harvest has to reach the shed by a deadline
    "depth",        # the longest chain on the tile, in tasks
)


def chain_depth(tasks: TaskArray) -> np.ndarray:
    """The longest precedence chain each task sits in, in tasks.

    `pred[i, j]` means j precedes i, so a task sits one below its deepest predecessor. The graph is
    acyclic, so repeated relaxation converges and no recursion is needed.
    """
    if tasks.n == 0:
        return np.zeros(0, dtype=np.int16)
    depth = np.ones(tasks.n, dtype=np.int16)
    for _ in range(tasks.n):
        with_pred = tasks.pred.any(axis=1)
        deeper = (depth[None, :] * tasks.pred).max(axis=1) + 1
        updated = np.where(with_pred, deeper, depth).astype(np.int16)
        if (updated == depth).all():
            break
        depth = updated
    return depth


def land_image(tasks: TaskArray, board_size: int = BOARD_SIZE) -> np.ndarray:
    """The day as one board per quantity, stacked: shape (board, board, len(IMAGE_LAYERS)).

    A view of the arrays rather than a second source of truth - every layer is a scatter of a column
    `build` already filled. Read it as `image[x, y]` for a tile's own vector, or `image[:, :, n]` for
    a whole board of one quantity.
    """
    from agent.world.action import Animal, Product

    image = np.zeros((board_size, board_size, len(IMAGE_LAYERS)), dtype=np.int16)
    if tasks.n == 0:
        return image
    x = tasks.cells[:, 0].astype(np.int64)
    y = tasks.cells[:, 1].astype(np.int64)
    depth = chain_depth(tasks)

    np.add.at(image[:, :, 0], (x, y), 1)
    wheat, fertilizer = _item_code(Product.WHEAT), _item_code(Product.FERTILIZER)
    animal_codes = {_item_code(a) for a in Animal}
    np.maximum.at(image[:, :, 1], (x, y), (tasks.items == wheat).astype(np.int16))
    np.maximum.at(image[:, :, 2], (x, y), (tasks.items == fertilizer).astype(np.int16))
    np.maximum.at(image[:, :, 3], (x, y),
                  np.isin(tasks.items, list(animal_codes)).astype(np.int16))
    np.maximum.at(image[:, :, 5], (x, y), depth)
    if tasks.drop_rows.size:
        rows = tasks.drop_rows
        np.maximum.at(image[:, :, 4], (tasks.cells[rows, 0].astype(np.int64),
                                       tasks.cells[rows, 1].astype(np.int64)), 1)
    return image


def spanning_walk(tasks: TaskArray) -> int:
    """A floor on the walking: the minimum spanning tree over the worked tiles, rooted at the shed.

    Any set of walks that covers the tiles, starting from the shed, is a connected subgraph over the
    tiles and the shed together, and the cheapest such subgraph is the tree. `tiles - 1` is the same
    idea with the crossings left out, which is why a day that works three quadrants needs more.

    The shed is ONE node and it is the root, so a tile's edge to it costs the distance to the NEAREST
    of its four access tiles: a worker may start on any of them, and charging one representative door
    overcharges every tile that is nearer another. Leaving the four as four nodes is worse still - the
    tree then pays the 2x2 block's own cost, up to three steps no worker walks - and a floor that is
    too high is worse than useless: it reports a hand the day does not need.
    """
    if tasks.n == 0:
        return 0
    cells, first = np.unique(tasks.cells, axis=0, return_index=True)
    nodes = [tuple(int(v) for v in cell) for cell in cells]
    if not nodes:
        return 0
    nearest = DISTANCE[SHED_INDEX].min(axis=0)[tasks.cell_index[first]]

    far = lambda a, b: abs(a[0] - b[0]) + abs(a[1] - b[1])  # noqa: E731 - one expression, one name
    inside: set[int] = set()
    best = {i: int(nearest[i]) for i in range(len(nodes))}
    total = 0
    while len(inside) < len(nodes):
        pick = min((i for i in best if i not in inside), key=lambda i: best[i])
        total += best[pick]
        inside.add(pick)
        for i in range(len(nodes)):
            if i not in inside:
                best[i] = min(best[i], far(nodes[pick], nodes[i]))
    return total


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
          drop_by=None, harvests=None) -> TaskArray:
    """The planner's chains -> the arrays a beam search reads.

    The tasks themselves come from `expand_chain`, which is the layer's own expansion of the chain
    the caller hands over - the registry's id is the caller's business, not this one's. The time
    columns are filled from the timetable that is handed in: a consumer waits for its good, a
    planting waits for its seed, and everything else may run from the first hour. Nothing is read
    from the world.

    `harvests`, aligned with `chains`, is what each chain's HARVEST takes: `(good, units)` - the
    tile's crop or product and its `yield_units` at that moment - or None to keep the chain's own
    entity and one unit. The caller reads it off the board; this layer never does.
    """
    from agent.world.action import item_of
    from agent.world.model import UnitAction
    from agent.wsr.models import MinorTask, expand_chain
    from agent.wsr.routing import nearest_shed

    available = available or {}
    harvests = list(harvests) if harvests is not None else []
    #: One deadline per chain, aligned with `chains`: the latest hour that chain's harvest must be in
    #: the shed, or None to leave it for the night. A DROP is derived from it, never declared.
    drop_by = list(drop_by) if drop_by is not None else []
    tasks = []
    order: list[tuple[str, str]] = []
    #: Groups of task ids that must be done BY THE SAME WORKER. `expand_chain` builds one per fetch,
    #: and a drop joins the group of the task it banks (see below).
    groups: list[list[str]] = []
    column_of: dict[str, int] = {}
    for index, (cell, ops, entity) in enumerate(chains):
        # The chain's entity is passed twice, exactly as the instance compiler does: once as the
        # structure it builds and once as the good it places. Dropping the second leaves the
        # animals' feed unfetched, because the fetch is derived from what the chain places.
        expansion = expand_chain(ops, entity=entity, cell=cell, item=entity,
                                 prefix=f"d{index}_")
        tasks.extend(expansion.tasks)
        order.extend(expansion.order)
        groups.extend(expansion.groups)
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

    # A DROP is derived: a good the worker took off a tile gets a drop of its own, and the drop's
    # cell is the door it hands the bag over at - so the walk to it is priced by the same rule as
    # any other task and nothing in the search has to know what a drop is. The ops that put a good in
    # a worker's bag are the world's own `YIELDS`; PICKUP is in that table too, but what it takes is
    # already in the shed, so it needs no walk back. The bag is the worker's, so a drop banks what its
    # own worker took: the order edge below is the whole of the precedence a drop needs.
    deadline_of: dict[str, int] = {}
    banks_of: dict[str, str] = {}
    for index in range(len(chains)):
        deadline = drop_by[index] if index < len(drop_by) else None
        if deadline is None:
            continue
        cell = chains[index][0]
        for task in list(tasks):
            if column_of.get(task.id) != index or task.action.value not in BANKS_A_DROP:
                continue
            drop_id = f"{task.id}_drop"
            tasks.append(MinorTask(id=drop_id, cell=nearest_shed(cell), action=UnitAction.DROP))
            order.append((task.id, drop_id))              # bank it after you take it
            groups.append([task.id, drop_id])             # and bank it by the same worker
            column_of[drop_id] = index
            deadline_of[drop_id] = int(deadline)
            banks_of[drop_id] = task.id

    ids = [t.id for t in tasks]
    row_of = {tid: i for i, tid in enumerate(ids)}
    n = len(tasks)

    # The groups as row indices, one column per mate slot. A member that is not a row carries no
    # tie - a fetch is the trip a consumer makes rather than a task of its own - so a group that is
    # left with one member is no constraint at all.
    rows_of_group = [[row_of[tid] for tid in group if tid in row_of] for group in groups]
    rows_of_group = [group for group in rows_of_group if len(group) > 1]
    mate_width = max((len(group) - 1 for group in rows_of_group), default=0)
    ties = np.full((n, mate_width), -1, dtype=np.int16)
    for group in rows_of_group:
        for row in group:
            mates = [other for other in group if other != row][:mate_width]
            ties[row, :len(mates)] = mates

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
            taken = harvests[column_of[task.id]] if column_of[task.id] < len(harvests) else None
            if taken is not None:
                # What the tile holds when it is harvested, not what the chain plants next: the
                # HARVEST moves the tile's own `yield_units` of its own crop or product
                # (`action_rules` HARVEST).
                good, units = taken
                yield_codes[i] = _item_code(item_of(good)) if good else NO_ITEM
                yield_units[i] = max(1, int(units))
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
        ties=ties,
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


def day_walking(tasks: TaskArray) -> int:
    """The walking a day cannot pay less than: the tree over its tiles, doubled on a deadline day.

    A spanning tree over the worked tiles and the shed doors is a floor on any set of walks that
    covers them, and unlike the tile count it pays for the crossings. A day with a drop deadline adds
    a second floor: a unit has to reach the furthest tile and finish at a door, so its path is at
    least twice the distance from that door to that tile.
    """
    if tasks.n == 0:
        return 0
    walking = spanning_walk(tasks)
    if tasks.drop_rows.size:
        reach = int(DISTANCE[SHED_INDEX].min(axis=0)[tasks.cell_index].max())
        walking = max(walking, 2 * reach)
    return walking
