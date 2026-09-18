"""The WSR's data model: a day's work, and the instance a solver schedules.

A chain of ops (from the DP's registry) becomes the minor tasks that make it up, plus the
precedence between them and the pairs that one worker must do together. The `Instance`
validates all of that once and pre-computes the lookups the solvers share, so no solver
rediscovers the structure.

The chain's own order is a constraint only where it changes the outcome: a ONE-SHOT crop is
watered before it is harvested, and the engine does not care in what order an ongoing crop is
fed, watered and harvested. Imposing the order everywhere would forbid days that are fine.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import NamedTuple, Optional, Sequence

from agent.world.action import Item, item_of
from agent.world.action_rules import CARRIES
from agent.world.action import Action
from agent.world.model import UnitAction, ANIMALS, CROPS, MOVES, PRODUCTS, TileKind
from agent.world.action import _op_name
from agent.world.rules import ANIMAL_STRUCTURE, CROP_RULES, SHED_ACCESS, TURNS_PER_DAY


# Derived views of the one vocabulary (ARCHITECTURE §5 step 4). PASS counts as a
# movement here because it is what a unit does instead of moving.
MOVEMENT_ACTIONS: frozenset[str] = frozenset(MOVES) | {UnitAction.PASS}
PRODUCE_ACTIONS: frozenset[str] = frozenset(
    (UnitAction.PICKUP, UnitAction.HARVEST, UnitAction.COLLECT_FERTILIZER))
CONSUME_ACTIONS: frozenset[str] = frozenset(
    (UnitAction.PLACE, UnitAction.FEED, UnitAction.FERTILIZE))
ANIMAL_ITEMS: frozenset[str] = frozenset(ANIMALS)
CROP_ITEMS: frozenset[str] = frozenset(CROPS)
PRODUCT_ITEMS: frozenset[str] = frozenset(PRODUCTS)

# ============================================================================
# 1. Basic Enums and Data Structures
# ============================================================================

class Cell(NamedTuple):
    x: int
    y: int

# The shed's four access tiles, in the engine's NWSE order (world/model.py).
WAREHOUSE_ENTRY_ORDER: tuple[str, ...] = ("NW", "NE", "SW", "SE")
WAREHOUSE_ENTRY_CELLS: dict[str, Cell] = {
    name: Cell(*tile) for name, tile in zip(WAREHOUSE_ENTRY_ORDER, SHED_ACCESS)}


@dataclass
class MinorTask:
    id: str
    cell: Optional[Cell]
    action: Action
    item: Optional[Item] = None
    n: int = 1
    crop: Optional[Item] = None


@dataclass
class Worker:
    index: int
    earliest_start: int

    @property
    def cost(self) -> int:
        """What adding this worker costs the SCHEDULE, not the purse.

        The WSR does not handle money: what it minimises is the number of workers, so the cost
        rises with the index and the first two hands are free. What the day actually pays - the
        engine's hire ladder, and how many hands were hired - is the market's business.
        """
        return max(self.index - 1, 0)


@dataclass
class ScheduledTask:
    task_id: str
    exec_time: int
    resolved_cell: Optional[Cell] = None
    resolved_n: Optional[int] = None


@dataclass
class WorkerRoute:
    worker_index: int
    start_time: int
    tasks: list[ScheduledTask] = field(default_factory=list)
    start_cell: Optional[Cell] = None  # assumed entry point (placement rule)


@dataclass
class Solution:
    routes: list[WorkerRoute] = field(default_factory=list)
    hired: Optional[int] = None      # how many hands the day hires: max_active - 1


# ============================================================================
# 2. Major Task Definitions & Expansion Logic
# ============================================================================

#: A day's work: a chain of the DP's registry, the cell it runs on, and the entity standing
#: there (the crop or animal whose rules decide whether the op order matters).
DayOnCell = NamedTuple("DayOnCell", [("ops", Sequence[str]), ("cell", Cell),
                                     ("entity", Optional[Item]), ("item", Optional[Item])])

#: Op pairs whose order the engine enforces whatever the entity, because the first op brings
#: the thing the second one acts on into existence. The order of the daily acts - fertilise,
#: water, harvest - is NOT here: those commute for an ongoing crop and for an animal.
STRUCTURAL_ORDER: tuple[tuple[str, str], ...] = (
    ("BUILD_COOP", "PLACE"),
    ("BUILD_PASTURE", "PLACE"),
    ("PLACE", "FEED"),
    ("PLANT", "WATER"),
)

def structural_edges(op_ids: Sequence[tuple[str, str]]) -> list[tuple[str, str]]:
    """The order the engine enforces whatever the entity, as edges between task ids.

    For each pair the engine cares about, an op is linked to the FIRST later op of the wanted
    kind - which is what "the planting's own water" means in `WATER, HARVEST, PLANT, WATER`:
    the first water belongs to the crop being harvested, the second to the one just planted.
    """
    edges: list[tuple[str, str]] = []
    for before, after in STRUCTURAL_ORDER:
        for index, (name, task_id) in enumerate(op_ids):
            if name != before:
                continue
            for later_name, later_id in op_ids[index + 1:]:
                if later_name == after:
                    edges.append((task_id, later_id))
                    break
    return edges


ExpansionResult = tuple[list[MinorTask], list[tuple[str, str]], list[str]]


def expand_chain(ops: Sequence[str], entity: Item | None = None, cell: Cell | None = None,
                 item: Item | None = None, harvested_n: int = 1,
                 prefix: str = "") -> ExpansionResult:
    """One day's chain -> its tasks, the precedence between them, and the pairs that
    the SAME worker must do (a PICKUP and the op that consumes what it carried).

    Every op that eats a carried good is preceded by its own PICKUP. The chain's order is
    imposed as precedence only for a ONE-SHOT crop: there it decides the outcome (a crop must
    be watered before it is harvested), while an ongoing crop and an animal are indifferent to
    the order of their ops, and a total order would forbid legal days.
    """
    ordered = entity in CROP_RULES and not CROP_RULES[entity]["ongoing"]
    tasks: list[MinorTask] = []
    precedence: list[tuple[str, str]] = []
    same_worker: list[str] = []
    seen: dict[str, int] = {}
    op_ids: list[tuple[str, str]] = []
    previous: Optional[str] = None

    def unique(name: str) -> str:
        seen[name] = seen.get(name, 0) + 1
        suffix = "" if seen[name] == 1 else str(seen[name])
        return f"{prefix}{name}{suffix}"

    def link(before: Optional[str], after: str) -> None:
        """Record the order, when the order is a constraint at all."""
        if before is not None and ordered:
            precedence.append((before, after))

    for op in ops:
        name = f"BUILD_{ANIMAL_STRUCTURE[item]}" if op == "BUILD" else op
        carried = CARRIES.get(name)
        if carried is not None:
            carried = item_of(carried)
        elif name == "PLACE":
            carried = item

        if carried is not None:
            acquire = MinorTask(id=unique("acquire"), cell=None,
                                action=UnitAction.PICKUP, item=carried, n=1)
            tasks.append(acquire)
            link(previous, acquire.id)
            previous = acquire.id
            same_worker.append(acquire.id)      # the worker that carries it is the one to use it

        task = MinorTask(
            id=unique(name.lower()),
            cell=cell,
            action=UnitAction(name),
            item=entity if name == "HARVEST" else carried,
            n=harvested_n if name == "HARVEST" else 1,
            crop=entity if name == "PLANT" else None)
        tasks.append(task)
        link(previous, task.id)
        previous = task.id
        if carried is not None:
            same_worker.append(task.id)
        op_ids.append((name, task.id))

    # a pair can be required both by the chain's own order and by the structural table
    for edge in structural_edges(op_ids):
        if edge not in precedence:
            precedence.append(edge)
    return tasks, precedence, same_worker


# ============================================================================
# 3. Rich Domain Model (The Instance)
# ============================================================================

@dataclass
class Instance:
    """
    A fully compiled, validated, and pre-processed problem instance.
    Guarantees structural integrity (no cycles, no missing refs) and provides 
    O(1) lookup tables for all solvers.
    """
    minor_tasks: list[MinorTask]
    precedence: list[tuple[str, str]] = field(default_factory=list)
    single_worker_groups: list[list[str]] = field(default_factory=list)
    warehouse_stock: dict[Item, int] = field(default_factory=dict)
    workers: list[Worker] = field(default_factory=list)
    horizon: int = TURNS_PER_DAY
    clct_deadline: Optional[int] = None
    worker_pool_size: Optional[int] = None

    # --- Pre-computed Lookups (Hidden from repr to keep console clean) ---
    tasks_by_id: dict[str, MinorTask] = field(init=False, repr=False)
    task_index: dict[str, int] = field(init=False, repr=False)
    group_of: dict[str, int] = field(init=False, repr=False)
    
    # --- Structural Data for Solvers ---
    aggregatable_pickups: frozenset[str] = field(init=False, repr=False)
    target_needs_item: dict[str, Item] = field(init=False, repr=False)
    item_to_pickups: dict[Item, list[str]] = field(init=False, repr=False)
    target_tasks: frozenset[str] = field(init=False, repr=False)
    target_preds: dict[str, list[str]] = field(init=False, repr=False)

    def __post_init__(self):
        """Validates the instance and computes all necessary lookup tables."""
        # 1. Base Task Lookups & Deduplication Check
        self.tasks_by_id = {}
        self.task_index = {}
        for idx, task in enumerate(self.minor_tasks):
            if task.id in self.tasks_by_id:
                raise ValueError(f"Duplicate task ID found: {task.id!r}")
            self.tasks_by_id[task.id] = task
            self.task_index[task.id] = idx

        # 2. Validate Precedence
        for p, s in self.precedence:
            if p not in self.tasks_by_id or s not in self.tasks_by_id:
                raise ValueError(f"Precedence edge references unknown task: {p!r} -> {s!r}")

        # 3. Validate Groups & Build group_of
        self.group_of = {}
        for g_idx, group in enumerate(self.single_worker_groups):
            for tid in group:
                if tid not in self.tasks_by_id:
                    raise ValueError(f"Single worker group references unknown task: {tid!r}")
                self.group_of[tid] = g_idx

        # 4. Cycle Detection (Fail-fast before reaching solvers)
        self._validate_no_cycles()

        # 5. Pre-compute Aggregatable Pickups (Logic extracted from solvers)
        agg_pickups = set()
        self.target_needs_item = {}
        self.item_to_pickups = {}
        precedence_set = set(self.precedence)

        for group in self.single_worker_groups:
            if len(group) == 2:
                t1, t2 = self.tasks_by_id[group[0]], self.tasks_by_id[group[1]]
                # `action` is an `Action` for the chain's own ops and a `UnitAction` for the
                # PICKUPs inserted above, so compare the op's name: `Action(...) ==
                # UnitAction.PICKUP` is never true and the aggregation never fired.
                is_pickup = lambda t: _op_name(t.action) == UnitAction.PICKUP.value
                pickup = t1 if is_pickup(t1) else (t2 if is_pickup(t2) else None)
                consume = t2 if pickup == t1 else (t1 if pickup == t2 else None)
                
                if (pickup and consume and pickup.item and pickup.cell is None and 
                    _op_name(consume.action) in {op.value for op in CONSUME_ACTIONS}
                    and consume.item == pickup.item):
                    
                    if (pickup.id, consume.id) in precedence_set:
                        agg_pickups.add(pickup.id)
                        self.target_needs_item[consume.id] = pickup.item
                        if pickup.item not in self.item_to_pickups:
                            self.item_to_pickups[pickup.item] = []
                        self.item_to_pickups[pickup.item].append(pickup.id)

        self.aggregatable_pickups = frozenset(agg_pickups)

        # 6. Pre-compute Target Tasks (For Spatial/Heuristic Solvers)
        self.target_tasks = frozenset(t.id for t in self.minor_tasks if t.id not in self.aggregatable_pickups)
        self.target_preds = {t: [] for t in self.target_tasks}
        for p, s in self.precedence:
            if p in self.target_tasks and s in self.target_tasks:
                self.target_preds[s].append(p)


    def _validate_no_cycles(self):
        """Topological cycle detection to ensure a feasible precedence graph."""
        adj = {t.id: [] for t in self.minor_tasks}
        for p, s in self.precedence:
            adj[p].append(s)
            
        visited = set()
        rec_stack = set()
        
        def is_cyclic(node: str) -> bool:
            visited.add(node)
            rec_stack.add(node)
            for neighbor in adj[node]:
                if neighbor not in visited:
                    if is_cyclic(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True
            rec_stack.remove(node)
            return False
            
        for node in adj:
            if node not in visited:
                if is_cyclic(node):
                    raise ValueError("Cycle detected in precedence graph! The problem is mathematically unsolvable.")


    @classmethod
    def compile(
        cls,
        *,
        workers: list[Worker],
        days: Sequence[DayOnCell] = (),
        standalone_minor_tasks: Optional[list[MinorTask]] = None,
        explicit_precedence: Optional[list[tuple[str, str]]] = None,
        explicit_single_worker_groups: Optional[list[list[str]]] = None,
        warehouse_stock: Optional[dict[Item, int]] = None,
        horizon: int = TURNS_PER_DAY,
        clct_deadline: Optional[int] = None,
        worker_pool_size: Optional[int] = None,
    ) -> "Instance":
        """A day's work -> the instance a scheduler reads.

        A day is one chain of the DP's registry on one cell, so the caller passes the days and
        the tasks come out of them; `standalone_minor_tasks` is for work that is not a chain.
        Task ids are prefixed per day so two cells running the same chain stay distinct.
        """
        minor_tasks = list(standalone_minor_tasks or [])
        precedence = list(explicit_precedence or [])
        groups = [list(group) for group in (explicit_single_worker_groups or [])]

        seen_ids = {task.id for task in minor_tasks}
        
        for index, day in enumerate(days):
            tasks, edges, group = expand_chain(day.ops, day.entity, day.cell, day.item,
                                               prefix=f"d{index}_")
            for task in tasks:
                if task.id in seen_ids:
                    raise ValueError(f"duplicate task id {task.id!r} (day {index})")
                seen_ids.add(task.id)
            minor_tasks.extend(tasks)
            precedence.extend(edges)
            if group:
                groups.append(group)

        return cls(
            minor_tasks=minor_tasks,
            precedence=precedence,
            single_worker_groups=groups,
            warehouse_stock=dict(warehouse_stock or {}),
            workers=list(workers),
            horizon=horizon,
            clct_deadline=clct_deadline,
            worker_pool_size=worker_pool_size,
        )