"""
Core data model and Rich Domain Instance for ChistaWRS.
Combines data structures, major task expansion, and instance compilation into a 
single, validated, and fully pre-processed domain model.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import NamedTuple, Optional

from agent.world.action import Item, item_of
from agent.world.action_rules import CARRIES
from agent.world.action import Action
from agent.world.model import UnitAction, ANIMALS, CROPS, MOVES, PRODUCTS, TileKind
from agent.world.rules import ANIMAL_STRUCTURE, SHED_ACCESS

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
class CellState:
    type: Optional[TileKind] = None
    is_wheat: bool = False
    watered: Optional[bool] = None
    fertilized: Optional[bool] = None
    fed: Optional[bool] = None
    produced_fer: Optional[bool] = None
    occupied: Optional[bool] = None
    cared: Optional[bool] = None
    n_yield: int = 0
    harvest_to_none: Optional[bool] = None


@dataclass
class MinorTask:
    id: str
    cell: Optional[Cell]
    action: Action
    item: Optional[Item] = None
    qty: int = 1
    crop: Optional[Item] = None


@dataclass
class Worker:
    index: int
    earliest_start: int

    @property
    def cost(self) -> int:
        from .fibonacci import fibonacci_cost
        return fibonacci_cost(self.index)


@dataclass
class ScheduledTask:
    task_id: str
    exec_time: int
    resolved_cell: Optional[Cell] = None
    resolved_qty: Optional[int] = None


@dataclass
class WorkerRoute:
    worker_index: int
    start_time: int
    tasks: list[ScheduledTask] = field(default_factory=list)
    start_cell: Optional[Cell] = None  # assumed entry point (placement rule)


@dataclass
class Solution:
    routes: list[WorkerRoute] = field(default_factory=list)
    reported_cost: Optional[int] = None


# ============================================================================
# 2. Major Task Definitions & Expansion Logic
# ============================================================================

#: The scheduling types the WSR's solvers are built on. Each is a chain from the
#: DP registry, so there is no second definition of a day's work.
#: `wet_harvst_plnt` is the exception: the registry's rotation is
#: WATER-HARVEST-DIG-PLANT-WATER, and the solver's measured tables (F057) were
#: taken on the four-op form, so re-basing it is a re-measure, not a rename.
MAJOR_CHAINS: dict[str, tuple[str, ...]] = {
    "feed": ("FEED",),
    "frtz": ("FERTILIZE",),
    "frtz_water": ("FERTILIZE", "WATER"),
    "plnt": ("PLANT", "WATER"),
    "wet_harvst": ("WATER", "HARVEST"),
    "wet_harvst_plnt": ("WATER", "HARVEST", "PLANT", "WATER"),
    "place_animal": ("PLACE",),
}

@dataclass
class MajorTask:
    id: str
    type: str              # a key of MAJOR_CHAINS
    cell: Cell
    crop: Optional[Item] = None
    harvested_item: Optional[Item] = None
    harvested_qty: int = 1
    item: Optional[Item] = None


ExpansionResult = tuple[list[MinorTask], list[tuple[str, str]], list[str]]


def expand_major_task(major: MajorTask) -> ExpansionResult:
    """A major task -> its minor tasks, precedences and material-critical ids.

    Derived from the type's chain (`MAJOR_CHAINS`) through `world/model.py`: the
    chain's ops in order, each op that eats a carried good preceded by its
    PICKUP. The chain's order IS the precedence.
    """
    chain = MAJOR_CHAINS.get(major.type)
    if chain is None:
        raise ValueError(f"unknown major_task type: {major.type!r}")
    tid = major.id
    minors: list[MinorTask] = []
    prec: list[tuple[str, str]] = []
    critical: list[str] = []
    seen: dict[str, int] = {}
    previous: Optional[str] = None

    def fresh(name: str) -> str:
        seen[name] = seen.get(name, 0) + 1
        return f"{tid}_{name}" if seen[name] == 1 else f"{tid}_{name}{seen[name]}"

    for op in chain:
        # The minor task's action is the chain op itself: the WSR schedules an
        # abstract day, and `compile_op`'s validation belongs to the engine-facing
        # compiler (a `PLANT` minor task carries its crop in `crop`, and the
        # solver's instances legitimately leave it out).
        if op == "BUILD":
            name = f"BUILD_{ANIMAL_STRUCTURE[major.item]}"
        elif op == "NO_ACT":
            name = "PASS"
        else:
            name = op
        carried = CARRIES.get(name)
        if carried is not None:
            carried = item_of(carried)       # the engine's name -> the model's item
        elif name == "PLACE":
            carried = major.item
        if carried is not None:
            acquire = MinorTask(id=fresh("acquire"), cell=None, action=UnitAction.PICKUP,
                                item=carried, qty=1)
            minors.append(acquire)
            if previous is not None:
                prec.append((previous, acquire.id))
            previous = acquire.id
            critical.append(acquire.id)
        minor = MinorTask(
            id=fresh(name.lower()), cell=major.cell, action=Action(name),
            item=major.harvested_item if name == "HARVEST" else carried,
            qty=major.harvested_qty if name == "HARVEST" else 1,
            crop=major.crop if name == "PLANT" else None)
        minors.append(minor)
        if previous is not None:
            prec.append((previous, minor.id))
        previous = minor.id
        if carried is not None:
            critical.append(minor.id)
    return minors, prec, critical


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
    horizon: int = 24
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
                pickup = t1 if t1.action == UnitAction.PICKUP else (t2 if t2.action == UnitAction.PICKUP else None)
                consume = t2 if pickup == t1 else (t1 if pickup == t2 else None)
                
                if (pickup and consume and pickup.item and pickup.cell is None and 
                    consume.action in CONSUME_ACTIONS and consume.item == pickup.item):
                    
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
        standalone_minor_tasks: Optional[list[MinorTask]] = None,
        major_tasks: Optional[list[MajorTask]] = None,
        explicit_precedence: Optional[list[tuple[str, str]]] = None,
        explicit_single_worker_groups: Optional[list[list[str]]] = None,
        warehouse_stock: Optional[dict[Item, int]] = None,
        horizon: int = 24,
        clct_deadline: Optional[int] = None,
        worker_pool_size: Optional[int] = None,
    ) -> "Instance":
        """
        Factory method to compile major tasks and minor tasks into a unified Instance.
        Replaces the old `instance_compiler.py`.
        """
        minor_tasks = list(standalone_minor_tasks or [])
        precedence = list(explicit_precedence or [])
        groups = [list(group) for group in (explicit_single_worker_groups or [])]

        seen_ids = {task.id for task in minor_tasks}
        
        for major in major_tasks or []:
            minors, edges, group = expand_major_task(major)
            for minor in minors:
                if minor.id in seen_ids:
                    raise ValueError(f"Duplicate minor_task id: {minor.id!r} (from major_task {major.id!r})")
                seen_ids.add(minor.id)
            minor_tasks.extend(minors)
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