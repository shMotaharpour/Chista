"""
Core data model and Rich Domain Instance for ChistaWRS.
Combines data structures, major task expansion, and instance compilation into a 
single, validated, and fully pre-processed domain model.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Literal, NamedTuple, Optional

# ============================================================================
# 1. Basic Enums and Data Structures
# ============================================================================

class Cell(NamedTuple):
    x: int
    y: int

class CellType(str, Enum):
    NONE = "None"
    PLANT = "Plant"
    WEED = "Weed"
    PASTURE = "Pasture"
    COOP = "Coop"

class Item(str, Enum):
    WHEAT = "wheat"
    FERTILIZER = "fertilizer"
    COW = "cow"
    SHEEP = "sheep"
    GOOSE = "goose"
    MELON = "melon"
    TOMATO = "tomato"
    STRAWBERRY = "strawberry"
    CARROT = "carrot"
    MILK = "milk"
    WOOL = "wool"
    EGG = "egg"

ANIMAL_ITEMS = frozenset({Item.COW, Item.SHEEP, Item.GOOSE})
CROP_ITEMS = frozenset({Item.MELON, Item.TOMATO, Item.STRAWBERRY, Item.CARROT, Item.WHEAT})
PRODUCT_ITEMS = frozenset({Item.MILK, Item.WOOL, Item.EGG})

ANIMAL_STRUCTURE: dict[Item, CellType] = {
    Item.COW: CellType.PASTURE,
    Item.SHEEP: CellType.PASTURE,
    Item.GOOSE: CellType.COOP,
}

class MinorActionType(str, Enum):
    NORTH = "NORTH"
    SOUTH = "SOUTH"
    EAST = "EAST"
    WEST = "WEST"
    PASS = "PASS"
    PICKUP = "PICKUP"
    PLACE = "PLACE"
    DROP = "DROP"
    PLANT = "PLANT"
    WATER = "WATER"
    HARVEST = "HARVEST"
    FERTILIZE = "FERTILIZE"
    BUILD_COOP = "BUILD_COOP"
    BUILD_PASTURE = "BUILD_PASTURE"
    FEED = "FEED"
    COLLECT_FERTILIZER = "COLLECT_FERTILIZER"
    CARE = "CARE"
    DIG = "DIG"


MOVEMENT_ACTIONS = frozenset(
    {MinorActionType.NORTH, MinorActionType.SOUTH, MinorActionType.EAST, MinorActionType.WEST, MinorActionType.PASS}
)
PRODUCE_ACTIONS = frozenset({MinorActionType.PICKUP, MinorActionType.HARVEST, MinorActionType.COLLECT_FERTILIZER})
CONSUME_ACTIONS = frozenset({MinorActionType.PLACE, MinorActionType.FEED, MinorActionType.FERTILIZE})

WAREHOUSE_ENTRY_ORDER: tuple[str, ...] = ("NW", "NE", "SW", "SE")
WAREHOUSE_ENTRY_CELLS: dict[str, Cell] = {"NW": Cell(4, 4), "NE": Cell(5, 4), "SW": Cell(4, 5), "SE": Cell(5, 5)}


@dataclass
class CellState:
    type: CellType = CellType.NONE
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
    action: MinorActionType
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

MajorTaskType = Literal["feed", "frtz", "wet_harvst", "plnt", "wet_harvst_plnt", "frtz_water", "place_animal"]
_SINGLE_WORKER_TYPES = frozenset({"feed", "frtz", "frtz_water", "place_animal"})

@dataclass
class MajorTask:
    id: str
    type: MajorTaskType
    cell: Cell
    crop: Optional[Item] = None
    harvested_item: Optional[Item] = None
    harvested_qty: int = 1
    item: Optional[Item] = None

ExpansionResult = tuple[list[MinorTask], list[tuple[str, str]], list[str]]

def _acquire_minor(task_id: str, item: Item) -> MinorTask:
    return MinorTask(id=task_id, cell=None, action=MinorActionType.PICKUP, item=item, qty=1)

def expand_major_task(major: MajorTask) -> ExpansionResult:
    tid = major.id
    if major.type == "wet_harvst":
        water = MinorTask(id=f"{tid}_water", cell=major.cell, action=MinorActionType.WATER)
        harvest = MinorTask(id=f"{tid}_harvest", cell=major.cell, action=MinorActionType.HARVEST, item=major.harvested_item, qty=major.harvested_qty)
        return [water, harvest], [(water.id, harvest.id)], []
    
    if major.type == "plnt":
        plant = MinorTask(id=f"{tid}_plant", cell=major.cell, action=MinorActionType.PLANT, crop=major.crop)
        water = MinorTask(id=f"{tid}_water", cell=major.cell, action=MinorActionType.WATER)
        return [plant, water], [(plant.id, water.id)], []
    
    if major.type == "wet_harvst_plnt":
        water1 = MinorTask(id=f"{tid}_water1", cell=major.cell, action=MinorActionType.WATER)
        harvest = MinorTask(id=f"{tid}_harvest", cell=major.cell, action=MinorActionType.HARVEST, item=major.harvested_item, qty=major.harvested_qty)
        plant = MinorTask(id=f"{tid}_plant", cell=major.cell, action=MinorActionType.PLANT, crop=major.crop)
        water2 = MinorTask(id=f"{tid}_water2", cell=major.cell, action=MinorActionType.WATER)
        return [water1, harvest, plant, water2], [(water1.id, harvest.id), (harvest.id, plant.id), (plant.id, water2.id)], []
    
    if major.type == "feed":
        acquire = _acquire_minor(f"{tid}_acquire", Item.WHEAT)
        feed = MinorTask(id=f"{tid}_feed", cell=major.cell, action=MinorActionType.FEED, item=Item.WHEAT)
        return [acquire, feed], [(acquire.id, feed.id)], [acquire.id, feed.id]
    
    if major.type == "frtz":
        acquire = _acquire_minor(f"{tid}_acquire", Item.FERTILIZER)
        frtz = MinorTask(id=f"{tid}_fertilize", cell=major.cell, action=MinorActionType.FERTILIZE, item=Item.FERTILIZER)
        return [acquire, frtz], [(acquire.id, frtz.id)], [acquire.id, frtz.id]
    
    if major.type == "frtz_water":
        acquire = _acquire_minor(f"{tid}_acquire", Item.FERTILIZER)
        frtz = MinorTask(id=f"{tid}_fertilize", cell=major.cell, action=MinorActionType.FERTILIZE, item=Item.FERTILIZER)
        water = MinorTask(id=f"{tid}_water", cell=major.cell, action=MinorActionType.WATER)
        return [acquire, frtz, water], [(acquire.id, frtz.id), (frtz.id, water.id)], [acquire.id, frtz.id]
    
    if major.type == "place_animal":
        if major.item is None:
            raise ValueError(f"'place_animal' major_task {tid!r} requires an item")
        pickup = MinorTask(id=f"{tid}_pickup", cell=None, action=MinorActionType.PICKUP, item=major.item, qty=1)
        place = MinorTask(id=f"{tid}_place", cell=major.cell, action=MinorActionType.PLACE, item=major.item, qty=1)
        return [pickup, place], [(pickup.id, place.id)], [pickup.id, place.id]
    
    raise ValueError(f"unknown major_task type: {major.type!r}")


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
                pickup = t1 if t1.action == MinorActionType.PICKUP else (t2 if t2.action == MinorActionType.PICKUP else None)
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