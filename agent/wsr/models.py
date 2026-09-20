"""A chain of ops, expanded into the tasks the search schedules.

A chain from the DP's registry becomes the minor tasks that make it up, the precedence between
them, and the pairs one worker must do together. A carried op gets its own PICKUP, and that pickup
belongs to the same worker as the op that consumes it - the good is in one worker's bag and nowhere
else.

The chain's own order is a constraint only where it changes the outcome: a ONE-SHOT crop is watered
before it is harvested, and the engine does not care in what order an ongoing crop is fed, watered
and harvested. Imposing the order everywhere would forbid days that are fine.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple, Optional, Sequence

from agent.world.action import Action, Item, item_of
from agent.world.action_rules import CARRIES, YIELDS
from agent.world.model import MOVES, PRODUCTS, UnitAction
from agent.world.rules import ANIMAL_STRUCTURE, CROP_RULES

#: Derived views of the one vocabulary (ARCHITECTURE §5 step 4). PASS counts as a movement here
#: because it is what a unit does instead of moving.
MOVEMENT_ACTIONS: frozenset[str] = frozenset(MOVES) | {UnitAction.PASS}
PRODUCT_ITEMS: frozenset[str] = frozenset(PRODUCTS)


class Cell(NamedTuple):
    x: int
    y: int


@dataclass
class MinorTask:
    id: str
    cell: Optional[Cell]
    action: Action
    item: Optional[Item] = None
    n: int = 1
    crop: Optional[Item] = None


#: Op pairs whose order decides the OUTCOME, so the day must keep it whatever the entity. The tile
#: must be bare before anything is planted or built on it, a structure must exist before an animal
#: goes in, an animal before it is fed, and a tile is free again only after its crop is harvested.
ORDER_MATTERS: tuple[tuple[str, str], ...] = (
    ("DIG", "PLANT"),
    ("DIG", "BUILD_COOP"),
    ("DIG", "BUILD_PASTURE"),
    ("BUILD_COOP", "PLACE"),
    ("BUILD_PASTURE", "PLACE"),
    ("PLACE", "FEED"),
    ("PLACE", "CARE"),          # care is about an animal that is on the tile, not in a bag
    ("HARVEST", "PLANT"),
    ("HARVEST", "DIG"),
    ("PLANT", "WATER"),
)

#: The same, but only where a crop yields once: its dose only counts inside a window and its
#: harvest is only worth what the watering before it made. For an ongoing crop and for an animal
#: these pairs commute, and imposing them would forbid legal days. `PLANT -> WATER` is NOT here: a
#: planting day is watered after the planting whatever the crop.
ORDER_MATTERS_ONE_SHOT: tuple[tuple[str, str], ...] = (
    ("FERTILIZE", "WATER"),
    ("WATER", "HARVEST"),
)


class Expansion(NamedTuple):
    """One chain, expanded into the tasks the search reads.

    `tasks` is every op of the chain plus the PICKUP each carried op needs. `order` is the
    (before, after) pairs the engine enforces.

    `groups` is one group per fetch: the PICKUP and the op that consumes it must be the same
    worker's, because the good is in that worker's bag and nowhere else. An op that fetches
    nothing belongs to no group - its place in the day is the order, not a worker tie.
    """
    tasks: list[MinorTask]
    order: list[tuple[str, str]]
    groups: list[list[str]]


def expand_chain(ops: Sequence[str], entity: Item | None = None, cell: Cell | None = None,
                 item: Item | None = None, harvested_n: int = 1,
                 prefix: str = "") -> Expansion:
    """One day's chain -> the tasks the search reads, the order between them, and the worker ties.

    Every op that eats a carried good gets its own PICKUP, and the pickup must come before the
    op that uses it - that pair is always a constraint.

    The chain's own order is NOT: it was fixed in the builder for one tile, half to keep a chain
    from being generated twice and half because the acts are logical in that order. Across cells
    only the pairs whose OUTCOME depends on the order survive, and those are the two tables
    above - the structural ones always, and a one-shot crop's dose-water-harvest only there.
    """
    one_shot = entity in CROP_RULES and not CROP_RULES[entity]["ongoing"]
    pairs = ORDER_MATTERS + (ORDER_MATTERS_ONE_SHOT if one_shot else ())

    tasks: list[MinorTask] = []
    precedence: list[tuple[str, str]] = []
    groups: list[list[str]] = []
    seen: dict[str, int] = {}
    op_ids: list[tuple[str, str]] = []

    def unique(name: str) -> str:
        seen[name] = seen.get(name, 0) + 1
        suffix = "" if seen[name] == 1 else str(seen[name])
        return f"{prefix}{name}{suffix}"

    #: What the chain itself has put in the worker's bag so far. A good that is already there needs
    #: no trip: `HARVEST` then `FEED` is a worker eating what it just picked, not a walk to the shed
    #: and back - and the archive says that is how the game's units worked, six to ten PICKUPs a day
    #: against fourteen or fifteen COLLECT_FERTILIZER and eleven to twenty-six HARVEST.
    bag: set[Item] = set()

    for op in ops:
        name = f"BUILD_{ANIMAL_STRUCTURE[item]}" if op == "BUILD" else op
        carried = CARRIES.get(name)
        if carried is not None:
            carried = item_of(carried)
        elif name == "PLACE":
            carried = item

        pickup: Optional[str] = None
        if carried is not None and carried not in bag:
            acquire = MinorTask(id=unique("acquire"), cell=None,
                                action=UnitAction.PICKUP, item=carried, n=1)
            tasks.append(acquire)
            pickup = acquire.id
            bag.add(carried)

        task = MinorTask(
            id=unique(name.lower()),
            cell=cell,
            action=UnitAction(name),
            item=entity if name == "HARVEST" else carried,
            n=harvested_n if name == "HARVEST" else 1,
            crop=entity if name == "PLANT" else None)
        tasks.append(task)
        # And this op may have put something in the bag for the ops after it.
        if name == "HARVEST" and entity is not None:
            bag.add(entity)
        elif YIELDS.get(name) not in (None, "the tile's yield_units", "the shed"):
            bag.add(item_of(YIELDS[name]))
        if pickup is not None:
            precedence.append((pickup, task.id))    # carry it before you use it
            groups.append([pickup, task.id])        # and the same worker carries it
        op_ids.append((name, task.id))

    for before, after in pairs:
        for index, (name, task_id) in enumerate(op_ids):
            if name != before:
                continue
            for later_name, later_id in op_ids[index + 1:]:
                if later_name == after:
                    if (task_id, later_id) not in precedence:
                        precedence.append((task_id, later_id))
                    break

    return Expansion(tasks=tasks, order=precedence, groups=groups)
