"""The day compiler: chains and an offer in, routes out. The market is not here.

The tile graph prices a chain as if the worker already stood on the tile with its inputs in its
bag. The engine does not: it refuses an unmet precondition in silence, which looks like work in
the log. This module builds the day that satisfies the preconditions - the walk, the shed trips,
the drop - and names whatever it could not carry rather than emitting an op the engine would
ignore.

What it needs from the world outside is an `Offer`, and the caller (the planner) owns it:

* **when a hand may start** - hiring is the planner's decision and it knows what the purse and
  the market allow, so it says which hours a new hand can begin working;
* **when an item is in the shed** - a fetch can only happen once the good is there, and the
  planner knows when its orders land. An op whose good never arrives is reported, not guessed at.

Two answers are available: `compile_day` gives the routes, and `feasibility` gives the fewest
hands the day fits in, which is what a planner needs before it decides to hire anyone.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

from agent.tile_dp.chains import actions_of
from agent.world.action_rules import CARRIES, YIELDS
from agent.world.board import MOVE_DELTA, manhattan
from agent.world.model import UnitAction
from agent.world.rules import BOARD_SIZE, SHED_ACCESS, TURNS_PER_DAY, spawn_cell
from agent.wsr.models import DayOnCell, Instance, Worker
from agent.wsr.oxa_solver import OxaConfig, solve_oxa

Cell = tuple[int, int]

#: The largest pool a day may be offered: the engine's own hire ladder runs out of names past
#: this, and no day's chains need more hands than it has tiles.
MAX_HANDS = 8


@dataclass(frozen=True)
class Offer:
    """What the planner can supply today.

    `hire_times` is the hour each hand it is willing to pay for may begin; a hand is offered at
    most one start time, and the engine's own rule is that a hand hired in turn 0 acts from hour
    1. `available` is the hour each item is in the shed - a fetch cannot happen before it.
    """

    hire_times: tuple[int, ...] = ()
    available: Mapping[str, int] = field(default_factory=dict)

    @property
    def hands(self) -> int:
        return len(self.hire_times)


def shed_access(board: int = BOARD_SIZE) -> tuple[Cell, ...]:
    """The tiles from which PICKUP and DROP work, from the world's own definition."""
    return tuple((int(x), int(y)) for x, y in SHED_ACCESS)


def walk(start: Cell, goal: Cell, board: int = BOARD_SIZE) -> list[tuple[str, ...]]:
    """The move ops that carry a unit from `start` to `goal`, x first then y.

    Shortest by construction and always legal: the engine only refuses a move that leaves the
    board. Ties break x-then-y, so a plan is a function of its inputs rather than of a hash order.
    """
    x, y = int(start[0]), int(start[1])
    gx, gy = int(goal[0]), int(goal[1])
    out: list[tuple[str, ...]] = []
    while x != gx:
        step = "EAST" if gx > x else "WEST"
        x += MOVE_DELTA[step][0]
        out.append((step,))
    while y != gy:
        step = "SOUTH" if gy > y else "NORTH"
        y += MOVE_DELTA[step][1]
        out.append((step,))
    if not (0 <= x < board and 0 <= y < board):
        raise ValueError(f"walk({start}, {goal}) leaves the board")
    return out


def nearest_shed(pos: Cell, board: int = BOARD_SIZE) -> Cell:
    """The closest shed-access tile, ties broken by the world's own order."""
    tiles = shed_access(board)
    return min(tiles, key=lambda t: (manhattan(pos, t), tiles.index(t)))


def op_name(action) -> str:
    """The engine's spelling of an op, for one that arrived from the chain registry."""
    return str(getattr(action.op, "value", action.op))


def carried_item(action, entity: str | None) -> str | None:
    """What this op needs in the unit's bag, or None if it needs nothing."""
    name = op_name(action)
    if name == "PLACE":
        return entity
    return CARRIES.get(name)


@dataclass(frozen=True)
class UnitRoute:
    """One unit's whole day, plus what it cost the plan to build it."""

    unit: int
    ops: tuple[tuple[str, ...], ...]     # TURNS_PER_DAY entries, PASS-padded
    missing: tuple[str, ...]             # items the day never got
    dropped: tuple[str, ...]             # ops the day could not carry
    hours_used: int
    drop_hour: int | None
    walked: int                          # move ops spent
    #: (hour, item, units) the day's DROP puts in the shed: what the market may sell.
    arrivals: tuple[tuple[int, str, int], ...] = ()

    @property
    def worked(self) -> bool:
        """Whether this unit does anything at all."""
        return any(op and op[0] != "PASS" for op in self.ops)


def route_unit(ops: Sequence[str], entity: str | None, pos: Cell, *,
               unit: int = 0, hour: int = 0, hours: int = TURNS_PER_DAY,
               available: Mapping[str, int] | None = None,
               times: Sequence[int] | None = None,
               fetch_times: Mapping[str, int] | None = None,
               target: Cell | None = None,
               carried: Mapping[str, int] | None = None,
               harvest_yields: Mapping[str, int] | None = None,
               drop: bool = True, board: int = BOARD_SIZE) -> UnitRoute:
    """Compile one chain into one unit's day.

    `available` is the hour each item reaches the shed (the planner's table): a fetch is written
    at that hour and nowhere earlier, because the engine refuses a PICKUP of nothing. `times` are
    the hours the scheduler gave each op, and `fetch_times` the hours it gave each fetch; when
    they are given the compiler writes them instead of deriving travel of its own, which is what
    kept the schedule and the day from disagreeing.

    Ops whose preconditions cannot be met inside `hours` are dropped and named, never emitted.
    """
    target = (int((target or pos)[0]), int((target or pos)[1]))
    bag = {k: int(v) for k, v in (carried or {}).items()}
    yields = dict(harvest_yields or {})
    arrives_at = {str(k): int(v) for k, v in (available or {}).items()}
    bagged: dict[str, int] = {}
    bagged_ops = 0
    turns = actions_of(tuple(ops), entity)
    seq: list[tuple[str, ...]] = []
    missing: list[str] = []
    dropped: list[str] = []
    at = (int(pos[0]), int(pos[1]))
    walked = 0
    drop_hour: int | None = None

    def free() -> int:
        """Turns left, counting from `hour`."""
        return hours - (hour + len(seq))

    def at_hour(want: int) -> None:
        """Pad with PASS so the next op lands on the hour it was given."""
        while hour + len(seq) < want:
            seq.append(("PASS",))

    def push(steps: Iterable[tuple[str, ...]]) -> None:
        nonlocal walked
        for step in steps:
            if step and step[0] in MOVE_DELTA:
                walked += 1
            seq.append(step)

    for op_index, action in enumerate(turns):
        name = op_name(action)
        item = carried_item(action, entity)
        if item is not None and bag.get(item, 0) <= 0:
            if item not in arrives_at:
                missing.append(item)                 # the planner never said it arrives
                dropped.append(name)
                continue
            fetch_hour = int(arrives_at[item])
            if fetch_hour > hours - 1:
                missing.append(item)
                dropped.append(name)
                continue
            if fetch_times and str(item) in fetch_times:
                # The scheduler placed this fetch, at a door and at its own turn.
                at_hour(int(fetch_times[str(item)]))
            else:
                # No schedule given: the fetch is the trip, and it must fit.
                shed = nearest_shed(at, board)
                back = walk(shed, target, board)
                trip = walk(at, shed, board) + [("PICKUP", item, 1)] + back
                if free() < len(trip) + 1:
                    dropped.append(name)
                    continue
                at_hour(max(int(hour), fetch_hour))
                push(trip)
            if not seq or seq[-1][:1] != ("PICKUP",):
                push([("PICKUP", item, 1)])
            bag[item] = 1
            at = target
        elif at != target:
            trip = walk(at, target, board)
            if free() < len(trip) + 1:
                dropped.append(name)
                continue
            push(trip)
            at = target
        if free() < 1:
            dropped.append(name)
            continue
        if times is not None and op_index < len(times):
            at_hour(int(times[op_index]))
        push([tuple(action.as_list())])
        if name == "HARVEST":
            bagged_ops += 1
            item = action.item.value if action.item is not None else (entity or "")
            bagged[item] = bagged.get(item, 0) + int(yields.get(item, 0))
        elif name == "COLLECT_FERTILIZER":
            bagged_ops += 1
            bagged[YIELDS["COLLECT_FERTILIZER"]] = bagged.get(YIELDS["COLLECT_FERTILIZER"], 0) + 1

    arrivals: list[tuple[int, str, int]] = []
    if drop and bagged_ops > 0:
        shed = nearest_shed(at, board)
        trip = walk(at, shed, board)
        if free() >= len(trip) + 1:
            push(trip)
            push([("DROP",)])
            at = shed
            drop_hour = hour + len(seq) - 1
            arrivals = [(drop_hour, item, units)
                        for item, units in sorted(bagged.items()) if units > 0]

    head = (("PASS",),) * max(0, int(hour))
    padded = head + tuple(seq)
    padded = padded + (("PASS",),) * max(0, hours - len(padded))
    route = UnitRoute(unit=unit, ops=padded[:hours], missing=tuple(missing),
                      dropped=tuple(dropped), hours_used=len(seq),
                      drop_hour=drop_hour, walked=walked,
                      arrivals=tuple(arrivals))
    _assert_shed_ops_are_reachable(route, pos, 0, board)
    return route


def _assert_shed_ops_are_reachable(route: UnitRoute, start: Cell, hour: int,
                                   board: int) -> None:
    """Every PICKUP/DROP the route emits must be on a shed-access tile.

    The engine refuses both ops anywhere else in silence, so a route that emits one is a plan
    that lies about its day. Simulating the walk is cheap and it is the only way to know where
    the unit actually stands at that turn.
    """
    tiles = set(shed_access(board))
    pos = (int(start[0]), int(start[1]))
    for index, op in enumerate(route.ops):
        if not op:
            continue
        if op[0] in MOVE_DELTA:
            dx, dy = MOVE_DELTA[op[0]]
            pos = (pos[0] + dx, pos[1] + dy)
        elif op[0] in ("PICKUP", "DROP") and pos not in tiles:
            raise AssertionError(
                f"unit {route.unit} would {op[0]} at {pos} on turn {hour + index}: "
                "not a shed-access tile, so the engine refuses it in silence")


@dataclass(frozen=True)
class DayPlan:
    """A whole farm-day: what each unit does, and what the day could not get."""

    units: tuple[tuple[tuple[str, ...], ...], ...]
    routes: tuple[UnitRoute, ...]
    hands: int                                   # hired hands the day was built with
    idle_units: int
    assignments: tuple[Cell | None, ...]         # unit -> the tile it works
    missing: tuple[str, ...]                     # items the day never got

    @property
    def worked(self) -> int:
        """How many units do anything."""
        return sum(1 for route in self.routes if route.worked)


def _schedule(tiles, units: Sequence[Cell], hands: int, *, offer: Offer,
              hours: int) -> tuple[dict, dict]:
    """Who works what, from the scheduler. Returns (per worker, per worker's fetches)."""
    days = [DayOnCell(chain, cell, entity, entity) for cell, chain, entity in tiles]
    workers = [Worker(index=index, earliest_start=0) for index in range(len(units))]
    for index in range(hands):
        start = int(offer.hire_times[index]) if index < len(offer.hire_times) else 1
        workers.append(Worker(index=len(units) + index, earliest_start=max(1, start)))
    instance = Instance.compile(workers=workers, days=days)
    solved = solve_oxa(instance, OxaConfig(min_workers=1))
    if solved.solution is None:
        return {}, {}
    per_worker: dict[int, dict[int, list[int]]] = {}
    fetches: dict[int, dict[int, dict[str, int]]] = {}
    for route in solved.solution.routes:
        for task in route.tasks:
            column = int(task.task_id.split("_", 1)[0][1:])
            per_worker.setdefault(route.worker_index, {}).setdefault(column, []).append(
                int(task.exec_time))
            minor = instance.tasks_by_id.get(task.task_id)
            if minor is not None and minor.action == UnitAction.PICKUP and minor.item:
                fetches.setdefault(route.worker_index, {}).setdefault(
                    column, {})[str(minor.item)] = int(task.exec_time)
    return per_worker, fetches


def _positions(units: Sequence[Cell], hands: int, offer: Offer) -> list[Cell]:
    """The units that exist, then the ones a hire would add at the engine's own spawn rule."""
    out = [(int(p[0]), int(p[1])) for p in units]
    occupied = list(out)
    for _ in range(int(hands)):
        out.append(spawn_cell(occupied))
        occupied.append(out[-1])
    return out


def compile_day(tiles: Sequence[tuple[Cell, Sequence[str], str | None]],
                units: Sequence[Cell], *, offer: Offer = Offer(), hands: int = 0,
                bags: Sequence[Mapping[str, int]] = (),
                yields: Sequence[Mapping[str, int]] | None = None,
                board: int = BOARD_SIZE, hours: int = TURNS_PER_DAY) -> DayPlan:
    """The whole farm's day: assign the chains to the units and compile their routes.

    `tiles` is what the contractor priced - `(cell, chain ops, entity)` per column, in the
    caller's order. `units` are the positions that exist now, farmer first. `hands` are the ones
    the caller is willing to pay for, placed by the engine's own spawn rule so their routes start
    where they will really stand, and `offer.hire_times` says when each may begin.
    """
    positions = _positions(units, hands, offer)
    schedule, fetches = _schedule(tiles, units, hands, offer=offer, hours=hours)

    routes: list[UnitRoute] = []
    for index, per_column in sorted(schedule.items()):
        for column, op_hours in per_column.items():
            cell, chain, entity = tiles[column]
            routes.append(route_unit(
                chain, entity, positions[index], unit=index,
                hour=0 if index < len(units) else max(1, int(offer.hire_times[
                    index - len(units)]) if index - len(units) < len(offer.hire_times) else 1),
                target=cell, times=op_hours, fetch_times=fetches.get(index, {}).get(column, {}),
                available=offer.available,
                carried=bags[index] if index < len(bags) else None,
                harvest_yields=(yields[column] if yields and column < len(yields) else None),
                board=board, hours=hours))

    assigned: list[Cell | None] = [None] * len(positions)
    for route in routes:
        if route.worked:
            column = next(iter(schedule[route.unit]))
            assigned[route.unit] = tiles[column][0]

    by_unit: list[tuple[tuple[str, ...], ...]] = [((("PASS",),) * hours) for _ in positions]
    for route in routes:
        by_unit[route.unit] = route.ops

    return DayPlan(
        units=tuple(by_unit), routes=tuple(routes), hands=int(hands),
        idle_units=sum(1 for cell in assigned if cell is None),
        assignments=tuple(assigned),
        missing=tuple(item for route in routes for item in route.missing))


def feasibility(tiles: Sequence[tuple[Cell, Sequence[str], str | None]],
                units: Sequence[Cell], *, offer: Offer = Offer(),
                board: int = BOARD_SIZE, hours: int = TURNS_PER_DAY) -> int:
    """The fewest hands that let every chain run.

    What a planner needs before it hires: the same day, asked for with growing pools until the
    scheduler places all the work. `MAX_HANDS` is the ceiling, and reaching it means the day does
    not fit even then - the caller learns that from the number, not from a silent shortfall.
    """
    for hands in range(0, MAX_HANDS + 1):
        plan = compile_day(tiles, units, offer=offer, hands=hands, board=board, hours=hours)
        if plan.idle_units == 0 and not plan.missing:
            return hands
    return MAX_HANDS
