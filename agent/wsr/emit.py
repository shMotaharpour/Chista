"""A searched route -> the ops the engine reads.

The search decides which turn a task occupies and which worker does it. The walk between tasks is
NOT part of that decision, because the search already priced it: the turns between two tasks are
the moves that carry the worker from one to the next, and the search left exactly that many.

So this fills the gaps. For each worker it walks from where the worker stands to the next task's
tile, writes the moves into the turns the search left free, and writes the op into the turn the
search chose. A route whose walk does not fit those turns is refused rather than padded: the
compiler is the last place a disagreement between the schedule and the day can be caught, and
emitting a move over a turn that already holds an op would hand the engine a day the search never
approved.

A good a task consumes is not in the worker's bag yet the first time that worker needs it, so the
trip is written here: out to a door, one PICKUP carrying every one of that good the worker will use
that day, and on to the tile. Which door is not a free choice - it is the nearest one to where the
worker stands, the same rule the search priced with.
"""

from __future__ import annotations

from agent.world.board import manhattan
from agent.world.rules import SHED_ACCESS
from agent.wsr.beam import (Day, Result, _settled_after_first_turn, _start_hours,
                            _start_positions, first_arrival, preload_turns)
from agent.wsr.routing import walk
from agent.wsr.tasks import ITEM_CODE, TaskArray

PASS = ("PASS",)

ITEM_NAME = {code: item.name for item, code in ITEM_CODE.items()}


def compile_route(day: Day, tasks: TaskArray, result: Result, *,
                  horizon: int | None = None, settled=None) -> list[list[tuple]]:
    """A route -> one op list per worker, `horizon` turns long and PASS-padded.

    The list is indexed by turn, so `ops[worker][hour]` is what that worker does at that hour -
    which is the shape the dispatcher slices.

    `settled` is where the units already on the field stand once the first turn is over. The hands
    land on the doors that are free when they are hired, so a unit that walks off its door in turn
    zero moves every hand after it - the search prices that, and this must walk the same cells or
    the day it writes is not the day that was searched.
    """
    horizon = int(horizon if horizon is not None else day.horizon)
    if settled is None:
        settled = _settled_after_first_turn(day, tasks, result)
    starts = _start_positions(day, result.pool, settled)
    hours = _start_hours(day, result.pool)
    m = int(starts.shape[0])

    ops: list[list[tuple]] = [[PASS] * horizon for _ in range(m)]
    at = [(int(cell[0]), int(cell[1])) for cell in starts]
    # The last turn each worker was busy. A worker offered from hour 1 has turn 0 free, so the
    # clock starts one before its first allowed turn.
    last = [int(h) - 1 for h in hours]

    by_worker: dict[int, list[tuple[int, str]]] = {}
    for turn, task_id, worker in result.route:
        by_worker.setdefault(int(worker), []).append((int(turn), task_id))

    # The pickups: one turn each, at the door the worker starts on, carrying the whole day's use of
    # that good. The second feeding of a day buys nothing, so it is not fetched again. Each worker's
    # pickups begin at ITS own hour - a hand offered at hour 1 cannot pick anything up at hour 0,
    # and giving it another worker's hour is what put a walk one turn short of its room.
    arrival = first_arrival(tasks)
    for worker in range(m):
        bag = _bag(tasks, by_worker.get(worker, []))
        first = max(int(hours[worker]), arrival)
        for step, good in enumerate(sorted(bag)):
            ops[worker][first + step] = ("PICKUP", ITEM_NAME[good], bag[good])
        if bag:
            last[worker] = first + len(bag) - 1

    for worker, entries in by_worker.items():
        for turn, task_id in sorted(entries):
            row = tasks.ids.index(task_id)
            target = (int(tasks.cells[row][0]), int(tasks.cells[row][1]))
            moves = walk(at[worker], target)
            first = last[worker] + 1
            _room(task_id, worker, turn, len(moves), first)
            _write(ops[worker], first, moves)
            ops[worker][turn] = tasks.ops[row]
            at[worker] = target
            last[worker] = turn

    return ops


def to_plan(ops: list[list[tuple]], market=None) -> dict:
    """The compiler's output in the shape the dispatcher slices.

    `units[0]` is the farmer and the rest the hands in order, which is the engine's own order.
    """
    return {"units": [[list(op) for op in unit] for unit in ops],
            "market": list(market or [])}


def _bag(tasks: TaskArray, entries: list[tuple[int, str]]) -> dict[int, int]:
    """How much of each good this worker's day uses, so one trip can carry all of it."""
    bag: dict[int, int] = {}
    for _turn, task_id in entries:
        good = int(tasks.items[tasks.ids.index(task_id)])
        if good >= 0:
            bag[good] = bag.get(good, 0) + 1
    return bag


def _write(ops: list[tuple], first: int, moves: list[tuple]) -> None:
    for step, move in enumerate(moves):
        ops[first + step] = move


def _room(task_id: str, worker: int, turn: int, needed: int, first: int) -> None:
    if needed > turn - first:
        raise ValueError(
            f"{task_id} on worker {worker} at turn {turn}: it needs {needed} turns from {first} "
            f"and only {turn - first} are free - the schedule and the day disagree")


def _nearest_door(at: tuple[int, int]) -> tuple[int, int]:
    """The closest shed-access tile, ties broken by the world's own order - as the search priced."""
    return min(((int(x), int(y)) for x, y in SHED_ACCESS),
               key=lambda tile: (manhattan(at, tile),
                                 SHED_ACCESS.index((tile[0], tile[1]))))


def check_route(day: Day, tasks: TaskArray, result: Result, settled=None) -> list[str]:
    """Everything a compiled route promises, checked - so a caller can report instead of hope.

    The engine refuses a bad op in silence, which is why a day is worth verifying rather than
    trusting: this names what is wrong, and an empty list means nothing is.
    """
    complaints: list[str] = []
    if settled is None:
        settled = _settled_after_first_turn(day, tasks, result)
    starts = _start_positions(day, result.pool, settled)
    hours = _start_hours(day, result.pool)

    turns: dict[int, list[tuple[int, str]]] = {}
    for turn, task_id, worker in result.route:
        turns.setdefault(int(worker), []).append((int(turn), task_id))

    when = {task_id: int(turn) for turn, task_id, _w in result.route}
    placed = set(when)
    for worker, entries in turns.items():
        for turn, task_id in entries:
            if not 0 <= turn < day.horizon:
                complaints.append(f"{task_id} at turn {turn} is outside the day")
            if worker >= starts.shape[0]:
                complaints.append(f"{task_id} names worker {worker}, who does not exist")
            elif turn < int(hours[worker]):
                complaints.append(f"{task_id} at turn {turn} is before worker {worker} may begin")
    for i in range(tasks.n):
        row = tasks.ids[i]
        if row not in placed:
            continue
        for j in range(tasks.n):
            if tasks.pred[i, j] and tasks.ids[j] in placed and when[tasks.ids[j]] >= when[row]:
                complaints.append(f"{tasks.ids[j]} must come before {row}")
        if tasks.items[i] >= 0 and when[row] < int(tasks.earliest[i]):
            complaints.append(f"{row} needs its good at {when[row]}, before it is in the shed")
    return complaints
