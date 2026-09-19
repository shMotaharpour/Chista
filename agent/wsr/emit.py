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

A fetch is a detour through a door: the worker walks to a door, picks the good up there, and walks
on. Which door is not a free choice - it is the nearest one to where the worker stands, the same
rule the search priced with.
"""

from __future__ import annotations

from agent.world.board import manhattan
from agent.world.rules import SHED_ACCESS
from agent.wsr.beam import Day, Result, _fetch_mask, _start_hours, _start_positions
from agent.wsr.routing import walk
from agent.wsr.tasks import TaskArray

PASS = ("PASS",)


def compile_route(day: Day, tasks: TaskArray, result: Result, *,
                  horizon: int | None = None) -> list[list[tuple]]:
    """A route -> one op list per worker, `horizon` turns long and PASS-padded.

    The list is indexed by turn, so `ops[worker][hour]` is what that worker does at that hour -
    which is the shape the dispatcher slices.
    """
    horizon = int(horizon if horizon is not None else day.horizon)
    starts = _start_positions(day, result.pool)
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

    for worker, entries in by_worker.items():
        for turn, task_id in sorted(entries):
            row = tasks.ids.index(task_id)
            target = _target(tasks, row, at[worker])
            moves = walk(at[worker], target)
            first = last[worker] + 1
            free_turns = turn - first
            if len(moves) > free_turns:
                raise ValueError(
                    f"{task_id} on worker {worker} at turn {turn}: the walk from {at[worker]} to "
                    f"{target} takes {len(moves)} turns and only {free_turns} are free - the "
                    f"schedule and the day disagree")
            for step, move in enumerate(moves):
                ops[worker][first + step] = move
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


def _target(tasks: TaskArray, row: int, at: tuple[int, int]) -> tuple[int, int]:
    """Where a worker must stand to do a task: a door for a fetch, the tile for anything else."""
    if bool(_fetch_mask(tasks)[row]):
        return _nearest_door(at)
    return (int(tasks.cells[row][0]), int(tasks.cells[row][1]))


def _nearest_door(at: tuple[int, int]) -> tuple[int, int]:
    """The closest shed-access tile, ties broken by the world's own order - as the search priced."""
    return min(((int(x), int(y)) for x, y in SHED_ACCESS),
               key=lambda tile: (manhattan(at, tile),
                                 SHED_ACCESS.index((tile[0], tile[1]))))


def check_route(day: Day, tasks: TaskArray, result: Result) -> list[str]:
    """Everything a compiled route promises, checked - so a caller can report instead of hope.

    The engine refuses a bad op in silence, which is why a day is worth verifying rather than
    trusting: this names what is wrong, and an empty list means nothing is.
    """
    complaints: list[str] = []
    starts = _start_positions(day, result.pool)
    hours = _start_hours(day, result.pool)
    fetch = _fetch_mask(tasks)

    turns: dict[int, list[tuple[int, str]]] = {}
    for turn, task_id, worker in result.route:
        turns.setdefault(int(worker), []).append((int(turn), task_id))

    for worker, entries in turns.items():
        for turn, task_id in entries:
            if not 0 <= turn < day.horizon:
                complaints.append(f"{task_id} at turn {turn} is outside the day")
            if worker >= starts.shape[0]:
                complaints.append(f"{task_id} names worker {worker}, who does not exist")
            if turn < int(hours[worker]):
                complaints.append(f"{task_id} at turn {turn} is before worker {worker} may begin")
    for i in range(tasks.n):
        row = tasks.ids[i]
        placed = {t for _h, t, _w in result.route}
        if row not in placed:
            continue
        for j in range(tasks.n):
            if not tasks.pred[i, j] or tasks.ids[j] not in placed:
                continue
            when = {t: h for h, t, _w in result.route}
            if when[tasks.ids[j]] >= when[row]:
                complaints.append(f"{tasks.ids[j]} must come before {row}")
        if fetch[i] and tasks.items[i] >= 0:
            when = {t: h for h, t, _w in result.route}
            if when[row] < int(tasks.earliest[i]):
                complaints.append(f"{row} fetches at {when[row]}, before its good is in the shed")
    return complaints
