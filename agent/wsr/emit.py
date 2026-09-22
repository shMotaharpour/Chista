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

from typing import NamedTuple

from agent.wsr.beam import (Day, Result, _bag, _start_hours, _start_positions, day_first_good,
                            door_work, first_walk_turn, good_hours, leg_moves, leg_target, legs,
                            loads_before, pickup_turns, walk_start_turn)
from agent.wsr.tasks import ITEM_CODE, TaskArray

PASS = ("PASS",)


class DayOps(NamedTuple):
    """The compiled day: the ops per worker, and what its drops bank.

    `arrivals` is `(hour, item, units)` per drop. A SELL can only reach the shed, so this is what
    the day's market may actually sell today; what no drop carried waits for tonight.
    """

    units: list[list[tuple]]
    arrivals: tuple[tuple[int, str, int], ...] = ()

ITEM_NAME = {code: item.name for item, code in ITEM_CODE.items()}


def compile_route(day: Day, tasks: TaskArray, result: Result, *,
                  horizon: int | None = None) -> DayOps:
    """A route -> one op list per worker, `horizon` turns long and PASS-padded.

    `drop` writes the trip that carries what the day grew to the shed. The DROP empties the whole
    bag and hands it over; what the shed then keeps is the shed's decision, and none of this layer's
    business - so the arrivals are what the bags held, not what the shed made of them.

    The list is indexed by turn, so `ops[worker][hour]` is what that worker does at that hour -
    which is the shape the dispatcher slices.

    The hands start on `result.doors`, the doors the search priced them on. There is no second way
    to place them: a caller-supplied position was the path that wrote the hands from the farmer's
    start cell and made the planner's day uncompilable (#162).
    """
    horizon = int(horizon if horizon is not None else day.horizon)
    # An explicit `settled` is the caller placing the hands itself; the result's own doors are used
    # when it lets the result decide, which is the only way the day is written as it was priced.
    # wsr owns WHERE a hand lands (F040): it is computed here from the day's own route, never
    # handed in. `doors` still wins inside `_start_positions`, so the searched doors are never
    # smothered - that was #162's bug, and it stays fixed.
    settled = result.settled or _settled_after_first_turn(day, tasks, result)
    starts = _start_positions(day, result.pool, settled, result.doors)
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

    # The pickups: one turn each, at the door the worker stands on, carrying the whole day's use of
    # that good. The second feeding of a day buys nothing, so it is not fetched again. They come
    # before the worker's first task, unless that task is door work in the turns before its first
    # good lands (`loads_before`, the search's own rule); each good is taken no earlier than it is in
    # the shed (`pickup_turns`): a PICKUP before its good has arrived is refused in silence (F047).
    arrival = good_hours(tasks)
    before = door_work(tasks)

    for worker, entries in by_worker.items():
        bag = _bag(tasks, entries)
        loaded = not bag
        first = day_first_good(tasks)
        for turn, row, fetch in legs(tasks, entries):
            if not loaded and loads_before(bool(before[row]), turn, first):
                for pick, good in pickup_turns(last[worker] + 1, bag, arrival):
                    ops[worker][pick] = ("PICKUP", ITEM_NAME[good], bag[good])
                last[worker] = first_walk_turn(last[worker] + 1, bag, arrival) - 1
                loaded = True
            task_id = tasks.ids[row]
            target = leg_target(tasks, row, at[worker])
            # A good used after the worker's own DROP is fetched again on the way: the drop took the
            # load from the door with it (`legs`), and the leg walks through the nearest door.
            moves = leg_moves(at[worker], target, fetch)
            # The walk ENDS at the task's turn, not at the first turn the worker is free: a worker
            # with slack waits where it stands and then walks, so the turns the walk occupies are
            # exactly the ones the model counted when it priced the day. Writing it early instead
            # moved a unit off its door in turn 0 while the model had it standing there - and a unit
            # that leaves its door in the first turn moves where every hand after it lands (F040).
            # That was the rewrite's regression, not the old layer's rule: `walk_start_turn` is the
            # reading the pre-rewrite `plan_day` took from the compiled route.
            _room(task_id, worker, turn, len(moves), last[worker] + 1)
            _write(ops[worker], walk_start_turn(turn, len(moves)), moves)
            ops[worker][turn] = tasks.ops[row]
            at[worker] = target
            last[worker] = turn

    # What each drop banks: the goods the worker's harvests have put in its bag since the drop
    # before it. Read off the route rather than recomputed, because the engine's DROP empties the
    # whole bag - so the batch is a property of the order, not of any one harvest.
    arrivals: list[tuple[int, str, int]] = []
    for worker, entries in by_worker.items():
        bagged: dict[int, int] = {}
        for turn, task_id in sorted(entries):
            row = tasks.ids.index(task_id)
            good = int(tasks.yields[row])
            if good >= 0:
                bagged[good] = bagged.get(good, 0) + int(tasks.yield_n[row])
            elif turn >= 0 and bool(tasks.is_drop[row]):
                for item, units in sorted(bagged.items()):
                    arrivals.append((int(turn), ITEM_NAME[item], units))
                bagged.clear()

    return DayOps(units=ops, arrivals=tuple(arrivals))


def to_plan(day_ops: DayOps, market=None) -> dict:
    """The compiler's output in the shape the dispatcher slices.

    `units[0]` is the farmer and the rest the hands in order, which is the engine's own order.
    """
    return {"units": [[list(op) for op in unit] for unit in day_ops.units],
            "market": list(market or [])}


def _write(ops: list[tuple], first: int, moves: list[tuple]) -> None:
    for step, move in enumerate(moves):
        ops[first + step] = move


def _room(task_id: str, worker: int, turn: int, needed: int, first: int) -> None:
    if needed > turn - first:
        raise ValueError(
            f"{task_id} on worker {worker} at turn {turn}: it needs {needed} turns from {first} "
            f"and only {turn - first} are free - the schedule and the day disagree")


def check_route(day: Day, tasks: TaskArray, result: Result) -> list[str]:
    """Everything a compiled route promises, checked - so a caller can report instead of hope.

    The engine refuses a bad op in silence, which is why a day is worth verifying rather than
    trusting: this names what is wrong, and an empty list means nothing is.
    """
    complaints: list[str] = []
    # wsr owns WHERE a hand lands (F040): it is computed here from the day's own route, never
    # handed in. `doors` still wins inside `_start_positions`, so the searched doors are never
    # smothered - that was #162's bug, and it stays fixed.
    settled = result.settled or _settled_after_first_turn(day, tasks, result)
    starts = _start_positions(day, result.pool, settled, result.doors)
    hours = _start_hours(day, result.pool)

    turns: dict[int, list[tuple[int, str]]] = {}
    for turn, task_id, worker in result.route:
        turns.setdefault(int(worker), []).append((int(turn), task_id))

    # A drop with an empty bag is written at turn -1: it is done, it has no turn, and asking it to
    # precede anything would be asking a turn that does not exist.
    when = {task_id: int(turn) for turn, task_id, _w in result.route if int(turn) >= 0}
    # The tasks the route places, by id: `when` is keyed by task id and `turns` by worker, and the
    # rules below ask by id. Built from the route's turns it was a set of ints, so `row not in placed`
    # was always true and the two checks under it never ran.
    placed = {task_id for _turn, task_id, _w in result.route}
    for worker, entries in turns.items():
        for turn, task_id in entries:
            if turn < 0:
                continue                    # an idle drop: done, with no turn of its own
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
            if not tasks.pred[i, j] or tasks.ids[j] not in when or row not in when:
                continue
            if when[tasks.ids[j]] >= when[row]:
                complaints.append(f"{tasks.ids[j]} must come before {row}")
        if tasks.items[i] >= 0 and when[row] < int(tasks.earliest[i]):
            complaints.append(f"{row} needs its good at {when[row]}, before it is in the shed")

    # A worker tie: the tasks a task must share a worker with. A route that splits a group hands the
    # work to a worker that cannot do it - a drop by anybody but the worker holding the good banks
    # nothing - and the engine would run it without a word.
    if tasks.ties.size:
        who_of = {task_id: int(worker) for _turn, task_id, worker in result.route}
        for i in range(tasks.n):
            row = tasks.ids[i]
            if row not in who_of:
                continue
            for slot in range(tasks.ties.shape[1]):
                mate = int(tasks.ties[i, slot])
                if mate < 0:
                    continue
                other = tasks.ids[mate]
                if other in who_of and who_of[other] != who_of[row]:
                    complaints.append(
                        f"{row} and {other} must be the same worker: "
                        f"{who_of[row]} and {who_of[other]}")
    return complaints
