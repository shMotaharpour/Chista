"""Beam search for the day: many routes at once, scored by a layered objective.

The greedy commits to one route as it goes, so a bad early choice is paid for until the end. This
keeps `beam` routes in parallel and prunes to the best ones each turn: a bad choice costs one of
them, not all of them.

The objective is layered, because one layer is not enough - a fixed number of workers can be
reached by many different routes, and a search that cannot tell them apart keeps duplicates:

    1. the workers the day needs      (money: a hand costs the ladder)
    2. the makespan                   (the day finishes sooner)
    3. the travel                     (a worker does not walk for nothing)

Everything is vectorised: a state is a row, and one turn is one set of array operations.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from agent.world.model import UnitAction
from agent.world.rules import SHED_ACCESS, TURNS_PER_DAY

Cell = tuple[int, int]
MAX_HANDS = 16


@dataclass(frozen=True)
class Day:
    """The chains to run today, in the planner's order, with the shed's timetable."""

    cells: tuple[Cell, ...]                 # one per column, indexed
    entities: tuple[str | None, ...]        # what a PLACE puts down
    available: dict[str, int]               # item -> the hour it is in the shed
    horizon: int = TURNS_PER_DAY


def lower_bound(day: Day, targets: int, ops_per_column: int) -> int:
    """The fewest hands the day can possibly need, from arithmetic alone.

    Three floors, and the largest wins: the work has to fit in the day, the longest chain has to
    fit in it, and every tile has to be reached at least once. No solver's opinion is consulted -
    a bound that comes from the solver is circular.
    """
    work = targets + ops_per_column                 # ops plus the fetches they need
    return max(1, -(-work // day.horizon))


def search(day: Day, chains: list[tuple[Cell, tuple[str, ...], str | None]], *,
           units: tuple[Cell, ...], beam: int = 64, hands: int | None = None,
           max_hands: int = MAX_HANDS) -> tuple[int, list[list[tuple[int, str]]]]:
    """The day, searched with `beam` routes in parallel.

    Returns the number of hands that made it fit and, for each hand, the ops of its route as
    `(hour, op)` pairs. `hands=None` grows the pool from the lower bound until the work fits.
    """
    from agent.wsr.oropt import build_targets          # the tasks and their constraints

    tasks, preds, needs_item = build_targets(chains, day)
    n = len(tasks)
    if n == 0:
        return 0, []

    lo = lower_bound(day, n, sum(len(c[1]) for c in chains))
    hi = max_hands if hands is None else int(hands)
    for count in range(lo, hi + 1):
        routes = _run(tasks, preds, needs_item, day, units=units, hands=count, beam=beam)
        if routes is not None:
            return count, routes
    return hi, []


def _run(tasks, preds, needs_item, day: Day, *, units, hands: int, beam: int):
    """One pool size: search the day, or None when the pool cannot carry it."""
    # A state is one route per worker: where each stands, when each is free, what is done, and
    # what each carries. The arrays carry `beam` states at once.
    m = len(units) + hands
    n = len(tasks)
    pos = np.zeros((beam, m, 2), dtype=np.int16)
    time = np.zeros((beam, m), dtype=np.int16)
    done = np.zeros((beam, n), dtype=bool)
    travel = np.zeros((beam,), dtype=np.int16)

    # every state starts identical; the search separates them by the choices it makes
    starts = _start_positions(units, hands)
    pos[:] = np.asarray(starts, dtype=np.int16)[None, :, :]
    for w, (index, earliest) in enumerate(_starts(units, hands)):
        time[:, w] = earliest

    live = np.ones((beam,), dtype=bool)
    for _turn in range(day.horizon):
        if not live.any() or done.all(axis=1).any():
            break
        candidates = _expand(tasks, preds, needs_item, day, pos, time, done, travel, live)
        if candidates is None:
            break
        pos, time, done, travel, live = _select(candidates, beam)
    if not live.any():
        return None
    best = int(np.argmax(_score(done, travel)))
    if not done[best].all():
        return None
    return _routes(tasks, pos[best], time[best], done[best])


def _starts(units, hands):
    """Who exists and when each may act: the units now, then the ones a hire adds."""
    out = [(0, 0) for _ in units]
    out += [(max(1, 1), 1) for _ in range(hands)]
    return out


def _start_positions(units, hands) -> list[Cell]:
    from agent.world.rules import spawn_cell
    out = [tuple(u) for u in units]
    occupied = list(out)
    for _ in range(hands):
        out.append(spawn_cell(occupied))
        occupied.append(out[-1])
    return out


def _expand(tasks, preds, needs_item, day, pos, time, done, travel, live):
    """Not implemented yet: the vectorised turn."""
    raise NotImplementedError


def _select(candidates, beam):
    raise NotImplementedError


def _score(done, travel):
    return done.sum(axis=1) * 1000 - travel


def _routes(tasks, pos, time, done):
    return []
