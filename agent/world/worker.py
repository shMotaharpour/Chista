"""A worker's day: where it started, and what it did in each hour.

A `WorkerTrace` is what a plan is made of and what a verifier reads back: one cell for
the day's start, and one slot per turn of the day. A slot is either the action the
worker took (`("WATER",)`, `("PLANT", "MELON")`, `("PICKUP", "WHEAT", 2)` — an
engine-shaped action list, so a parameter an action needs is carried) or `None`, for an
hour that has not been decided yet.

Positions are derived, never stored: a worker walks, so its cell in an hour is its
start plus the moves it recorded before it. That keeps the trace from being able to
disagree with itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from agent.world.action import WorkerAction
from agent.world.board import Cell, in_board
from agent.world.model import Move
from agent.world.rules import TURNS_PER_DAY

#: What a move does to a cell (kaggriculture.py:88-93).
DELTA_OF_MOVE: dict[str, Cell] = {
    Move.NORTH: (0, -1), Move.SOUTH: (0, 1), Move.EAST: (1, 0), Move.WEST: (-1, 0)}


@dataclass(frozen=True)
class WorkerTrace:
    """One worker's day. `hours[h]` is what it did in hour `h`, or `None`."""

    #: Where the worker stands at hour 0 (the engine sends the farmer back to the spawn
    #: tile every night, and hands are removed: :879-882).
    start: Cell
    hours: tuple[WorkerAction | None, ...] = field(default_factory=tuple)

    @classmethod
    def begin(cls, start: Cell, turns_per_day: int = TURNS_PER_DAY) -> "WorkerTrace":
        """An empty day for a worker that begins on `start`."""
        return cls(start=(int(start[0]), int(start[1])),
                   hours=(None,) * turns_per_day)

    def record(self, hour: int, action: WorkerAction | None) -> "WorkerTrace":
        """The same trace with hour `hour` filled in (immutable, like every state)."""
        if not 0 <= hour < len(self.hours):
            raise ValueError(f"hour {hour} is outside the {len(self.hours)}-turn day")
        hours = list(self.hours)
        hours[hour] = action
        return replace(self, hours=tuple(hours))

    # --- what it did -------------------------------------------------------- #

    def action_at(self, hour: int) -> WorkerAction | None:
        return self.hours[hour]

    @property
    def done(self) -> tuple[tuple[int, WorkerAction], ...]:
        """`(hour, action)` for every hour that has been decided."""
        return tuple((h, a) for h, a in enumerate(self.hours) if a is not None)

    @property
    def pending_hours(self) -> tuple[int, ...]:
        """The hours with nothing in them yet."""
        return tuple(h for h, a in enumerate(self.hours) if a is None)

    @property
    def is_complete(self) -> bool:
        return all(a is not None for a in self.hours)

    def ops(self) -> tuple[str, ...]:
        """The op of every recorded action, in hour order."""
        return tuple(str(a.op.value) for a in self.hours if a is not None)

    # --- where it is -------------------------------------------------------- #

    def cell_at(self, hour: int) -> Cell:
        """The cell the worker acts FROM in hour `hour`: its start plus the moves it
        recorded in the hours before. A move off the board is a no-op (:326-327), so it
        is ignored here too."""
        x, y = self.start
        for action in self.hours[:hour]:
            if action is not None and action.op in DELTA_OF_MOVE:
                dx, dy = DELTA_OF_MOVE[action.op]
                if in_board((x + dx, y + dy)):
                    x, y = x + dx, y + dy
        return (x, y)

    def cell_after(self, hour: int) -> Cell:
        """The cell the worker is on once hour `hour` is over."""
        return self.cell_at(hour + 1)

    def path(self) -> tuple[Cell, ...]:
        """The cell it acts from, for every hour of the day."""
        return tuple(self.cell_at(h) for h in range(len(self.hours)))
