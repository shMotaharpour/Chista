"""The turns the layers speak through.

One rule, from the architecture review: **nobody commands an op across a layer
boundary.** A message says what must be true and by when; the receiving layer
owns how. `SellIntent` is the market side's objective, `DropRequirement` is the
only time demand it makes on the crew, and the crew answers with a
`DaySchedule` that reports real hours and names what it could not do.

Every field is either public (in the observation), computed by `belief`, or a
decision of the layer that emitted it. Nothing carries a hidden assumption:
`reason` and `why` exist so a later reader can see the arithmetic behind a
number (R005).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from agent.world.vocabulary import G_IX, GOODS, MAX_ORDERS, SHED_CAP   # noqa: F401  (re-export)

Turn = int                     # global turn index, 0..718 (F048)
Window = tuple[int, int]       # [earliest_turn, latest_turn]
Order = tuple[Any, ...]        # the engine's list form, e.g. ("SELL", "MILK", 3)


def field_of(obj: Any, key: str, default: Any = None) -> Any:
    """Read a field from the observation whether it is a dict or a struct."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


# --- what the state estimator publishes ------------------------------------- #

@dataclass(frozen=True)
class MarketState:
    """The world as `belief` sees it, for every layer that prices something."""
    turn: Turn
    inventory: np.ndarray          # (9,) public market inventory
    prices: np.ndarray             # (9,) engine quotes at that inventory
    drain_mean: np.ndarray         # (9,) expected town consumption, closed form
    drain_sd: np.ndarray           # (9,) its spread (future unlocks only)
    rival_sales: np.ndarray        # (9,) this turn's inferred sales
    rival_stock: np.ndarray        # (9,) shed-level estimate, capped at SHED_CAP
    rival_bag: np.ndarray          # (9,) not-yet-dropped lower bound
    shed_room: int                 # SHED_CAP - sum(our shed)

    @classmethod
    def from_obs(cls, obs: Any) -> "MarketState":
        """The snapshot, from the observation alone.

        Inventory and prices are the observation's and the engine's own quote
        function (`world.prices`, parity-tested); the drain is the closed form in
        `belief.opponent`. The rival fields stay zero and say so: they need the
        tracker's residual history, and a caller that has it passes its own state.
        """
        from belief.opponent import drain_forecast
        from agent.world.prices import price_table
        from agent.world.vocabulary import GOODS

        market = obs.get("market", {}) if isinstance(obs, dict) else {}
        raw = market.get("inventory", {}) or {}
        inventory = np.array([int(raw.get(g, 0)) for g in GOODS], dtype=float)
        mean, sd = drain_forecast(obs, 1)
        private = obs.get("private", {}) if isinstance(obs, dict) else {}
        shed = sum(int(v) for v in (private.get("shed", {}) or {}).values())
        return cls(
            turn=int(obs.get("step", 0)) if isinstance(obs, dict) else 0,
            inventory=inventory, prices=np.asarray(price_table(inventory), dtype=float),
            drain_mean=np.asarray(mean, dtype=float),
            drain_sd=np.asarray(sd, dtype=float),
            rival_sales=np.zeros(len(GOODS)), rival_stock=np.zeros(len(GOODS)),
            rival_bag=np.zeros(len(GOODS)), shed_room=int(SHED_CAP) - shed)


# --- what the season planner asks for --------------------------------------- #

@dataclass(frozen=True)
class SellIntent:
    """A lot to sell inside a window. Only the order book reads `priority`."""
    good: str
    qty: int
    day: int
    window: Window
    priority: int                  # 0 leads its turn
    reason: str                    # "shed room", "price peak day 18", "week-1 cash"


@dataclass(frozen=True)
class PurchaseIntent:
    """Buy `qty` of `item`; it is usable from `needed_at` and in the shed by `by_turn`."""
    item: str
    qty: int
    needed_at: Turn
    by_turn: Turn
    budget_cap: int


@dataclass(frozen=True)
class TileRequirement:
    """One tile's day: the ops the contractor priced, and what they assume."""
    tile: int
    ops: tuple[str, ...]
    window: Window
    weight: float                  # coins per turn of lateness
    inputs_needed: dict[str, int]  # what the ops assume in the actor's bag
    expected_yield: dict[str, int]


@dataclass(frozen=True)
class CarryRequirement:
    """Move goods from the shed ring to a tile before `deadline_turn`."""
    item: str
    qty: int
    to_tile: int
    ready_turn: Turn               # after the buy slot landed (+1 for the shed gate)
    deadline_turn: Turn


@dataclass(frozen=True)
class DropRequirement:
    """The market side's only time demand on the crew: goods in the shed by a turn."""
    items: dict[str, int]
    by_turn: Turn
    reason: str                    # the SellIntent it serves


# --- what the crew and the order book answer with ---------------------------- #

@dataclass
class DaySchedule:
    """The crew's answer: ops per unit per turn, real hours, and what it refused."""
    per_unit: list[list[list[str]]]
    realised_hours: dict[int, int]      # feeds the master's labour row next replan
    carry_plan: list[dict]              # the pickup/drop legs actually planned
    infeasible: list[tuple[str, str]]   # (requirement, why) -> the planner re-times


@dataclass
class OrderBook:
    """The market's answer: 24 ordered slot lists, each at most MAX_ORDERS long."""
    per_turn: list[list[Order]] = field(default_factory=list)
    expected_price: list[list[int]] = field(default_factory=list)

    def add(self, turn: int, order: Order, price: int = 0) -> None:
        while len(self.per_turn) <= turn:
            self.per_turn.append([])
            self.expected_price.append([])
        if len(self.per_turn[turn]) >= MAX_ORDERS:
            return                       # the 11th order is dropped by the engine (F031)
        self.per_turn[turn].append(tuple(order))
        self.expected_price[turn].append(price)

    def action_market(self, turn: int) -> list[list[Any]]:
        """This turn's market field, in slot order."""
        if turn >= len(self.per_turn):
            return []
        return [list(o) for o in self.per_turn[turn]]


def good_index(good: str) -> int:
    return G_IX[good]


def empty_state(turn: Turn = 0) -> MarketState:
    """A neutral `MarketState`, for stubs and for the first turn."""
    return MarketState(
        turn=turn,
        inventory=np.full(len(GOODS), float(0)),
        prices=np.zeros(len(GOODS), dtype=np.int64),
        drain_mean=np.zeros(len(GOODS)),
        drain_sd=np.zeros(len(GOODS)),
        rival_sales=np.zeros(len(GOODS)),
        rival_stock=np.zeros(len(GOODS)),
        rival_bag=np.zeros(len(GOODS)),
        shed_room=SHED_CAP,
    )
