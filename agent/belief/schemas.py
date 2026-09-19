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

from agent.world.model import DUAL, PRODUCTS
from agent.world.rules import (CENTER_SELL_INTERVAL_TURNS, MAX_MARKET_ORDERS_PER_TURN,
                               SHED_CAPACITY, SHOPS, SHOP_SELL_INTERVAL_TURNS,
                               SHOP_UNLOCK_INTERVAL_DAYS, TOWN_CENTER_PRODUCTS)

# --- belief's one surface over the world's names ----------------------------- #
# world/model.py and world/rules.py own the definitions; this block derives the
# (9,)-product view belief tracks in, once, each rule citing its engine line.
# Nothing here retypes a world table: goods, duals and shops are world's objects.

#: name -> index into PRODUCTS. The (9,) arrays every belief structure carries
#: are indexed in this order (model.py derives the same pattern for the 18
#: columns as RESOURCE_ID).
G_IX: dict[str, int] = {p: i for i, p in enumerate(PRODUCTS)}

#: The goods the market will never quote a BUY_PRODUCT for (:598): everything
#: but the two duals.
SELL_ONLY: tuple[str, ...] = tuple(g for g in PRODUCTS if g not in DUAL)

SHED_CAP: int = SHED_CAPACITY                          # rules.py:26 (:553,867)
MAX_ORDERS: int = MAX_MARKET_ORDERS_PER_TURN           # rules.py:25 (:551,560)
SHOP_INTERVAL: int = SHOP_SELL_INTERVAL_TURNS          # rules.py:150 (:733)
CENTER_INTERVAL: int = CENTER_SELL_INTERVAL_TURNS      # rules.py:151 (:734,745)
UNLOCK_INTERVAL: int = SHOP_UNLOCK_INTERVAL_DAYS       # rules.py:149 (:886)
CENTER_PRODUCTS: tuple[str, ...] = TOWN_CENTER_PRODUCTS  # rules.py:146

#: One shop instance's basket per consumption event: its products, x2 when the
#: shop sells a single product (kaggriculture.py:740-743).
SHOP_BASKET: dict[str, tuple[tuple[str, ...], int]] = {
    name: (items, 2 if len(items) == 1 else 1) for name, items in SHOPS.items()}

#: The unlock draw is `rng.choice(sorted(SHOPS))` (:891), so a type prior lives
#: in sorted order.
SHOP_TYPES: tuple[str, ...] = tuple(sorted(SHOPS))

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
        from agent.belief.opponent import drain_forecast
        from agent.world.prices import price_table

        market = obs.get("market", {}) if isinstance(obs, dict) else {}
        raw = market.get("inventory", {}) or {}
        inventory = np.array([int(raw.get(g, 0)) for g in PRODUCTS], dtype=float)
        mean, sd = drain_forecast(obs, 1)
        private = obs.get("private", {}) if isinstance(obs, dict) else {}
        shed = sum(int(v) for v in (private.get("shed", {}) or {}).values())
        return cls(
            turn=int(obs.get("step", 0)) if isinstance(obs, dict) else 0,
            inventory=inventory, prices=np.asarray(price_table(inventory), dtype=float),
            drain_mean=np.asarray(mean, dtype=float),
            drain_sd=np.asarray(sd, dtype=float),
            rival_sales=np.zeros(len(PRODUCTS)), rival_stock=np.zeros(len(PRODUCTS)),
            rival_bag=np.zeros(len(PRODUCTS)), shed_room=int(SHED_CAP) - shed)


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
        inventory=np.full(len(PRODUCTS), float(0)),
        prices=np.zeros(len(PRODUCTS), dtype=np.int64),
        drain_mean=np.zeros(len(PRODUCTS)),
        drain_sd=np.zeros(len(PRODUCTS)),
        rival_sales=np.zeros(len(PRODUCTS)),
        rival_stock=np.zeros(len(PRODUCTS)),
        rival_bag=np.zeros(len(PRODUCTS)),
        shed_room=SHED_CAP,
    )
