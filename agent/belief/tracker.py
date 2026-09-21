"""Phase 1 — the residual tracker: what the rival actually did, from public data.

Per good, for the turn that just ended:

    I(t) = I(t-1) + our_sells[price > 1] + their_sells - our_buys
                    - their_buys - town_drain(t-1)

Each term but theirs is known: the market inventory and both money balances are
public, our own action is ours, and the town's consumption is deterministic
given the shops already open (a shop never closes). So the residual identifies
the rival **exactly for the seven goods that cannot be bought**. For WHEAT and
FERTILIZER a buy is quoted at `price(I-1)` and a sale earns `price(I)`, so both
channels enter the flow and the money identity with the same sign; only their
**net** is identifiable, and `diagnostics()` reports the slope that would be
needed to separate them (measured: 0-1 coin per unit, i.e. not separable).

The measured claim, from `bench/bench_market_analyzer.py`: over 143 turns of a
real `fast_sim` episode the residual matched the rival's committed orders with
mean 0.0000 and max 0.0000 units on the seven one-way goods, and every unit of
error sat on WHEAT (mean 0.28, max 3).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from agent.world.model import PRODUCTS
from agent.world.rules import (ANIMAL_RULES, LAND_PRICES, SHED_ACCESS,
                               SHED_CAPACITY)
from agent.world.prices import MARKET_I0, PRICE_FLOOR, price_of, price_table
from agent.belief.schemas import (CENTER_INTERVAL, CENTER_PRODUCTS, DUAL,
                                  MAX_ORDERS, SELL_ONLY, SHOP_BASKET,
                                  SHOP_INTERVAL, SHOP_TYPES, UNLOCK_INTERVAL,
                                  Turn, field_of)

#: name -> PRODUCTS index. The (9,) arrays belief carries are in PRODUCTS order.
G_IX: dict[str, int] = {p: i for i, p in enumerate(PRODUCTS)}
GOODS: tuple[str, ...] = PRODUCTS
SHED_CAP: int = SHED_CAPACITY


@dataclass
class FlowRecord:
    """One turn's flow decomposition; every array is (9,) in `GOODS` order."""
    step: int
    drain: np.ndarray            # the town's consumption that turn (deterministic)
    our_sales: np.ndarray
    our_buys: np.ndarray
    rival_sales: np.ndarray      # exact for the 7 one-way goods
    rival_buys: np.ndarray       # only the NET with rival_sales is identifiable
    price_was_floor: np.ndarray  # a sale at price 1 adds no market supply
    rival_harvest: np.ndarray    # inferred from their public board
    rival_stock: np.ndarray      # shed-level estimate, capped at SHED_CAP
    rival_bag: np.ndarray        # harvest not yet dropped: a lower bound


def fib_hire_costs(n_hands: int) -> int:
    """Engine hire cost of `n` hands hired in one day: `sum(_fib(0..n-1))`, `_fib(0)=1`."""
    a, b, total = 1, 1, 0
    for _ in range(max(0, n_hands)):
        total += a
        a, b = b, a + b
    return total


def land_cost(n_extra_quadrants: int) -> int:
    """Prefix-locked land: NE, SW, SE at 1000/2000/4000 (F042)."""
    return int(sum(LAND_PRICES[:max(0, n_extra_quadrants)]))


def tile_item(tile: Any) -> str | None:
    """The good a tile pays out when harvested, or None."""
    if not isinstance(tile, dict):
        return None
    if tile.get("kind") == "PLANT":
        return tile.get("crop")
    if "animal" in tile:
        return ANIMAL_RULES[tile["animal"]]["product"]
    return None


def committed_flow(action: dict, shed: dict[str, int], money: float,
                   inventory: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The market flow a queue commits, given the shed, the purse and the quotes.

    BUY_PRODUCT is quoted at `price(I-1)` (the engine's own netting rule) and is
    refused when the purse or the shed room runs out; SELL is refused when the
    shed lacks the good. Only the quantities matter here.
    """
    sales = np.zeros(len(GOODS))
    buys = np.zeros(len(GOODS))
    shed = dict(shed)
    room = SHED_CAP - int(sum(shed.values()))
    for order in list(action.get("market", []) or [])[:MAX_ORDERS]:
        if not isinstance(order, (list, tuple)) or len(order) < 3:
            continue
        op, item, qty = order[0], order[1], int(order[2])
        if item not in G_IX or qty <= 0:
            continue
        gi = G_IX[item]
        if op == "SELL":
            take = min(qty, int(shed.get(item, 0)))
            if take > 0:
                shed[item] = shed.get(item, 0) - take
                room += take
                sales[gi] += take
        elif op == "BUY_PRODUCT" and item in DUAL:
            price = max(1, price_of(item, float(inventory[gi]) - 1.0))
            take = min(qty, max(0, room), int(money // price))
            if take > 0:
                money -= take * price
                room -= take
                buys[gi] += take
    return sales, buys


class MarketTracker:
    """Hour-by-hour tracker for one seat. Feed it every observation, in order."""

    def __init__(self, player: int = 0) -> None:
        self.player = player
        self.step = -1
        self.inventory = np.full(len(GOODS), float(MARKET_I0))
        self.prices = price_table(self.inventory)
        self.records: list[FlowRecord] = []
        self.rival_stock = np.zeros(len(GOODS))
        self.rival_bag = np.zeros(len(GOODS))
        self._pending_drain = np.zeros(len(GOODS))
        self._pending_our_sales = np.zeros(len(GOODS))
        self._pending_our_buys = np.zeros(len(GOODS))
        self._board: dict[tuple[int, int], tuple[int, str | None]] = {}
        self._rival_money: float | None = None
        self._rival_hands = 0
        self._rival_quadrants = 0
        self.diagnostics: dict[str, Any] = {}

    # -- engine-side facts --------------------------------------------------- #

    @staticmethod
    def drain_of(obs: Any) -> np.ndarray:
        """The town's consumption applied *this* turn: shops every 4 steps, centre every 24."""
        step = int(field_of(obs, "step", 0))
        out = np.zeros(len(GOODS))
        shops = list(field_of(field_of(obs, "town", {}) or {}, "unlocked_shops", []) or [])
        if step % SHOP_INTERVAL == 0:
            for s in shops:
                items, mult = SHOP_BASKET[s]
                for item in items:
                    out[G_IX[item]] += mult
        if step % CENTER_INTERVAL == 0:
            for item in CENTER_PRODUCTS:
                out[G_IX[item]] += 1
        return out

    def _board_of(self, farm: Any) -> dict[tuple[int, int], tuple[int, str | None]]:
        out: dict[tuple[int, int], tuple[int, str | None]] = {}
        for y, row in enumerate(field_of(farm, "tiles", []) or []):
            for x, tile in enumerate(row):
                if isinstance(tile, dict):
                    out[(x, y)] = (int(tile.get("yield_units", 0)), tile_item(tile))
        return out

    def note_our_action(self, action: dict, obs: Any) -> None:
        """Record what our action really produced.

        The unit phase runs before the market, so a DROP in this turn's action is
        what the SELL gate sees -- reading the shed from the top of the turn would
        put our own error on the rival's side of the residual.
        """
        private = field_of(obs, "private", {}) or {}
        shed = {k: int(v) for k, v in (field_of(private, "shed", {}) or {}).items()}
        bags = [dict(b) for b in (field_of(private, "inventories", []) or [])]
        farms = field_of(obs, "farms", []) or []
        farm = farms[self.player] if len(farms) > self.player else {}
        money = float(field_of(farm, "money", 0.0))

        positions = [tuple(field_of(farm, "farmer", []) or ())]
        positions += [tuple(p) for p in (field_of(farm, "hands", []) or [])]
        unit_acts = [list(field_of(action, "farmer", ["PASS"]) or ["PASS"])]
        unit_acts += [list(a) for a in (field_of(action, "hands", []) or [])]
        for idx, act in enumerate(unit_acts):
            if idx >= len(bags) or not act or idx >= len(positions):
                continue
            if act[0] == "DROP" and positions[idx] in SHED_ACCESS:
                for item, n in list(bags[idx].items()):
                    take = max(0, min(int(n), SHED_CAP - sum(shed.values())))
                    if take:
                        shed[item] = shed.get(item, 0) + take
                bags[idx] = {}
            elif act[0] == "PICKUP" and len(act) >= 2:
                item = act[1]
                want = int(act[2]) if len(act) >= 3 else 1
                take = max(0, min(want, int(shed.get(item, 0))))
                if take:
                    shed[item] = shed.get(item, 0) - take

        sales, buys = committed_flow(action, shed, money, self.inventory)
        self._pending_our_sales += sales
        self._pending_our_buys += buys

    # -- main entry ---------------------------------------------------------- #

    def observe(self, obs: Any) -> FlowRecord | None:
        """Absorb one observation; returns the flow record (None on the first turn)."""
        step = int(field_of(obs, "step", 0))
        market = field_of(obs, "market", {}) or {}
        inventory = np.array([float(dict(field_of(market, "inventory", {}) or {}).get(g, 0))
                              for g in GOODS])
        prices = np.array([int(dict(field_of(market, "prices", {}) or {}).get(g, 0))
                           for g in GOODS])
        farms = field_of(obs, "farms", []) or []
        rival = farms[1 - self.player]
        rival_money = float(field_of(rival, "money", 0.0))
        rival_hands = len(field_of(rival, "hands", []) or [])
        rival_quads = len(field_of(rival, "unlocked_quadrants", []) or [])
        board = self._board_of(rival)

        if self.step < 0:
            self.step, self.inventory, self.prices = step, inventory, prices
            self._pending_drain = self.drain_of(obs)
            self._board = board
            self._rival_money, self._rival_hands = rival_money, rival_hands
            self._rival_quadrants = rival_quads
            return None

        # their harvest: a yield counter zeroed into the acting unit's bag. A
        # one-unit drop is `_decay_plants` past max lifespan, not a harvest.
        harvest = np.zeros(len(GOODS))
        for pos, (prev_yield, item) in self._board.items():
            cur_yield = board.get(pos, (0, None))[0]
            if prev_yield > 0 and cur_yield == 0 and item in G_IX:
                harvest[G_IX[item]] += prev_yield

        delta = inventory - self.inventory
        net = delta + self._pending_drain + self._pending_our_buys - self._pending_our_sales
        rival_sales = np.maximum(net, 0.0)
        rival_buys = np.maximum(-net, 0.0)
        for i, g in enumerate(GOODS):
            if g in SELL_ONLY:
                rival_buys[i] = 0.0

        self.diagnostics = dict(
            money_delta=rival_money - (self._rival_money
                                       if self._rival_money is not None else rival_money),
            hire_delta=(fib_hire_costs(rival_hands) - fib_hire_costs(self._rival_hands)),
            land_delta=(land_cost(max(0, rival_quads - 1))
                        - land_cost(max(0, self._rival_quadrants - 1))),
            net_on_duals={g: float(rival_sales[G_IX[g]] - rival_buys[G_IX[g]]) for g in DUAL},
            slope_per_unit={g: price_of(g, float(inventory[G_IX[g]]))
                            - price_of(g, float(inventory[G_IX[g]]) - 1) for g in DUAL},
        )

        self.rival_bag = np.maximum(0.0, self.rival_bag + harvest - rival_sales)
        self.rival_stock = np.clip(self.rival_stock + harvest - rival_sales, 0, SHED_CAP)

        rec = FlowRecord(
            step=step,
            drain=self._pending_drain.copy(),
            our_sales=self._pending_our_sales.copy(),
            our_buys=self._pending_our_buys.copy(),
            rival_sales=rival_sales,
            rival_buys=rival_buys,
            price_was_floor=(self.prices <= PRICE_FLOOR).copy(),
            rival_harvest=harvest,
            rival_stock=self.rival_stock.copy(),
            rival_bag=self.rival_bag.copy(),
        )
        self.records.append(rec)

        self.step, self.inventory, self.prices = step, inventory, prices
        self._pending_drain = self.drain_of(obs)
        self._pending_our_sales = np.zeros(len(GOODS))
        self._pending_our_buys = np.zeros(len(GOODS))
        self._board = board
        self._rival_money, self._rival_hands = rival_money, rival_hands
        self._rival_quadrants = rival_quads
        return rec

    def rival_volume(self, turn_index: int) -> np.ndarray:
        """Their inferred sales for the most recent recorded turn."""
        if not self.records:
            return np.zeros(len(GOODS))
        return self.records[turn_index].rival_sales

    def activity_bucket(self, step: int, window: int = 24) -> int:
        """The rival's own sell bucket over the last `window` turns.

        0 start (no history yet), 1 silent (0 units), 2 low, 3 mid,
        4 high — the same regime the trained artifact is keyed by
        (`_key_activity`). The rival's per-turn volume here is the
        tracker's inferred `rival_sales`: exact on the seven one-way
        goods, net on the two duals.
        """
        n = len(self.records)
        if step < window or n == 0:
            return 0
        recent = self.records[-min(window, n):]
        sold = float(sum(int(r.rival_sales.sum()) for r in recent))
        if sold <= 0:
            return 1
        if sold > 60:
            return 4
        if sold > 10:
            return 3
        return 2
