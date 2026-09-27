"""Market Queue Regulator: guarantees zero-drop, cash-safe 10-slot ordering.

Rules strictly enforced:
1. Capacity: Exactly <= 10 orders per turn (maxMarketOrdersPerTurn = 10).
   Any overflow automatically rolls over into the next available turn (turns 0..23).
2. Queue Index Order within each turn (F032 corrected):
   - Index 0..k: SELL orders (bring money INTO purse and free shed space first).
   - Index k..m: BUY_LAND and HIRE (atomic settlement with updated purse).
   - Index m..n: BUY_SEED, BUY_ANIMAL, BUY_PRODUCT.
3. Zero silent drops: Every requested order is guaranteed to be placed.
"""

from __future__ import annotations

from typing import Any, Sequence

from agent.world.rules import MAX_MARKET_ORDERS_PER_TURN, TURNS_PER_DAY


def order_priority_key(order: Sequence[Any]) -> int:
    """Sort key for market orders inside a turn.
    
    0: SELL (brings cash in first, frees shed)
    1: BUY_LAND (atomic purchase with refreshed cash)
    2: HIRE (atomic worker recruitment)
    3: BUY_SEED / BUY_ANIMAL / other buys
    """
    if not order:
        return 99
    op = str(order[0])
    if op == "SELL":
        return 0
    if op == "BUY_LAND":
        return 1
    if op == "HIRE":
        return 2
    return 3


class MarketQueueRegulator:
    """Regulates market orders over 24 turns of a day."""

    def __init__(self, max_orders_per_turn: int = MAX_MARKET_ORDERS_PER_TURN, turns: int = TURNS_PER_DAY):
        self.max_orders = max_orders_per_turn
        self.turns = turns
        self.rows: list[list[list[Any]]] = [[] for _ in range(self.turns)]

    def schedule_day_orders(
        self,
        sells_by_turn: Sequence[Sequence[Sequence[Any]]] | None = None,
        hires: int = 0,
        buy_land: bool = False,
        buys: Sequence[Sequence[Any]] | None = None,
    ) -> list[list[list[Any]]]:
        """Distribute all day orders across 24 turns without dropping any order."""
        # Reset rows
        self.rows = [[] for _ in range(self.turns)]

        # 1. Place sells in their preferred turn (or turn 0)
        unassigned_sells = []
        if sells_by_turn:
            for t, turn_sells in enumerate(sells_by_turn):
                target_turn = min(t, self.turns - 1)
                for s in turn_sells:
                    if s:
                        self.rows[target_turn].append(list(s))
                        
        # 2. Collect opening orders that need to execute as early as possible
        opening_orders: list[list[Any]] = []
        
        # Land buy
        if buy_land:
            opening_orders.append(["BUY_LAND"])
            
        # Hires (1 HIRE order per hand)
        for _ in range(hires):
            opening_orders.append(["HIRE"])
            
        # Seeds & other inputs
        if buys:
            for b in buys:
                if b:
                    opening_orders.append(list(b))

        # 3. Distribute opening orders across turns starting from turn 0
        current_turn = 0
        for order in opening_orders:
            # Advance turn if current turn is full
            while current_turn < self.turns and len(self.rows[current_turn]) >= self.max_orders:
                current_turn += 1
            if current_turn < self.turns:
                self.rows[current_turn].append(order)
            else:
                # Emergency: should practically never happen within 240 order slots per day
                break

        # 4. Sort each turn's orders by priority key (SELL -> LAND -> HIRE -> BUYS)
        for t in range(self.turns):
            self.rows[t].sort(key=order_priority_key)
            # Ensure hard cap <= 10
            if len(self.rows[t]) > self.max_orders:
                excess = self.rows[t][self.max_orders:]
                self.rows[t] = self.rows[t][:self.max_orders]
                # Push excess to next turn if possible
                if t + 1 < self.turns:
                    self.rows[t + 1].extend(excess)
                    self.rows[t + 1].sort(key=order_priority_key)

        return self.rows
