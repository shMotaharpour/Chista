"""Unit tests for MarketQueueRegulator."""

from __future__ import annotations

import pytest
from agent.planner.market_regulator import MarketQueueRegulator, order_priority_key


def test_order_priority_sorting():
    orders = [
        ["BUY_SEED", "WHEAT", 5],
        ["HIRE"],
        ["BUY_LAND"],
        ["SELL", "CARROT", 10],
    ]
    sorted_orders = sorted(orders, key=order_priority_key)
    # SELL must be index 0, BUY_LAND index 1, HIRE index 2, BUY_SEED index 3
    assert sorted_orders[0][0] == "SELL"
    assert sorted_orders[1][0] == "BUY_LAND"
    assert sorted_orders[2][0] == "HIRE"
    assert sorted_orders[3][0] == "BUY_SEED"


def test_cap_and_spillover():
    regulator = MarketQueueRegulator()
    # 6 sells, 1 land, 6 hires, 5 seed buys = 18 orders in total
    sells = [[["SELL", "WHEAT", 1]] * 6]
    buys = [["BUY_SEED", "MELON", 1]] * 5
    
    rows = regulator.schedule_day_orders(
        sells_by_turn=sells,
        hires=6,
        buy_land=True,
        buys=buys,
    )
    
    # Check 24 turns
    assert len(rows) == 24
    
    # Check max 10 per turn
    for t in range(24):
        assert len(rows[t]) <= 10
        
    # Total orders across all turns must be exactly 18 (zero dropped!)
    total_orders = sum(len(r) for r in rows)
    assert total_orders == 18
    
    # Turn 0 has 10 orders
    assert len(rows[0]) == 10
    # Turn 1 has remaining 8 orders
    assert len(rows[1]) == 8
    
    # In turn 0, SELL must come before BUY_LAND and HIRE
    assert rows[0][0][0] == "SELL"
    # Find positions
    ops_in_0 = [o[0] for o in rows[0]]
    assert ops_in_0.index("SELL") < ops_in_0.index("BUY_LAND")
    assert ops_in_0.index("BUY_LAND") < ops_in_0.index("HIRE")
