"""Guards for the land order: priced by the day's own bill, first in its turn.

The owner's rule this session: orders come from the hourly secretary
(`agent/planner/market.py::build`, called from `day.py`), not from the retired
`repair.py`. Land therefore enters the day exactly where a hire does: priced, added to the
bill the SELL side is sized against, and queued where F032 puts it (land, sells, hires,
purchases).

R007: each guard was broken and seen red before it was trusted.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.planner.market import SETTLE_RANK, land_orders, merge
from agent.world.rules import LAND_PRICES, TURNS_PER_DAY


def test_the_first_purchase_costs_the_first_price() -> None:
    """F042: the price depends on how many quadrants are already owned, not on the day."""
    orders, bill = land_orders(0, 1)
    assert orders == [["BUY_LAND"]], orders
    assert bill == int(LAND_PRICES[0]), bill


def test_the_price_walks_the_forced_order() -> None:
    """A second and third purchase cost the second and third prices: 1000, 2000, 4000."""
    _, bill1 = land_orders(1, 1)
    assert bill1 == int(LAND_PRICES[1]), bill1
    _, bill2 = land_orders(2, 1)
    assert bill2 == int(LAND_PRICES[2]), bill2


def test_nothing_is_bought_past_the_third_quadrant() -> None:
    """With every quadrant owned there is no fourth purchase, and no order is invented."""
    assert land_orders(3, 3) == ([], 0)
    # Two asked for, one available: the day gets one order and pays one price.
    orders, bill = land_orders(2, 2)
    assert orders == [["BUY_LAND"]], orders
    assert bill == int(LAND_PRICES[2]), bill


def test_a_day_never_orders_more_than_the_three_quadrants() -> None:
    """Three from scratch is the most a day can buy, and the bill is their sum."""
    orders, bill = land_orders(0, 9)
    assert len(orders) == 3, orders
    assert bill == sum(int(p) for p in LAND_PRICES), bill


def test_no_purchase_is_ordered_when_none_is_asked_for() -> None:
    """The default keeps today's behaviour: a day that does not decide to buy orders nothing."""
    assert land_orders(0, 0) == ([], 0)
    assert land_orders(2, 0) == ([], 0)


def test_land_opens_its_turn_before_sells_and_hires() -> None:
    """F032's queue order is land, then sells, then hires, then purchases — and the engine
    settles atomic orders first WITHIN an index (`market.py`'s own note), so what a land
    order needs is a slot inside the cap, not the head of the row: the sells keep the head
    because those are the orders the rival is quoted against."""
    sells: list = [[] for _ in range(TURNS_PER_DAY)]
    sells[0] = [["SELL", "WHEAT", 3]]
    rows, leftover = merge(sells, [["HIRE"]], [], lands=[["BUY_LAND"]])
    assert ["BUY_LAND"] in rows[0], rows[0]
    assert rows[0][0][0] == "SELL", "the market-priced order lost the head of the row"
    assert SETTLE_RANK["BUY_LAND"] < SETTLE_RANK["SELL"] < SETTLE_RANK["HIRE"]
    assert not leftover, leftover


def test_a_land_order_is_queued_even_when_the_row_is_full_of_our_own_orders() -> None:
    """Ten sells in a turn fill the cap; the land order spills to a later turn rather than
    being dropped, which is what `merge`'s leftover reporting is for."""
    sells: list = [[] for _ in range(TURNS_PER_DAY)]
    sells[0] = [["SELL", "WHEAT", 1] for _ in range(10)]
    rows, leftover = merge(sells, [], [], lands=[["BUY_LAND"]])
    assert ["BUY_LAND"] not in rows[0], "the cap was exceeded in turn 0"
    assert any(["BUY_LAND"] in row for row in rows), "the land order was lost, not spilled"


def test_the_land_order_joins_the_queue_like_any_other() -> None:
    """With no land asked for, the queue is exactly what it was: nothing changes by default."""
    sells: list = [[] for _ in range(TURNS_PER_DAY)]
    sells[0] = [["SELL", "WHEAT", 3]]
    rows, _ = merge(sells, [["HIRE"]], [])
    assert ["BUY_LAND"] not in rows[0], rows[0]


def main() -> int:
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
            except Exception as exc:  # noqa: BLE001 - test runner
                failures += 1
                print(f"FAIL {name}: {type(exc).__name__}: {exc}")
            else:
                print(f"PASS {name}")
    if failures:
        print(f"{failures} test(s) failed")
        return 1
    print("all land-order guards passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
