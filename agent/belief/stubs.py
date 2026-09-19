"""The replacements: what runs until a unit is built, and what retires it.

Every function here implements the *contract* of a module that does not exist
yet, so the pipeline can be assembled, measured and reviewed end to end while
the real unit is written. Each one names the issue that retires it, and each is
deliberately the cheapest honest behaviour — a stub that pretends to be clever
would hide the missing unit instead of exposing it.

| stub | contract | retires with |
|---|---|---|
| `rival_supply_stub` | `MarketState`'s rival forecast | the rival inference graph issue |
| `naive_order_book` | `OrderBook` | the order book issue |
| `naive_day_plan` | the crew's `DaySchedule` | the WSR compiler issue (the missing half of #14) |

`tests/test_market_analyzer.py::test_stubs_keep_the_pipeline_importable` checks
shape, not behaviour: the point is that a caller can be written today.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from agent.world.model import DUAL, PRODUCTS
from agent.belief.schemas import (DaySchedule, G_IX, MAX_ORDERS, MarketState,
                                  OrderBook, SELL_ONLY, SellIntent, SHED_CAP,
                                  field_of)

#: The (9,) goods order, from the world's own name for it.
GOODS: tuple[str, ...] = PRODUCTS


def rival_supply_stub(goods: tuple[str, ...] = GOODS) -> tuple[np.ndarray, float]:
    """No inference graph yet: predict nothing, and say so with confidence 0.

    A zero forecast is honest; a made-up forecast would put a fudge factor inside
    the price path where nobody would find it.
    """
    return np.zeros(len(goods)), 0.0


def naive_order_book(shed: dict[str, int], intents: list[SellIntent] | None = None,
                     hires_today: int = 0) -> OrderBook:
    """Sell the shed at slot 0 of turn 0, hire first. The baseline, not a plan.

    It ignores price impact entirely, which is precisely what the order book
    issue has to beat on the arena; keeping it here gives that issue a measured
    starting point instead of an opinion.
    """
    book = OrderBook()
    if hires_today > 0:
        book.add(0, ("HIRE",))
    placed = 0
    for good, qty in sorted(shed.items()):
        if placed >= MAX_ORDERS:
            break
        if qty > 0 and good not in ("FERTILIZER",):
            book.add(0, ("SELL", good, int(qty)))
            placed += 1
    return book


def naive_day_plan(unit_tiles: list[int | None], seeds: int = 0,
                   market: list[list[Any]] | None = None) -> DaySchedule:
    """One unit per tile, one op per turn: what `agent/replan.py` does today.

    No movement, no carries, no pickup — so a chain that assumes fertiliser in
    the actor's bag is refused in silence (F047). That failure mode is the
    reason the WSR compiler exists, and this stub reproduces it on purpose.
    """
    per_unit: list[list[list[str]]] = []
    for tile in unit_tiles:
        if tile is None:
            per_unit.append([["PASS"]])
        elif seeds > 0:
            per_unit.append([["PLANT", "WHEAT"]])
            seeds -= 1
        else:
            per_unit.append([["WATER"]])
    return DaySchedule(per_unit=per_unit, realised_hours={}, carry_plan=[],
                       infeasible=[])


def state_from_tracker(tracker: Any, obs: Any, mean: np.ndarray, sd: np.ndarray
                       ) -> MarketState:
    """Assemble the published `MarketState` from the tracker's current readings.

    `rival_sales` here is *measured* (the tracker's residual), not forecast: the
    forecast column is the stub's zero until the inference graph lands.
    """
    _forecast, _confidence = rival_supply_stub()
    last = tracker.records[-1].rival_sales if tracker.records else np.zeros(len(GOODS))
    shed = (field_of(field_of(obs, "private", {}) or {}, "shed", {}) or {})
    return MarketState(
        turn=int(field_of(obs, "step", 0)),
        inventory=tracker.inventory.copy(),
        prices=tracker.prices.copy(),
        drain_mean=np.asarray(mean, dtype=float),
        drain_sd=np.asarray(sd, dtype=float),
        rival_sales=np.asarray(last, dtype=float).copy(),
        rival_stock=tracker.rival_stock.copy(),
        rival_bag=tracker.rival_bag.copy(),
        shed_room=SHED_CAP - int(sum(shed.values())),
    )
