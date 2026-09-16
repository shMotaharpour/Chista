"""Dispatcher: a committed day plan -> one action dict for the engine.

The plan format in M1 is a list of per-unit op lists
(``[["PLANT", "WHEAT"], ["WATER"], ...]`` — farmer first, then hands in
``hands`` order, F030). The dispatcher slices it by turn: hour h of the
day gives every unit its h-th op, or PASS when the unit's list is
exhausted. Market orders ride on the plan's hour-0 entry (F032 order:
land, sells, hires, purchases), capped at 10 — per turn, not per day,
the 11th silently dropped by the engine (F031), so dev mode asserts it.

Engine behaviours respected here (all silent when violated, F047):
- one op per unit per turn, hands in `hands` order (F030);
- a unit acts BEFORE that turn's market: a purchase scheduled for hour h
  is only available to ops at hour h+1, so plans buy one turn ahead (F030);
- HIRE and BUY_LAND settle atomically at their index, before the unit
  loop (F031);
- a hand hired at hour 0 first acts at hour 1 — 23 actions, not 24 (F040);
- anything behind an empty purse is refused silently (F031).
"""

from __future__ import annotations

from typing import Any

MAX_MARKET_ORDERS = 10     # F031; the engine drops the 11th silently

PASS_ACTION = {"farmer": ["PASS"], "hands": [], "market": []}


def market_at(market, hour: int) -> list:
    """This turn's market orders, from a per-hour queue or a flat list.

    #15 made `plan["market"]` per-hour (`market[hour] -> [order, ...]`),
    because the ≤ 10-order cap is PER TURN (F031) and a 24-turn day has 240
    slots. The M1 flat list (`[["SELL", "WHEAT", 3]]`, all of it at hour 0)
    still dispatches: its first element is an ORDER, while a queue's first
    element is a ROW of orders.
    """
    if not market or not isinstance(market, (list, tuple)):
        return []
    head = market[0]
    per_hour = (isinstance(head, (list, tuple))
                and (not head or isinstance(head[0], (list, tuple))))
    if per_hour:
        row = market[hour] if 0 <= hour < len(market) else []
        return [list(order) for order in (row or [])]
    return [list(order) for order in market] if hour == 0 else []


def dispatch_plan(plan, obs) -> dict:
    """Slice `plan` into this turn's action dict.

    `plan` is a dict: ``{"units": [[op, ...], ...], "market": [[op, ...], ...]}``
    where units[0] is the farmer and units[1:] the hands in order. The hour
    comes from the observation. A unit with no op at this hour passes.
    """
    if not isinstance(plan, dict) or "units" not in plan:
        raise ValueError("plan must be {'units': [...], 'market': [...]}")
    units = plan["units"]
    if not (isinstance(units, list)
            and all(isinstance(u, list) for u in units)):
        raise ValueError("plan units must be a list of per-unit op lists")
    hour = int(obs.get("hour", 0)) if isinstance(obs, dict) else 0

    if not units:
        # an empty plan is legitimate (every tile locked, nothing worth
        # doing): everyone passes, the market orders still ride
        market = market_at(plan.get("market", []), hour)
        return {"farmer": ["PASS"], "hands": [],
                "market": market[:MAX_MARKET_ORDERS]}

    farmer = list(units[0][hour]) if hour < len(units[0]) else ["PASS"]
    # F031: a HIRE behind a short purse is refused SILENTLY, so the day's
    # real hand count can differ from what the plan assumed. The board is
    # authoritative; ops for hands that do not exist are dropped, and a
    # real hand without a planned op passes. (F047 class - never ship ops
    # for units the engine will ignore.)
    real_hands = len(obs.get("farms", [{}])[obs.get("player", 0)]
                     .get("hands", [])) if isinstance(obs, dict) else len(units) - 1
    hands = []
    for i in range(real_hands):
        if i + 1 < len(units) and hour < len(units[i + 1]):
            hands.append(list(units[i + 1][hour]))
        else:
            hands.append(["PASS"])
    market = market_at(plan.get("market", []), hour)
    if len(market) > MAX_MARKET_ORDERS:
        # F031: the engine drops the 11th silently - never send one
        market = market[:MAX_MARKET_ORDERS]
    # #15: `plan["market"]` is a per-hour queue (`market[hour] -> [orders]`),
    # built by `secretary/inventory.py::market_queue`: the cap is per TURN
    # (F031), so a 24-turn day has 240 slots, and a large forced sale is
    # spread across the day's turns instead of landing as one basket (the
    # measured scope of that is in `plan_sales`).
    return {"farmer": farmer, "hands": hands, "market": market}
