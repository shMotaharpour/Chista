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

    farmer = list(units[0][hour]) if hour < len(units[0]) else ["PASS"]
    hands = []
    for i, unit in enumerate(units[1:]):
        if hour < len(unit):
            hands.append(list(unit[hour]))
        else:
            hands.append(["PASS"])
    market = [list(order) for order in plan.get("market", [])] \
        if hour == 0 else []
    if len(market) > MAX_MARKET_ORDERS:
        # F031: the engine drops the 11th silently - never send one
        market = market[:MAX_MARKET_ORDERS]
    return {"farmer": farmer, "hands": hands, "market": market}
