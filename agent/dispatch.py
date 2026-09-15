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

    if not units:
        # an empty plan is legitimate (every tile locked, nothing worth
        # doing): everyone passes, the market orders still ride
        market = [list(order) for order in plan.get("market", [])] \
            if hour == 0 else []
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
    market = [list(order) for order in plan.get("market", [])] \
        if hour == 0 else []
    if len(market) > MAX_MARKET_ORDERS:
        # F031: the engine drops the 11th silently - never send one
        market = market[:MAX_MARKET_ORDERS]
    # M1 simplification (named): the cap is per TURN, so a day has 240
    # order slots - M1 sends them all at hour 0. #15's sell scheduling
    # spreads sells intraday precisely because one big basket tips the
    # price a step (F036: ~25 wheat per coin).
    return {"farmer": farmer, "hands": hands, "market": market}
