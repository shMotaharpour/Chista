"""M1 stub brain: a greedy per-tile policy, deliberately dumb.

Exercises the spine and stands as the permanent bottom rung of the
fallback ladder. Policy per the wheat loop of docs/player_agent.md,
extended with the honest calendars (F009: wheat 6 by day 4 start, carrot
4 by day 3, melon 6 by day 10 — numbers from the engine tables, never
retyped here) and two hard rules:

- F002: a plant not watered on its planting day is a weed by the next
  night — watering a just-planted tile outranks everything else;
- F043: shed capacity is 100 across all items, overflow destroyed at the
  nightly drop — sell before the shed fills (sell at >= 80 units).

The unit acts only on the tile it stands on; movement is the caller's
plan's job in a later milestone. Everything the engine refuses is a
silent no-op (F047), so this policy is safe by construction.
"""

from __future__ import annotations

from typing import Any

# Shed sell threshold: 80 of the 100 capacity (F043). Named TODO for the
# M1 bench to revisit; chosen as "sell before the nightly drop can
# overflow" rather than invented as an optimum.
SELL_SHED_FILL = 80


def _first_plant_dict(obs) -> dict | None:
    """The standing unit's tile, if it holds a plant."""
    me = obs["farms"][obs["player"]]
    fx, fy = me["farmer"]
    tile = me["tiles"][fy][fx]
    return tile if isinstance(tile, dict) and tile.get("kind") == "PLANT" \
        else None


def greedy_action(obs) -> dict:
    """One legal action dict from the raw observation. Never raises by
    contract with the ladder (and never raises by construction: every
    branch ends in a shape-valid dict)."""
    try:
        return _greedy(obs)
    except Exception:                            # noqa: BLE001 - bottom rung
        return {"farmer": ["PASS"], "hands": [], "market": []}


def _greedy(obs) -> dict:
    if not isinstance(obs, dict) or "farms" not in obs:
        return {"farmer": ["PASS"], "hands": [], "market": []}
    me = obs["farms"][obs["player"]]
    private = obs["private"]
    money = me.get("money", 0)

    # --- market: sells first (fund everything else), then a seed buy ---
    market = []
    for item in ("WHEAT", "CARROT"):
        held = private.get("shed", {}).get(item, 0)
        if held >= SELL_SHED_FILL:
            market.append(["SELL", item, held])
    if private.get("seeds", {}).get("WHEAT", 0) == 0 and money >= 10:
        market.append(["BUY_SEED", "WHEAT", 1])
    market = market[:10]                          # F031 cap, per turn

    # --- the standing tile: plant / water / harvest ---
    tile = _first_plant_dict(obs)
    seeds = private.get("seeds", {}).get("WHEAT", 0)
    if tile is None:
        if seeds > 0:
            return {"farmer": ["PLANT", "WHEAT"], "hands": [],
                    "market": market}
        return {"farmer": ["PASS"], "hands": [], "market": market}

    day = obs.get("day", 0)
    age = day - tile.get("planted_day", day)
    # F002 first: an unwatered plant day is tomorrow's weed
    if not tile.get("watered_today", False):
        return {"farmer": ["WATER"], "hands": [], "market": market}
    # harvest from the first yield day on (WHEAT first_yield_day = 2)
    if age >= 2 and tile.get("yield_units", 0) > 0:
        return {"farmer": ["HARVEST"], "hands": [], "market": market}
    return {"farmer": ["PASS"], "hands": [], "market": market}
