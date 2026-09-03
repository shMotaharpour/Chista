"""Role-book agent: reads per-day tasks from playbook.json and executes them.

Pattern proven against Thomas_Tschinkel's 668-episode replay analysis:
- Day 0: 12 melon + 7 wheat + 2 cow + 2 sheep + 4 pastures
- Daily animal growth to ~9 cow / ~6 sheep by day 11
- Strawberry ramps day 5-11 to ~36, wheat mid-season backbone, melon front-loaded
- Land expansion on days 4,6,8,12 (matching _expansion_days.csv)
- Fertilize strawberry (day 14+) and wheat (day 13+)
- Hire 4 hands/day

Usage (inside agent/main.py):
    from agent.playbook_agent import agent
"""
from __future__ import annotations

import json
from pathlib import Path

_PB_PATH = Path(__file__).parent / "playbook.json"
_PB = json.loads(_PB_PATH.read_text())
_ANIMALS = _PB["animal_targets"]
_CROPS = _PB["daily_targets"]
_SEED_PRIORITY = ["MELON", "WHEAT", "STRAWBERRY", "CARROT"]
# one-time crops: harvest at/after max_yield_day (per README yield curve)
_MAX_YIELD_DAY = {"WHEAT": 4, "CARROT": 3, "MELON": 12}
_ANIMAL_STRUCTURE = {"GOOSE": "COOP", "COW": "PASTURE", "SHEEP": "PASTURE"}


def _ramp_target(ramp, day, start_day, start_qty):
    if day < start_day:
        return 0
    qty = start_qty
    for ramp_day, target in ramp:
        if day >= ramp_day:
            qty = target
    return qty


def _count(me, pred):
    return sum(1 for row in me["tiles"] for t in row if isinstance(t, dict) and pred(t))


def agent(obs):
    me = obs["farms"][obs["player"]]
    priv = obs["private"]
    day, hour = obs["day"], obs.get("hour", 0)
    fx, fy = me["farmer"]
    tile = me["tiles"][fy][fx]

    market = []

    # ---- land expansion on playbook days ----
    if str(day) in _PB["expansion_days"] and len(me["unlocked_quadrants"]) < 4:
        cost = {1: 1000, 2: 2000, 3: 4000}[len(me["unlocked_quadrants"])]
        if me["money"] >= cost:
            market.append(["BUY_LAND"])

    # ---- hire hands (fixed rate from playbook) ----
    if me.get("hires_today", 0) < _PB.get("hire_per_day", 0) and me["money"] > 500:
        market.append(["HIRE"])

    # ---- animal purchases to reach per-day stock ----
    for animal, spec in _ANIMALS.items():
        target = max((qty for buy_day, qty in spec["buy"] if day >= buy_day), default=0)
        have = _count(me, lambda t: t.get("animal") == animal)
        if have < target:
            n = min(target - have, me["money"] // 500)
            if n > 0:
                market.append(["BUY_ANIMAL", animal, int(n)])

    # ---- seed purchases for daily crop targets ----
    for crop, spec in _CROPS.items():
        target = _ramp_target(spec.get("ramp", []), day, spec["start_day"], spec["start_qty"])
        planted = _count(me, lambda t: t.get("kind") == "PLANT" and t.get("crop") == crop)
        want = target - planted
        need = want - priv["seeds"].get(crop, 0)
        if need > 0:
            price = {"WHEAT": 10, "CARROT": 20, "TOMATO": 50, "STRAWBERRY": 100, "MELON": 80}[crop]
            n = min(need, me["money"] // max(price, 1))
            if n > 0:
                market.append(["BUY_SEED", crop, int(n)])

    # ---- end-of-day sales (produce only — never animals/wheat) ----
    if hour == 23:
        for item, n in priv["shed"].items():
            if n > 0 and item in ("CARROT", "TOMATO", "STRAWBERRY", "MELON",
                                  "EGG", "MILK", "WOOL", "FERTILIZER"):
                market.append(["SELL", item, n])

    # ================= farmer fieldwork =================
    in_hand_animal = next((a for a, n in priv["inventories"][0].items()
                           if a in ("GOOSE", "COW", "SHEEP") and n), None)
    inv_wheat = priv["inventories"][0].get("WHEAT", 0)
    at_shed = (fx, fy) in ((4, 4), (5, 4), (4, 5), (5, 5))

    # ================= hands fieldwork (computed for every return path) =================
    def _hands_ops():
        hands_out = []
        for i, hpos in enumerate(me.get("hands", [])):
            hx, hy = hpos[0], hpos[1]
            htile = me["tiles"][hy][hx]
            op = ["PASS"]
            if isinstance(htile, dict) and htile.get("kind") == "PLANT":
                crop = htile.get("crop")
                age = day - htile.get("planted_day", day)
                if not htile["watered_today"]:
                    op = ["WATER"]
                elif crop in _MAX_YIELD_DAY and age >= _MAX_YIELD_DAY[crop]:
                    op = ["HARVEST"]
                elif crop not in _MAX_YIELD_DAY and htile["yield_units"] > 0:
                    op = ["HARVEST"]
            elif isinstance(htile, dict) and htile.get("kind") in ("COOP", "PASTURE") \
                    and htile.get("animal"):
                if htile.get("yield_units", 0) > 0:
                    op = ["HARVEST"]
                elif not htile["fed_today"] and priv["inventories"][i + 1].get("WHEAT", 0):
                    op = ["FEED"]
                elif not htile["cared_today"]:
                    op = ["CARE"]
                elif htile.get("fertilizer_available"):
                    op = ["COLLECT_FERTILIZER"]
            elif htile is None:
                for crop in _SEED_PRIORITY:
                    if priv["seeds"].get(crop, 0) > 0:
                        op = ["PLANT", crop]
                        break
            if op == ["PASS"]:
                goal, best = None, 10 ** 9
                for y, row in enumerate(me["tiles"]):
                    for x, t in enumerate(row):
                        d = abs(x - hx) + abs(y - hy)
                        if isinstance(t, dict) and t.get("kind") == "PLANT" \
                                and not t["watered_today"] and d < best:
                            goal, best = (x, y), d
                        elif isinstance(t, dict) and t.get("kind") in ("COOP", "PASTURE") \
                                and t.get("animal") and t.get("yield_units", 0) > 0 and d < best:
                            goal, best = (x, y), d
                        elif t is None and d < best and any(
                                priv["seeds"].get(c, 0) > 0 for c in _SEED_PRIORITY):
                            goal, best = (x, y), d
                if goal:
                    gx, gy = goal
                    if gy != hy:
                        op = ["SOUTH" if gy > hy else "NORTH"]
                    elif gx != hx:
                        op = ["EAST" if gx > hx else "WEST"]
            hands_out.append(op)
        return hands_out


    # holding an animal: build on empty tile or place on empty structure
    if in_hand_animal:
        if isinstance(tile, dict) and tile.get("kind") in ("COOP", "PASTURE") \
                and not tile.get("animal"):
            return {"farmer": ["PLACE", in_hand_animal], "hands": _hands_ops(), "market": market}
        if tile is None:
            kind = _ANIMAL_STRUCTURE.get(in_hand_animal)
            if kind:
                return {"farmer": ["BUILD_" + kind], "hands": _hands_ops(), "market": market}
        # walk toward nearest EMPTY unlocked tile (never pass one up)
        lim = 4 * len(me["unlocked_quadrants"])
        empty = [(x, y) for y, row in enumerate(me["tiles"]) for x, t in enumerate(row)
                 if t is None and x < lim and y < lim]
        if empty:
            tx, ty = min(empty, key=lambda p: (abs(p[0]-fx)+abs(p[1]-fy), p[1], p[0]))
            if ty != fy:
                return {"farmer": ["SOUTH" if ty > fy else "NORTH"], "hands": _hands_ops(), "market": market}
            if tx != fx:
                return {"farmer": ["EAST" if tx > fx else "WEST"], "hands": _hands_ops(), "market": market}

    # standing on a plant
    if isinstance(tile, dict) and tile.get("kind") == "PLANT":
        crop = tile.get("crop")
        age = day - tile.get("planted_day", day)
        mature = tile["yield_units"] > 0 and (
            crop not in _MAX_YIELD_DAY or age >= _MAX_YIELD_DAY[crop])
        if crop in _MAX_YIELD_DAY and age >= _MAX_YIELD_DAY[crop]:
            return {"farmer": ["HARVEST"], "hands": _hands_ops(), "market": market}
        if crop in _MAX_YIELD_DAY and tile["yield_units"] > 0 and age >= 2 and crop == "WHEAT":
            return {"farmer": ["HARVEST"], "hands": _hands_ops(), "market": market}
        if not tile["watered_today"]:
            return {"farmer": ["WATER"], "hands": _hands_ops(), "market": market}
        # fertilize on playbook days
        fert_day = any(spec.get("fert_day") == day for spec in _CROPS.values())
        if fert_day and tile.get("fertilized_until_day", -1) < day:
            return {"farmer": ["FERTILIZE"], "hands": _hands_ops(), "market": market}

    # standing on an animal structure
    if isinstance(tile, dict) and tile.get("kind") in ("COOP", "PASTURE") and tile.get("animal"):
        if not tile["fed_today"] and priv["shed"].get("WHEAT", 0):
            if inv_wheat:
                return {"farmer": ["FEED"], "hands": _hands_ops(), "market": market}
            if at_shed:
                return {"farmer": ["PICKUP", "WHEAT", 3], "hands": _hands_ops(), "market": market}
        if not tile["cared_today"]:
            return {"farmer": ["CARE"], "hands": _hands_ops(), "market": market}
        if tile["yield_units"] > 0:
            return {"farmer": ["HARVEST"], "hands": _hands_ops(), "market": market}

    # empty tile + seeds in hand: plant by priority
    if tile is None:
        for crop in _SEED_PRIORITY:
            if priv["seeds"].get(crop, 0) > 0:
                return {"farmer": ["PLANT", crop], "hands": _hands_ops(), "market": market}

    # collect fertilizer if standing on structure with any
    if isinstance(tile, dict) and tile.get("kind") in ("COOP", "PASTURE") \
            and tile.get("fertilizer_available"):
        return {"farmer": ["COLLECT_FERTILIZER"], "hands": _hands_ops(), "market": market}

    # pickup animals from shed if any waiting and structures available
    shed_animal = next((a for a, n in priv["shed"].items() if a in ("GOOSE", "COW", "SHEEP") and n), None)
    if shed_animal and at_shed:
        return {"farmer": ["PICKUP", shed_animal, 1], "hands": _hands_ops(), "market": market}
    if shed_animal:
        if fy != 4:
            return {"farmer": ["NORTH" if fy > 4 else "SOUTH"], "hands": _hands_ops(), "market": market}
        return {"farmer": ["EAST" if fx < 4 else "WEST"], "hands": _hands_ops(), "market": market}

    _tg = [(x,y) for y,row in enumerate(me["tiles"]) for x,t in enumerate(row)
           if isinstance(t,dict) and t.get("kind")=="PLANT" and (t["yield_units"]>0 or not t["watered_today"])]
    # walk to nearest tile needing water/feed/harvest
    targets = []
    for y, row in enumerate(me["tiles"]):
        for x, t in enumerate(row):
            if not isinstance(t, dict):
                continue
            if t.get("kind") == "PLANT" and (t["yield_units"] > 0 or not t["watered_today"]):
                targets.append((x, y))
            elif t.get("kind") in ("COOP", "PASTURE") and t.get("animal") and \
                    (not t["fed_today"] or not t["cared_today"] or t["yield_units"] > 0):
                targets.append((x, y))
    targets = [p for p in targets if p != (fx, fy)]
    if targets:
        tx, ty = min(targets, key=lambda p: (abs(p[0] - fx) + abs(p[1] - fy), p[1], p[0]))
        if ty != fy:
            return {"farmer": ["SOUTH" if ty > fy else "NORTH"], "hands": _hands_ops(), "market": market}
        if tx != fx:
            return {"farmer": ["EAST" if tx > fx else "WEST"], "hands": _hands_ops(), "market": market}

    # planting run (only when nothing to care for): go plant seeds on empty tiles
    has_seeds = any(priv["seeds"].get(c, 0) > 0 for c in _SEED_PRIORITY)
    pending_care = any(
        isinstance(t, dict) and (
            (t.get("kind") == "PLANT" and not t["watered_today"])
            or (t.get("kind") in ("COOP", "PASTURE") and t.get("animal")
                and (not t["fed_today"] or not t["cared_today"])))
        for row in me["tiles"] for t in row)
    if has_seeds and not pending_care:
        if tile is None:
            for crop in _SEED_PRIORITY:
                if priv["seeds"].get(crop, 0) > 0:
                    return {"farmer": ["PLANT", crop], "hands": _hands_ops(), "market": market}
        lim = 4 * len(me["unlocked_quadrants"])
        empties_nw = [(x, y) for y, row in enumerate(me["tiles"]) for x, t in enumerate(row)
                      if t is None and x < lim and y < lim]
        if empties_nw:
            tx, ty = min(empties_nw,
                         key=lambda p: (abs(p[0] - fx) + abs(p[1] - fy), p[1], p[0]))
            if ty != fy:
                return {"farmer": ["SOUTH" if ty > fy else "NORTH"], "hands": _hands_ops(), "market": market}
            if tx != fx:
                return {"farmer": ["EAST" if tx > fx else "WEST"], "hands": _hands_ops(), "market": market}

    return {"farmer": ["PASS"], "hands": _hands_ops(), "market": market}

