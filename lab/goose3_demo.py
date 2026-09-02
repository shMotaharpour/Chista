"""Three-goose care/feeding schedule demo over a 5-day game.

Schedule (user-specified):
  goose 1: feed + CARE every day, starting day 0
  goose 2: feed + CARE every day, starting day 1
  goose 3: alternating-day FEED only (no CARE), starting day 1

Layout: coops at (3,4), (4,4), (5,4). Setup via a state machine:
  at (4,4) shed-adjacent: PICKUP goose -> BUILD_COOP -> PLACE -> EAST
  ... until 3 placed. Goose i = coop i sorted by x (i=0 leftmost).

Usage: python -m lab.goose3_demo
"""
from __future__ import annotations

import json

from kaggle_environments import make

SEED = 70
DAYS = 5


def run_goose3(seed: int = SEED, days: int = DAYS):
    env = make("kaggriculture",
               configuration={"episodeSteps": days * 24, "seed": seed}, debug=False)

    def ag(obs):
        me = obs["farms"][0]
        priv = obs["private"]
        fx, fy = me["farmer"]
        tile = me["tiles"][fy][fx]
        day, hour, step = obs["day"], obs.get("hour", 0), obs["step"]

        if step == 0:
            return {"farmer": ["PASS"], "hands": [], "market":
                    [["BUY_ANIMAL", "GOOSE", 3], ["BUY_PRODUCT", "WHEAT", 20]]}

        # ---- setup: place geese one at a time via BUILD->PLACE chain ----
        n_placed = sum(1 for row in me["tiles"] for t in row
                       if isinstance(t, dict) and t.get("kind") == "COOP" and t.get("animal"))
        in_hand = priv["inventories"][0].get("GOOSE", 0)
        if n_placed < 3 and day == 0:
            if in_hand:
                if isinstance(tile, dict) and tile.get("kind") == "COOP" and not tile.get("animal"):
                    return {"farmer": ["PLACE", "GOOSE"], "hands": [], "market": []}
                if tile is None:
                    return {"farmer": ["BUILD_COOP"], "hands": [], "market": []}
                # standing on own coop with goose in hand: head to next NW site
                return {"farmer": ["WEST" if fx > 2 else "NORTH"], "hands": [], "market": []}
            # no goose in hand: go to (4,4) and pick up
            if (fx, fy) != (4, 4):
                if fy != 4:
                    return {"farmer": ["NORTH" if fy > 4 else "SOUTH"], "hands": [], "market": []}
                return {"farmer": ["EAST" if fx < 4 else "WEST"], "hands": [], "market": []}
            return {"farmer": ["PICKUP", "GOOSE", 1], "hands": [], "market": []}

        # ---- daily care routine ----
        coops = sorted(
            [(x, y, t) for y, row in enumerate(me["tiles"]) for x, t in enumerate(row)
             if isinstance(t, dict) and t.get("kind") == "COOP" and t.get("animal")],
            key=lambda c: c[0])
        if len(coops) < 3:
            # setup unfinished (day > 0 shouldn't happen, but be safe)
            return {"farmer": ["PASS"], "hands": [], "market": []}

        here = next(((x, y, t) for x, y, t in coops if (x, y) == (fx, fy)), None)

        def care_due(i, d):   # goose1 (i=0) from day0; goose2 (i=1) from day1; goose3 never
            return i == 0 or (i == 1 and d >= 1)

        def feed_due(i, d):
            return care_due(i, d) or (i == 2 and d >= 1 and d % 2 == 1)

        # stock wheat: any feed pending today and hands empty -> grab 3 from shed (center tiles)
        any_feed_pending = any(feed_due(i, day) for i in range(len(coops)))
        if any_feed_pending and not priv["inventories"][0].get("WHEAT", 0) \
                and priv["shed"].get("WHEAT", 0) and (fx, fy) == (4, 4):
            return {"farmer": ["PICKUP", "WHEAT", 3], "hands": [], "market": []}

        if here:
            i = coops.index(here)
            t = here[2]
            if t["yield_units"] > 0:
                return {"farmer": ["HARVEST"], "hands": [], "market": []}
            if feed_due(i, day) and not t["fed_today"]:
                if priv["inventories"][0].get("WHEAT", 0):
                    return {"farmer": ["FEED"], "hands": [], "market": []}
                # PICKUP only works shed-adjacent: go to (4,4) first
                if (fx, fy) != (4, 4) and priv["shed"].get("WHEAT", 0):
                    if fy != 4:
                        return {"farmer": ["NORTH" if fy > 4 else "SOUTH"], "hands": [], "market": []}
                    return {"farmer": ["EAST" if fx < 4 else "WEST"], "hands": [], "market": []}
                if priv["shed"].get("WHEAT", 0):
                    return {"farmer": ["PICKUP", "WHEAT", 3], "hands": [], "market": []}
            if care_due(i, day) and not t["cared_today"]:
                return {"farmer": ["CARE"], "hands": [], "market": []}
            if t["yield_units"] > 0:
                return {"farmer": ["HARVEST"], "hands": [], "market": []}

        # move to nearest coop with pending feed/care
        targets = [c for i, c in enumerate(coops)
                   if (feed_due(i, day) and not c[2]["fed_today"])
                   or (care_due(i, day) and not c[2]["cared_today"])]
        if targets:
            tx, ty, _ = min(targets, key=lambda c: abs(c[0]-fx) + abs(c[1]-fy))
            if ty != fy:
                return {"farmer": ["SOUTH" if ty > fy else "NORTH"], "hands": [], "market": []}
            if tx != fx:
                return {"farmer": ["EAST" if tx > fx else "WEST"], "hands": [], "market": []}

        # harvest sweep: coops with eggs
        egg_targets = [c for c in coops if c[2]["yield_units"] > 0]
        if egg_targets:
            tx, ty, _ = min(egg_targets, key=lambda c: abs(c[0]-fx) + abs(c[1]-fy))
            if (tx, ty) != (fx, fy):
                if ty != fy:
                    return {"farmer": ["SOUTH" if ty > fy else "NORTH"], "hands": [], "market": []}
                return {"farmer": ["EAST" if tx > fx else "WEST"], "hands": [], "market": []}

        # end of day: sell eggs
        if hour == 23 and priv["shed"].get("EGG", 0):
            return {"farmer": ["PASS"], "hands": [],
                    "market": [["SELL", "EGG", priv["shed"]["EGG"]]]}
        return {"farmer": ["PASS"], "hands": [], "market": []}

    env.run([ag, "pass"])
    return env


def summarize(env, days=DAYS):
    print(f"=== goose3 demo ({days} days) ===")
    for d in range(days):
        step = min((d + 1) * 24 - 1, len(env.steps) - 1)
        o = env.steps[step][0].observation
        me = o.farms[0]
        shed = o.private["shed"]
        coops = sorted([(x, y, t) for y, row in enumerate(me["tiles"])
                        for x, t in enumerate(row)
                        if isinstance(t, dict) and t.get("kind") == "COOP" and t.get("animal")],
                       key=lambda c: c[0])
        row = f"day {d}: eggs_shed={shed.get('EGG', 0)} money=${me['money']:.0f} | "
        row += " ".join(
            f"g{i+1}[yield={t.get('yield_units')},care_bank={t.get('pending_care_bonus')},"
            f"fed={int(bool(t.get('fed_today')))},cared={int(bool(t.get('cared_today')))}]"
            for i, (x, y, t) in enumerate(coops))
        print(row)


if __name__ == "__main__":
    env = run_goose3()
    summarize(env)
    with open("lab/results/goose3_replay.json", "w") as f:
        json.dump(env.toJSON(), f)
    print("replay json -> lab/results/goose3_replay.json")
