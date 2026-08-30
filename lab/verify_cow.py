"""Verify COW and SHEEP lifecycles end-to-end (ground truth from env.steps).

Claims to check:
  - COW: buy→pasture→place→feed→harvest milk (interval 2, first yield day 8)→care banking→fertilizer daily
  - SHEEP: same with wool (interval 3, first yield day 6)
  - feed = 1 wheat/day, escape after 2 unfed days
  - fertilizer 1/day per animal

Usage: python -m lab.verify3
"""
from __future__ import annotations

from kaggle_environments import make

SEED = 70


def animal_cycle(animal: str, feed: bool = True, care: bool = True, seed=SEED, days=12):
    """Run one animal lifecycle; return egg/milk/wool + fertilizer totals and final tile."""
    env = make("kaggriculture", configuration={"episodeSteps": days * 24, "seed": seed}, debug=False)
    product = {"GOOSE": "EGG", "COW": "MILK", "SHEEP": "WOOL"}[animal]

    def ag(obs):
        me = obs["farms"][0]; priv = obs["private"]
        fx, fy = me["farmer"]; tile = me["tiles"][fy][fx]
        day, hour, step = obs["day"], obs.get("hour", 0), obs["step"]
        if step == 0:
            return {"farmer": ["PASS"], "hands": [], "market": [["BUY_ANIMAL", animal, 1], ["BUY_PRODUCT", "WHEAT", 35]]}
        if step == 1:
            return {"farmer": ["PICKUP", animal, 1], "hands": [], "market": []}
        if step == 2:
            return {"farmer": ["BUILD_PASTURE"], "hands": [], "market": []}
        if step == 3:
            return {"farmer": ["PLACE", animal], "hands": [], "market": []}
        if isinstance(tile, dict) and "animal" in tile:
            if hour == 1 and priv["shed"].get("WHEAT", 0):
                return {"farmer": ["PICKUP", "WHEAT", 1], "hands": [], "market": []}
            if hour == 2 and not tile["fed_today"] and priv["inventories"][0].get("WHEAT", 0):
                return {"farmer": ["FEED"], "hands": [], "market": []}
            if hour == 3 and tile["fertilizer_available"]:
                return {"farmer": ["COLLECT_FERTILIZER"], "hands": [], "market": []}
            if hour == 4 and tile["yield_units"] > 0:
                return {"farmer": ["HARVEST"], "hands": [], "market": []}
            if hour == 5 and not tile["cared_today"] and care:
                return {"farmer": ["CARE"], "hands": [], "market": []}
        return {"farmer": ["PASS"], "hands": [], "market": []}

    env.run([ag, "pass"])
    return env


def cow_vs_care(seed=SEED):
    """Care bonus effect: cow with CARE every day vs without (10 days)."""
    def run(care_on: bool):
        env = make("kaggriculture", configuration={"episodeSteps": 12 * 24, "seed": seed}, debug=False)

        def ag(obs):
            me = obs["farms"][0]; priv = obs["private"]
            fx, fy = me["farmer"]; tile = me["tiles"][fy][fx]
            day, hour, step = obs["day"], obs.get("hour", 0), obs["step"]
            if step == 0:
                return {"farmer": ["PASS"], "hands": [], "market": [["BUY_ANIMAL", "COW", 1], ["BUY_PRODUCT", "WHEAT", 30]]}
            if step == 1:
                return {"farmer": ["PICKUP", "COW", 1], "hands": [], "market": []}
            if step == 2:
                return {"farmer": ["BUILD_PASTURE"], "hands": [], "market": []}
            if step == 3:
                return {"farmer": ["PLACE", "COW"], "hands": [], "market": []}
            if isinstance(tile, dict) and "animal" in tile:
                if hour == 1 and priv["shed"].get("WHEAT", 0):
                    return {"farmer": ["PICKUP", "WHEAT", 1], "hands": [], "market": []}
                if hour == 2 and not tile["fed_today"] and priv["inventories"][0].get("WHEAT", 0):
                    return {"farmer": ["FEED"], "hands": [], "market": []}
                if hour == 3 and tile["fertilizer_available"]:
                    return {"farmer": ["COLLECT_FERTILIZER"], "hands": [], "market": []}
                if hour == 4 and tile["yield_units"] > 0:
                    return {"farmer": ["HARVEST"], "hands": [], "market": []}
                if care_on and hour == 5 and not tile["cared_today"]:
                    return {"farmer": ["CARE"], "hands": [], "market": []}
            return {"farmer": ["PASS"], "hands": [], "market": []}

        env.run([ag, "pass"])
        o = env.steps[-1][0].observation
        milk = int(o.private["shed"].get("MILK", 0)) + sum(int(i.get("MILK", 0)) for i in o.private["inventories"])
        me = o.farms[0]
        t = me["tiles"][me["farmer"][1]][me["farmer"][0]]
        state = t if isinstance(t, str) else {k: t.get(k) for k in ("animal", "yield_units", "fed_today", "cared_today", "pending_care_bonus")}
        return milk, state

    milk_care, state_c = animal_cycle("COW")
    milk_c, _ = milk_care, _ = (animal_cycle("COW") if False else (None, None))
    # care on vs off
    env_off = make("kaggriculture", configuration={"episodeSteps": 10 * 24, "seed": seed}, debug=False)

    def ag2(obs):
        me = obs["farms"][0]; priv = obs["private"]
        fx, fy = me["farmer"]; tile = me["tiles"][fy][fx]
        day, hour, step = obs["day"], obs.get("hour", 0), obs["step"]
        if step == 0:
            return {"farmer": ["PASS"], "hands": [], "market": [["BUY_ANIMAL", "COW", 1], ["BUY_PRODUCT", "WHEAT", 30]]}
        if step == 1:
            return {"farmer": ["PICKUP", "COW", 1], "hands": [], "market": []}
        if step == 2:
            return {"farmer": ["BUILD_PASTURE"], "hands": [], "market": []}
        if step == 3:
            return {"farmer": ["PLACE", "COW"], "hands": [], "market": []}
        if isinstance(tile, dict) and "animal" in tile:
            if hour == 1 and priv["shed"].get("WHEAT", 0):
                return {"farmer": ["PICKUP", "WHEAT", 1], "hands": [], "market": []}
            if hour == 2 and not tile["fed_today"] and priv["inventories"][0].get("WHEAT", 0):
                return {"farmer": ["FEED"], "hands": [], "market": []}
            if hour == 3 and tile["fertilizer_available"]:
                return {"farmer": ["COLLECT_FERTILIZER"], "hands": [], "market": []}
            if hour == 4 and tile["yield_units"] > 0:
                return {"farmer": ["HARVEST"], "hands": [], "market": []}
            # no CARE
        return {"farmer": ["PASS"], "hands": [], "market": []}

    env_off = make("kaggriculture", configuration={"episodeSteps": 10 * 24, "seed": seed}, debug=False)
    # rerun without care by ignoring hour==5 branch — simpler: separate closure
    def ag2(obs):
        me = obs["farms"][0]; priv = obs["private"]
        fx, fy = me["farmer"]; tile = me["tiles"][fy][fx]
        day, hour, step = obs["day"], obs.get("hour", 0), obs["step"]
        if step == 0:
            return {"farmer": ["PASS"], "hands": [], "market": [["BUY_ANIMAL", "COW", 1], ["BUY_PRODUCT", "WHEAT", 30]]}
        if step == 1:
            return {"farmer": ["PICKUP", "COW", 1], "hands": [], "market": []}
        if step == 2:
            return {"farmer": ["BUILD_PASTURE"], "hands": [], "market": []}
        if step == 3:
            return {"farmer": ["PLACE", "COW"], "hands": [], "market": []}
        if isinstance(tile, dict) and "animal" in tile:
            if hour == 1 and priv["shed"].get("WHEAT", 0):
                return {"farmer": ["PICKUP", "WHEAT", 1], "hands": [], "market": []}
            if hour == 2 and not tile["fed_today"] and priv["inventories"][0].get("WHEAT", 0):
                return {"farmer": ["FEED"], "hands": [], "market": []}
            if hour == 3 and tile["fertilizer_available"]:
                return {"farmer": ["COLLECT_FERTILIZER"], "hands": [], "market": []}
            if hour == 4 and tile["yield_units"] > 0:
                return {"farmer": ["HARVEST"], "hands": [], "market": []}
        return {"farmer": ["PASS"], "hands": [], "market": []}

    env2 = env_off
    env2.run([ag2, "pass"])
    milk_no_care = int(env2.steps[-1][0].observation.private["shed"].get("MILK", 0))

    # escape test: buy goose, place, never feed
    env3 = make("kaggriculture", configuration={"episodeSteps": 6 * 24, "seed": seed}, debug=False)

    def ag3(obs):
        me = obs["farms"][0]; priv = obs["private"]
        fx, fy = me["farmer"]; tile = me["tiles"][fy][fx]
        step = obs["step"]
        if step == 0:
            return {"farmer": ["PASS"], "hands": [], "market": [["BUY_ANIMAL", "GOOSE", 1]]}
        if step == 1:
            return {"farmer": ["PICKUP", "GOOSE", 1], "hands": [], "market": []}
        if step == 2:
            return {"farmer": ["BUILD_COOP"], "hands": [], "market": []}
        if step == 3:
            return {"farmer": ["PLACE", "GOOSE"], "hands": [], "market": []}
        return {"farmer": ["PASS"], "hands": [], "market": []}

    env3 = make("kaggriculture", configuration={"episodeSteps": 6 * 24, "seed": seed}, debug=False)
    env3.run([ag3, "pass"])
    me = env3.steps[-1][0].observation.farms[0]
    t = me["tiles"][me["farmer"][1]][me["farmer"][0]]
    return {"milk_with_care": milk_care, "state_care": state_c, "milk_no_care": milk_no_care, "escape_tile": t}


def main():
    # COW with care
    print("=== COW 10-day cycle (feed+care) ===")
    env = animal_cycle("COW")
    o = env.steps[-1][0].observation
    milk = int(o.private["shed"].get("MILK", 0)) + int(o.private["inventories"][0].get("MILK", 0))
    fert = int(o.private["shed"].get("FERTILIZER", 0)) + int(o.private["inventories"][0].get("FERTILIZER", 0))
    me = o.farms[0]; t = me["tiles"][me["farmer"][1]][me["farmer"][0]]
    print(f"milk={milk} fertilizer={fert} tile={ {k: t.get(k) for k in ('animal','yield_units','fed_today','cared_today','pending_care_bonus')} if isinstance(t, dict) else t }")

    print("\n=== SHEEP 10-day cycle ===")
    env = animal_cycle("SHEEP")
    o = env.steps[-1][0].observation
    wool = int(o.private["shed"].get("WOOL", 0))
    fert = int(o.private["shed"].get("FERTILIZER", 0)) + int(o.private["inventories"][0].get("FERTILIZER", 0))
    print(f"wool={wool if (wool := wool) else wool} fert={fert}")


if __name__ == "__main__":
    main()
