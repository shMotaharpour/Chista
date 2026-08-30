"""Ground-truth verification of remaining mechanics claims (env.steps traces only).

Verified claims:
  V1  melon: first_yield_day=10 gates harvest (fert & plain both day 10)
  V2  every crop: first day units reach inventory, fert vs plain (harvest ASAP)
  V3  decay: harvest delay 0/1/2 days after ready → total loss
  V4  hire fibonacci cost + hands vanish at day end

Usage: python -m lab.verify2
"""
from __future__ import annotations

from kaggle_environments import make

SEED = 70


def total_of(env, item) -> int:
    o = env.steps[-1][0].observation
    return int(o.private["shed"].get(item, 0)) + sum(int(i.get(item, 0)) for i in o.private["inventories"])


def first_unit(env, item):
    """First (day, hour) the item appears in inventory or shed — ground truth."""
    for step in range(len(env.steps)):
        o = env.steps[step][0].observation
        got = int(o.private["inventories"][0].get(item, 0)) + int(o.private["shed"].get(item, 0))
        if got > 0:
            return (step // 24, step % 24, got)
    return None


def melon_verify():
    print("=== V1: MELON — first day units reach inventory (first_yield_day=10 gate) ===")
    for fert in (True, False):
        env = run_melon(fert)
        first = first_unit(env, "MELON")
        print(f"  fert={fert!s:<5} first unit at day/hour {first}  total={total_of(env, 'MELON')}")


def run_melon(fert: bool, seed=SEED):
    env = make("kaggriculture", configuration={"episodeSteps": 16 * 24, "seed": seed}, debug=False)

    def ag(obs):
        me = obs["farms"][0]; priv = obs["private"]
        fx, fy = me["farmer"]; tile = me["tiles"][fy][fx]
        day, hour, step = obs["day"], obs.get("hour", 0), obs["step"]
        if step == 0:
            o = [["BUY_SEED", "MELON", 1]]
            if fert:
                o.append(["BUY_PRODUCT", "FERTILIZER", 3])
            return {"farmer": ["PASS"], "hands": [], "market": o}
        if step == 1 and fert:
            return {"farmer": ["PICKUP", "FERTILIZER", 3], "hands": [], "market": []}
        if tile is None and priv["seeds"].get("MELON", 0):
            return {"farmer": ["PLANT", "MELON"], "hands": [], "market": []}
        if isinstance(tile, dict) and tile.get("kind") == "PLANT":
            age = day - tile["planted_day"]
            if not tile["watered_today"]:
                return {"farmer": ["WATER"], "hands": [], "market": []}
            if fert and age == 6 and hour == 1 and priv["shed"].get("FERTILIZER", 0):
                return {"farmer": ["PICKUP", "FERTILIZER", 3], "hands": [], "market": []}
            if fert and age == 6 and hour == 2 and priv["inventories"][0].get("FERTILIZER", 0):
                return {"farmer": ["FERTILIZE"], "hands": [], "market": []}
            if age >= 10 and hour == 1 and tile["yield_units"] > 0:
                return {"farmer": ["HARVEST"], "hands": [], "market": []}
        return {"farmer": ["PASS"], "hands": [], "market": []}

    env.run([ag, "pass"])
    return env


def crop_verify(crop: str, fert: bool, maxyd: int):
    """Harvest ASAP; report first day units actually reach inventory + total."""
    env = make("kaggriculture", configuration={"episodeSteps": (maxyd + 6) * 24, "seed": SEED}, debug=False)

    def ag(obs):
        me = obs["farms"][0]; priv = obs["private"]
        fx, fy = me["farmer"]; tile = me["tiles"][fy][fx]
        day, hour, step = obs["day"], obs.get("hour", 0), obs["step"]
        if step == 0:
            o = [["BUY_SEED", crop, 1]]
            if fert:
                o.append(["BUY_PRODUCT", "FERTILIZER", 3])
            return {"farmer": ["PASS"], "hands": [], "market": o}
        if step == 1 and fert:
            return {"farmer": ["PICKUP", "FERTILIZER", 3], "hands": [], "market": []}
        if tile is None and priv["seeds"].get(crop, 0):
            return {"farmer": ["PLANT", crop], "hands": [], "market": []}
        if isinstance(tile, dict) and tile.get("kind") == "PLANT":
            age = day - tile["planted_day"]
            if not tile["watered_today"]:
                return {"farmer": ["WATER"], "hands": [], "market": []}
            if fert and age >= (maxyd + 1) // 2 and age <= maxyd:
                if hour == 1 and priv["shed"].get("FERTILIZER", 0) and tile["fertilized_until_day"] < day:
                    return {"farmer": ["PICKUP", "FERTILIZER", 1], "hands": [], "market": []}
                if hour == 2 and priv["inventories"][0].get("FERTILIZER", 0) and tile["fertilized_until_day"] < day:
                    return {"farmer": ["FERTILIZE"], "hands": [], "market": []}
            if tile["yield_units"] > 0 and age >= 1:
                return {"farmer": ["HARVEST"], "hands": [], "market": []}
        return {"farmer": ["PASS"], "hands": [], "market": []}

    env.run([ag, "pass"])
    first = None
    for step in range(len(env.steps)):
        o = env.steps[step][0].observation
        got = int(o.private["inventories"][0].get(crop, 0)) + int(o.private["shed"].get(crop, 0))
        if got > 0:
            first = (step // 24, got)
            break
    return first, total_of(env, crop)


def all_crops():
    print("=== V2: every crop — first day units reach inventory (harvest attempted ASAP) ===")
    for crop, maxyd in [("WHEAT", 4), ("CARROT", 3), ("TOMATO", 8), ("STRAWBERRY", 10)]:
        for fert in (True, False):
            day_units = crop_verify(crop, fert)
            print(f"  {crop:<11} fert={fert!s:<5} first harvest: day {day_units[0]} ({day_units[1]} units)  total={day_units[2]}")


def crop_verify(crop: str, fert: bool, seed=SEED):
    maxyd = {"WHEAT": 4, "CARROT": 3, "TOMATO": 8, "STRAWBERRY": 10}[crop]
    env = make("kaggriculture", configuration={"episodeSteps": (maxyd + 8) * 24, "seed": seed}, debug=False)

    def ag(obs):
        me = obs["farms"][0]; priv = obs["private"]
        fx, fy = me["farmer"]; tile = me["tiles"][fy][fx]
        day, hour, step = obs["day"], obs.get("hour", 0), obs["step"]
        if step == 0:
            o = [["BUY_SEED", crop, 1]]
            if fert:
                o.append(["BUY_PRODUCT", "FERTILIZER", 3])
            return {"farmer": ["PASS"], "hands": [], "market": o}
        if step == 1 and fert:
            return {"farmer": ["PICKUP", "FERTILIZER", 3], "hands": [], "market": []}
        if tile is None and priv["seeds"].get(crop, 0):
            return {"farmer": ["PLANT", crop], "hands": [], "market": []}
        if isinstance(tile, dict) and tile.get("kind") == "PLANT":
            age = day - tile["planted_day"]
            if not tile["watered_today"]:
                return {"farmer": ["WATER"], "hands": [], "market": []}
            if fert and age >= (maxyd + 1) // 2 and age <= maxyd:
                if hour == 1 and priv["shed"].get("FERTILIZER", 0) and tile["fertilized_until_day"] < day:
                    return {"farmer": ["PICKUP", "FERTILIZER", 1], "hands": [], "market": []}
                if hour == 2 and priv["inventories"][0].get("FERTILIZER", 0) and tile["fertilized_until_day"] < day:
                    return {"farmer": ["FERTILIZE"], "hands": [], "market": []}
            if tile["yield_units"] > 0 and age >= 1:
                return {"farmer": ["HARVEST"], "hands": [], "market": []}
        return {"farmer": ["PASS"], "hands": [], "market": []}

    env.run([ag, "pass"])
    first = first_unit(env, crop)
    if first:
        return (first[0], first[1], total_of(env, crop))
    return (None, 0, 0)


def decay_verify():
    print("\n=== V3: decay — harvest delay 0/1/2 days (wheat, daily water) ===")
    for delay in (0, 1, 2):
        env = make("kaggriculture", configuration={"episodeSteps": 10 * 24, "seed": SEED}, debug=False)

        def ag(obs):
            me = obs["farms"][0]; priv = obs["private"]
            fx, fy = me["farmer"]; tile = me["tiles"][fy][fx]
            day, hour, step = obs["day"], obs.get("hour", 0), obs["step"]
            if step == 0:
                return {"farmer": ["PASS"], "hands": [], "market": [["BUY_SEED", "WHEAT", 1]]}
            if tile is None and priv["seeds"].get("WHEAT", 0):
                return {"farmer": ["PLANT", "WHEAT"], "hands": [], "market": []}
            if isinstance(tile, dict) and tile.get("kind") == "PLANT":
                age = day - tile["planted_day"]
                if not tile["watered_today"]:
                    return {"farmer": ["WATER"], "hands": [], "market": []}
                if age >= 4 + delay and tile["yield_units"] > 0:
                    return {"farmer": ["HARVEST"], "hands": [], "market": []}
            return {"farmer": ["PASS"], "hands": [], "market": []}

        env.run([ag, "pass"])
        print(f"  harvest delay={delay}d: total wheat = {total_of(env, 'WHEAT')}")


def animal_verify():
    print("\n=== V4: goose lifecycle (buy→coop→place→feed→collect fert→harvest→care) ===")
    env = make("kaggriculture", configuration={"episodeSteps": 10 * 24, "seed": SEED}, debug=False)

    def ag(obs):
        me = obs["farms"][0]; priv = obs["private"]
        fx, fy = me["farmer"]; tile = me["tiles"][fy][fx]
        day, hour, step = obs["day"], obs.get("hour", 0), obs["step"]
        if step == 0:
            return {"farmer": ["PASS"], "hands": [], "market": [["BUY_ANIMAL", "GOOSE", 1], ["BUY_PRODUCT", "WHEAT", 10]]}
        if step == 1:
            return {"farmer": ["PICKUP", "GOOSE", 1], "hands": [], "market": []}
        if step == 2:
            return {"farmer": ["BUILD_COOP"], "hands": [], "market": []}
        if step == 3:
            return {"farmer": ["PLACE", "GOOSE"], "hands": [], "market": []}
        if isinstance(tile, dict) and "animal" in tile:
            if hour == 1 and priv["shed"].get("WHEAT", 0):
                return {"farmer": ["PICKUP", "WHEAT", 1], "hands": [], "market": []}
            if hour == 2 and not tile["fed_today"] and priv["inventories"][0].get("WHEAT", 0):
                return {"farmer": ["FEED"], "hands": [], "market": []}
            if hour == 3 and tile["fertilizer_available"]:
                return {"farmer": ["COLLECT_FERTILIZER"], "hands": [], "market": []}
            if hour == 4 and tile["yield_units"] > 0:
                return {"farmer": ["HARVEST"], "hands": [], "market": []}
            if hour == 5 and not tile["cared_today"]:
                return {"farmer": ["CARE"], "hands": [], "market": []}
        return {"farmer": ["PASS"], "hands": [], "market": []}

    env.run([ag, "pass"])
    o = env.steps[-1][0].observation
    eggs = int(o.private["shed"].get("EGG", 0)) + int(o.private["inventories"][0].get("EGG", 0))
    fert = int(o.private["shed"].get("FERTILIZER", 0)) + int(o.private["inventories"][0].get("FERTILIZER", 0))
    me = o.farms[0]
    t = me["tiles"][me["farmer"][1]][me["farmer"][0]]
    state = t if isinstance(t, str) else {k: t.get(k) for k in ("kind", "animal", "yield_units", "fed_today", "cared_today", "pending_care_bonus")}
    print(f"  eggs={eggs} fertilizer={fert} final goose tile={state if (state := t) else '?'}")


def hire_verify():
    print("\n=== V5: HIRE fibonacci cost (4 hires day 0: expect 1+1+2+3=7) ===")
    env = make("kaggriculture", configuration={"episodeSteps": 2 * 24, "seed": SEED}, debug=False)

    def ag(obs):
        if obs["step"] == 0:
            return {"farmer": ["PASS"], "hands": [], "market": [["HIRE"], ["HIRE"], ["HIRE"], ["HIRE"]]}
        return {"farmer": ["PASS"], "hands": [], "market": []}

    env.run([ag, "pass"])
    o = env.steps[1][0].observation  # after turn 0 processed
    print(f"  money after 4 hires: ${o.farms[0]['money']:.0f} (expect 3000-7=2993), hands={len(o.farms[0]['hands'])}")


if __name__ == "__main__":
    melon_verify()
    print()
    for fert in (True, False):
        for crop in ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY"):
            day, units, total = crop_verify(crop, fert)
            print(f"  {crop:<11} fert={fert!s:<5} first harvest day {day} ({units} units), total={total}")
    decay_verify()
    animal_verify()
    hire_verify()
