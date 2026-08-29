"""Mechanics verification experiments — chain-effect checks against the live env.

Each experiment returns facts, not assumptions.
Run: python -m lab.verify_mechanics
"""
from __future__ import annotations

import json
from kaggle_environments import make

def run_env(steps_agent, steps=720, seed=123, opp="pass"):
    env = make("kaggriculture", configuration={"episodeSteps": steps, "seed": seed}, debug=False)
    env.run([steps_agent, opp])
    return env

# ---------------------------------------------------------------- EXP 1
def exp_fertilizer_melon():
    """Does FERTILIZE really raise melon yield in the live env?"""
    results = {}
    for fert in (True, False):
        def ag(obs):
            me = obs["farms"][obs["player"]]
            priv = obs["private"]
            fx, fy = me["farmer"]; tile = me["tiles"][fy][fx]
            day = obs["day"]
            market = []
            if obs.get("step", 0) == 0:
                market = [["BUY_SEED", "MELON", 1]]
            if fert and day in (5, 6, 7) and obs.get("hour", 0) == 0:
                # hold fertilizer: buy on day 5 via BUY_PRODUCT
                pass
            if tile is None and priv["seeds"].get("MELON", 0) and day == 0:
                return {"farmer": ["PLANT", "MELON"], "hands": [], "market": market}
            if isinstance(tile, dict) and tile.get("kind") == "PLANT":
                if not tile["watered_today"]:
                    return {"farmer": ["WATER"], "hands": [], "market": market}
                if fert and not obs.get("hour", 0):
                    # buy fertilizer day 4, apply day 5-7 window
                    if day == 4 and obs.get("hour", 0) == 0:
                        return {"farmer": ["WATER"], "hands": [], "market": [["BUY_PRODUCT", "FERTILIZER", 1]]}
                    if day in (5, 6, 7) and priv["inventories"][0].get("FERTILIZER"):
                        return {"farmer": ["FERTILIZE"], "hands": [], "market": market}
            if day >= 12 and obs.get("hour", 0) == 0 and isinstance(tile, dict) and tile.get("kind") == "PLANT" and tile["crop"] == "MELON":
                return {"farmer": ["HARVEST"], "hands": [], "market": market}
            return {"farmer": ["PASS"], "hands": [], "market": market}
        env = run_env(ag, steps=16 * 24, seed=55)
        o = env.steps[-1][0].observation
        inv = o.private["shed"].get("MELON", 0) + sum(i.get("MELON", 0) for i in o.private["inventories"])
        tile = o.farms[0]["tiles"][o.farms[0]["farmer"][1]][o.farms[0]["farmer"][0]]
        on_tile = tile.get("yield_units", 0) if isinstance(tile, dict) else 0
        results[fert] = inv + on_tile
    return {"fertilized_melon_yield": results[True], "unfertilized_melon_yield": results[False]}

# ---------------------------------------------------------------- EXP 2
def exp_watering_schedule():
    """Is daily watering necessary? Compare: daily vs skip-after-window vs never."""
    def make_ag(water_until_day):
        def ag(obs):
            me = obs["farms"][obs["player"]]
            priv = obs["private"]
            fx, fy = me["farmer"]; tile = me["tiles"][fy][fx]
            day = obs["day"]
            market = [["BUY_SEED", "WHEAT", 1]] if obs.get("step", 0) == 0 else []
            if tile is None and priv["seeds"].get("WHEAT", 0) and day == 0:
                return {"farmer": ["PLANT", "WHEAT"], "hands": [], "market": market}
            if isinstance(tile, dict) and tile.get("kind") == "PLANT":
                if day <= water_until_day and not tile["watered_today"]:
                    return {"farmer": ["WATER"], "hands": [], "market": market}
                if day >= 4 and tile["crop"] == "WHEAT" and tile["yield_units"] > 0:
                    return {"farmer": ["HARVEST"], "hands": [], "market": market}
            return {"farmer": ["PASS"], "hands": [], "market": market}
        return ag
    out = {}
    for label, until in [("daily(4d)", 4), ("stop_day2", 2), ("never", -1)]:
        env = run_env(make_ag(until), steps=10 * 24, seed=66)
        o = env.steps[-1][0].observation
        got = o.private["shed"].get("WHEAT", 0) + sum(i.get("WHEAT", 0) for i in o.private["inventories"])
        tile = o.farms[0]["tiles"][o.farms[0]["farmer"][1]][o.farms[0]["farmer"][0]]
        on_tile = tile.get("yield_units", 0) if isinstance(tile, dict) else 0
        kind = tile.get("kind") if isinstance(tile, dict) else tile
        out[label] = {"harvested": got, "on_tile": on_tile, "tile_kind": kind}
    return out

# ---------------------------------------------------------------- EXP 3
def exp_sell_animal():
    """Can we sell an animal from the shed?"""
    def ag(obs):
        priv = obs["private"]
        market = []
        if obs.get("step", 0) == 0:
            market = [["BUY_ANIMAL", "GOOSE", 1]]
        if obs.get("step", 0) == 2:
            market = [["SELL", "GOOSE", 1]]
        return {"farmer": ["PASS"], "hands": [], "market": market}
    env = run_env(ag, steps=8 * 24, seed=77)
    o = env.steps[-1][0].observation
    money_gain = o.farms[0]["money"] - 3000
    goose_left = o.private["shed"].get("GOOSE", 0)
    return {"money_delta": money_gain, "goose_in_shed": goose_left,
            "note": "money_delta == -300 → sell rejected (animals not sellable via SELL)"}

# ---------------------------------------------------------------- EXP 4
def exp_shed_overflow():
    """Fill shed beyond 100 → what order do items get lost?"""
    def ag(obs):
        step = obs.get("step", 0)
        market = []
        if step == 0:
            market = [["BUY_PRODUCT", "WHEAT", 80], ["BUY_PRODUCT", "FERTILIZER", 80]]
        # each unit sold at floor? no - try PLACE from inventory into shed repeatedly? 
        # Instead: buy 80+80 = 160 items > 100 cap. They land in shed via BUY_PRODUCT.
        return {"farmer": ["PASS"], "hands": [], "market": market}
    env = run_env(ag, steps=6 * 24, seed=88)
    o = env.steps[-1][0].observation
    shed = dict(o.private["shed"])
    return {"shed": {k: v for k, v in shed.items() if v > 0},
            "total": sum(shed.values()),
            "note": "BUY_PRODUCT respects shedCapacity — shows which buys were rejected"}

# ---------------------------------------------------------------- EXP 5
def exp_simultaneous_sell():
    """Both players SELL same item same turn → lockstep pricing."""
    def ag(obs):
        if obs.get("step", 0) == 0:
            return {"farmer": ["PASS"], "hands": [], "market": [["BUY_PRODUCT", "WHEAT", 500]]}
        if obs.get("step", 0) == 2:
            return {"farmer": ["PASS"], "hands": [], "market": [["SELL", "WHEAT", 500]]}
        return {"farmer": ["PASS"], "hands": [], "market": []}
    env = run_env(ag, steps=10 * 24, seed=99, opp=ag)  # both dump 500 wheat
    p0 = env.steps[3][0].observation.market["prices"]["WHEAT"]
    inv = env.steps[3][0].observation.market["inventory"]["WHEAT"]
    return {"both_sell_500_wheat": {"price_after": p0, "inventory_after": inv, "expected_shift": 1000}}

# ---------------------------------------------------------------- EXP 6
def exp_wheat_arbitrage():
    """Buy low / sell high wheat+fertilizer: does spread survive the round trip?"""
    def ag(obs):
        step = obs.get("step", 0)
        priv = obs["private"]
        market = []
        if step == 0:
            market = [["BUY_PRODUCT", "WHEAT", 300]]  # drain inventory → price up
        if step in (2, 3, 4):
            # sell back: price should be higher now (scarcity)
            n = priv["shed"].get("WHEAT", 0)
            if n:
                market = [["SELL", "WHEAT", min(n, 100)]]
        return {"farmer": ["PASS"], "hands": [], "market": market}
    env = run_env(ag, steps=12 * 24, seed=111)
    money = [env.steps[i][0].observation.farms[0]["money"] for i in (1, 2, 5, 6, 8, 11)]
    prices = [env.steps[i][0].observation.market["prices"]["WHEAT"] for i in (1, 2, 5, 6, 8, 11)]
    return {"money_path": money, "wheat_prices": prices,
            "note": "if money rises after selling back at higher price → arbitrage works"}

if __name__ == "__main__":
    out = {}
    print("EXP1 fertilizer on melon..."); out["fertilizer_melon"] = exp_fertilizer_melon(); print(out["fertilizer_melon"])
    print("EXP2 watering schedule..."); out["watering"] = exp_watering_schedule(); print(out["watering"])
    print("EXP3 sell animal..."); out["sell_animal"] = exp_sell_animal(); print(out["sell_animal"])
    print("EXP4 shed overflow..."); out["shed_overflow"] = exp_shed_overflow(); print(out["shed_overflow"])
    print("EXP5 simultaneous sell..."); out["simultaneous"] = exp_simultaneous_sell(); print(out["simultaneous"])
    print("EXP6 wheat arbitrage..."); out["arbitrage"] = exp_wheat_arbitrage(); print(out["arbitrage"])
    with open("lab/results/mechanics_verification.json", "w") as f:
        json.dump(out, f, indent=2)
    print("saved → lab/results/mechanics_verification.json")
