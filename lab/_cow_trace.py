from kaggle_environments import make

def cycle(animal, care=True, feed=True, seed=70, days=12):
    env = make("kaggriculture", configuration={"episodeSteps": days*24, "seed": seed}, debug=False)
    product = {"GOOSE":"EGG","COW":"MILK","SHEEP":"WOOL"}[animal]
    def ag(obs):
        me = obs["farms"][0]; priv = obs["private"]
        fx, fy = me["farmer"]; tile = me["tiles"][fy][fx]
        day, hour, step = obs["day"], obs.get("hour", 0), obs["step"]
        if step == 0:
            return {"farmer": ["PASS"], "hands": [], "market": [["BUY_ANIMAL", animal, 1], ["BUY_PRODUCT", "WHEAT", 30]]}
        if step == 1:
            return {"farmer": ["PICKUP", animal, 1], "hands": [], "market": []}
        if step == 2:
            return {"farmer": ["BUILD_PASTURE"], "hands": [], "market": []}
        if step == 3:
            return {"farmer": ["PLACE", animal], "hands": [], "market": []}
        if isinstance(tile, dict) and "animal" in tile:
            if hour == 1 and priv["shed"].get("WHEAT", 0):
                return {"farmer": ["PICKUP", "WHEAT", 1], "hands": [], "market": []}
            if hour == 2 and feed and not tile["fed_today"] and priv["inventories"][0].get("WHEAT", 0):
                return {"farmer": ["FEED"], "hands": [], "market": []}
            if hour == 3 and tile["fertilizer_available"]:
                return {"farmer": ["COLLECT_FERTILIZER"], "hands": [], "market": []}
            if hour == 4 and tile["yield_units"] > 0:
                return {"farmer": ["HARVEST"], "hands": [], "market": []}
            if care and hour == 5 and not tile["cared_today"]:
                return {"farmer": ["CARE"], "hands": [], "market": []}
        return {"farmer": ["PASS"], "hands": [], "market": []}
    env.run([ag, "pass"])
    rows = []
    for d in range(days):
        step = min(d*24, len(env.steps)-1)
        o = env.steps[step][0].observation
        prod = int(o.private["shed"].get(product, 0)) + int(o.private["inventories"][0].get(product, 0))
        me = o.farms[0]; t = me["tiles"][me["farmer"][1]][me["farmer"][0]]
        y = t.get("yield_units","-") if isinstance(t,dict) else t
        cb = t.get("pending_care_bonus","-") if isinstance(t,dict) else "-"
        fed = t.get("fed_today","-") if isinstance(t,dict) else "-"
        rows.append((d, prod, y, cb, fed))
    return rows

for animal in ("COW","SHEEP"):
    print(f"--- {animal} (fed+care): (day, prod_total, tile_yield, care_bank) ---")
    for row in cycle(animal):
        print("  ", row)
print()
print("--- COW never fed (escape test) ---")
rows = cycle("COW", feed=False, care=False, days=6)
for row in rows:
    print(" ", row)
