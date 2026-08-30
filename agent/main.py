"""ChistaAgent v1 — rule-based agent for Kaggriculture.

Kaggle submission: single main.py with `agent(obs)` as the LAST function.

Design basis (all verified): docs/research/003 economics, 006 mechanics, 007 engine notes.
Layers:
  1. market_orders(): SELL tranches -> batched HIRE -> investments (land/goose/cow/seeds)
  2. build_needs(): pending tile needs with priorities (HARVEST 0 > WATER/FEED 1 > ...)
  3. decide_unit(): farmer/hands claim nearest unclaimed need (greedy by prio,dist)
"""
import json

CROPS = {
    "WHEAT":      {"seed": 10,  "first": 2,  "maxyd": 4,  "ongoing": False, "base": 25},
    "CARROT":     {"seed": 20,  "first": 2,  "maxyd": 3,  "ongoing": False, "base": 35},
    "TOMATO":     {"seed": 50,  "first": 8,  "maxyd": 8,  "ongoing": True,  "base": 60},
    "STRAWBERRY": {"seed": 100, "first": 10, "maxyd": 10, "ongoing": True,  "base": 120},
    "MELON":      {"seed": 80,  "first": 10, "maxyd": 12, "ongoing": False, "base": 250},
}
ANIMALS = {
    "GOOSE": {"cost": 300, "structure": "COOP",    "first": 4, "interval": 1, "product": "EGG"},
    "COW":   {"cost": 400, "structure": "PASTURE", "first": 8, "interval": 2, "product": "MILK"},
    "SHEEP": {"cost": 500, "structure": "PASTURE", "first": 6, "interval": 3, "product": "WOOL"},
}
STRUCT_TO_ANIMAL = {v["structure"]: k for k, v in ANIMALS.items()}
SAFE_PACE = {"WHEAT": 66, "CARROT": 16, "TOMATO": 7, "STRAWBERRY": 7, "MELON": 10,
             "EGG": 24, "MILK": 8, "WOOL": 6, "FERTILIZER": 53}
LAND_PRICES = [1000, 2000, 4000]
NO_PLANT_AFTER = 26
FORCE_SELL_DAYS = {28, 29}
SEASON_LATE = 27
MAX_HANDS = 8
HIRE_FRACTION = 0.1
FEED_DAYS = 5

FIB = [1, 1]
while len(FIB) < 16:
    FIB.append(FIB[-1] + FIB[-2])

# ================================================================ helpers
def is_plant(t):
    return isinstance(t, dict) and t.get("kind") == "PLANT"


def is_animal(t):
    return isinstance(t, dict) and "animal" in t


def tiles(obs):
    me = obs["farms"][obs["player"]]
    for y in range(len(me["tiles"])):
        for x in range(len(me["tiles"][y])):
            yield x, y, me["tiles"][y][x]


def animals(obs, animal=None):
    return [(x, y) for x, y, t in tiles(obs) if is_animal(t)
            and (animal is None or t["animal"] == animal)]


def free_tiles(obs):
    return [(x, y) for x, y, t in tiles(obs) if t is None]


def shed_count(obs, item):
    return obs["private"]["shed"].get(item, 0)


def inv_count(obs, item):
    return sum(inv.get(item, 0) for inv in obs["private"]["inventories"])


def available(obs, item):
    return shed_count(obs, item) + inv_count(obs, item)


def n_animals(obs, animal=None):
    return len(animals(obs, animal))


def game_day(obs):
    return obs["day"]


def game_hour(obs):
    return obs.get("hour", obs.get("step", 0) % 24)


def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def step_towards(pos, target):
    sx, sy = pos
    tx, ty = target
    if (sx, sy) == (tx, ty):
        return "PASS"
    dx, dy = tx - sx, ty - sy
    if dx > 0:
        return "EAST"
    if dx < 0:
        return "WEST"
    if dy > 0:
        return "SOUTH"
    return "NORTH"


def at_shed(obs, pos):
    me = obs["farms"][obs["player"]]
    half = len(me["tiles"]) // 2
    return tuple(pos) in {(half - 1, half - 1), (half, half - 1),
                          (half - 1, half), (half, half)}


def crew_capacity(obs):
    """Tiles the crew can water per day (conservative). Day 0: farmer only."""
    if obs["day"] == 0:
        return 4
    return (1 + len(obs["farms"][obs["player"]]["hands"])) * 6


def feed_reserve(obs):
    n = n_animals(obs)
    return min(FEED_DAYS * n, max(0, SEASON_LATE - obs["day"]) * n)


def fert_want(obs):
    """Fertilizer needed by plants currently inside their bonus window."""
    d = game_day(obs)
    want = 0
    for x, y, t in tiles(obs):
        if is_plant(t):
            cd = CROPS[t["crop"]]
            age = d - t["planted_day"]
            ws = (cd["maxyd"] + 1) // 2
            if ws <= age <= cd["maxyd"] and t.get("fertilized_until_day", -1) < d:
                want += 1
    return want


def empty_structures(obs, animal):
    kind = ANIMALS[animal]["structure"]
    return [(x, y) for x, y, t in tiles(obs)
            if isinstance(t, dict) and t.get("kind") == kind and "animal" not in t]


# ================================================================ crop choice
def pick_crop(obs):
    """Crop for a free tile. Melon (days 5-17: harvestable in-season); carrot early."""
    d = game_day(obs)
    seeds = obs["private"]["seeds"]
    if 5 <= d <= 17 and seeds.get("MELON", 0) > 0:
        return "MELON"
    if seeds.get("CARROT", 0) > 0 and d < 14:
        return "CARROT"
    if seeds.get("WHEAT", 0) > 0 and (n_animals(obs) > 0 or d < 4):
        return "WHEAT"
    if seeds.get("CARROT", 0) > 0:
        return "CARROT"
    return None


# ================================================================ needs
def build_needs(obs):
    """Pending needs with pos, prio (0=highest), op."""
    needs = []
    d = game_day(obs)
    priv = obs["private"]
    for x, y, t in tiles(obs):
        if t is None or t == "LOCKED":
            continue
        if is_plant(t):
            cd = CROPS[t["crop"]]
            age = d - t["planted_day"]
            if t["yield_units"] > 0 and age >= cd["first"]:
                needs.append({"pos": (x, y), "prio": 0, "op": "HARVEST"})
            if not t["watered_today"]:
                needs.append({"pos": (x, y), "prio": 1, "op": "WATER"})
            ws = (cd["maxyd"] + 1) // 2
            if ws <= age <= cd["maxyd"] and t.get("fertilized_until_day", -1) < d \
                    and available(obs, "FERTILIZER") > 0:
                needs.append({"pos": (x, y), "prio": 3, "op": "FERTILIZE"})
        elif isinstance(t, dict) and t.get("kind") == "WEED":
            needs.append({"pos": (x, y), "prio": 6, "op": "DIG"})
        elif is_animal(t):
            if t["yield_units"] > 0:
                needs.append({"pos": (x, y), "prio": 0, "op": "HARVEST"})
            if t["fertilizer_available"]:
                needs.append({"pos": (x, y), "prio": 2, "op": "COLLECT_FERTILIZER"})
            if not t["fed_today"]:
                needs.append({"pos": (x, y), "prio": 1, "op": "FEED"})
            if not t["cared_today"]:
                needs.append({"pos": (x, y), "prio": 5, "op": "CARE"})
        elif isinstance(t, dict) and t.get("kind") in ("COOP", "PASTURE") and "animal" not in t:
            a = STRUCT_TO_ANIMAL.get(t["kind"])
            if a and shed_count(obs, a) > 0:
                needs.append({"pos": (x, y), "prio": 2, "op": "PLACE", "animal": a})
    # plant needs on free tiles, capped by crew watering capacity
    if d < NO_PLANT_AFTER:
        seeds = priv["seeds"]
        planted = sum(1 for _, _, t in tiles(obs) if is_plant(t))
        room = max(0, crew_capacity(obs) - planted)
        for x, y in free_tiles(obs):
            if room <= 0:
                break
            crop = pick_crop(obs)
            if crop and seeds.get(crop, 0) > 0:
                needs.append({"pos": (x, y), "prio": 4, "op": "PLANT", "crop": crop})
                room -= 1
    return needs


# ================================================================ market
def market_orders(obs):
    """SELL tranches -> batched HIRE -> investments. Max 10 orders."""
    d = game_day(obs)
    me = obs["farms"][obs["player"]]
    money = me["money"]
    orders = []

    # 1. SELL (one batched order per product; premium drips via SAFE_PACE)
    for item, n in obs["private"]["shed"].items():
        if n <= 0 or item in ANIMALS:
            continue
        if d in FORCE_SELL_DAYS:
            orders.append(["SELL", item, n])
        else:
            sell = min(n, SAFE_PACE.get(item, 66))
            if sell > 0:
                orders.append(["SELL", item, sell])

    # 2. HIRE batch (engine spawns each hand the same day)
    load = len(build_needs(obs))
    if d < SEASON_LATE - 2:
        while True:
            hired_today = me["hires_today"] + sum(1 for o in orders if o[0] == "HIRE")
            daily_cap = 2 if d < 8 else MAX_HANDS
            if hired_today >= daily_cap or hired_today >= MAX_HANDS:
                break
            cost = FIB[min(me["hires_today"], len(FIB) - 1)]
            if cost > money * HIRE_FRACTION or load <= (1 + hired_today) * 8:
                break
            orders.append(["HIRE"])
            money -= cost

    # 3. Investments: land > goose > cow > seeds (with reserves)
    reserve = 40 + n_animals(obs) * 10
    unlocked_extra = len(me["unlocked_quadrants"]) - 1
    if 3 <= d <= 15 and money >= LAND_PRICES[unlocked_extra] + reserve:
        orders.append(["BUY_LAND"])
        money -= LAND_PRICES[unlocked_extra]
    if 3 <= d <= 8 and n_animals(obs, "GOOSE") == 0 and shed_count(obs, "GOOSE") == 0 \
            and money >= 300 + reserve and d < NO_PLANT_AFTER:
        orders.append(["BUY_ANIMAL", "GOOSE", 1])
        money -= 300
    if d >= 6 and n_animals(obs, "COW") < 1 and money >= 400 + feed_reserve(obs) and d < NO_PLANT_AFTER:
        orders.append(["BUY_ANIMAL", "COW", 1])
        money -= 400
    for crop in ("MELON", "CARROT", "WHEAT"):
        want = seed_target(obs, crop)
        have = priv_seeds(obs).get(crop, 0)
        need = max(0, want - have)
        if need > 0:
            cost = CROPS[crop]["seed"] * need
            if money >= cost + 40:
                orders.append(["BUY_SEED", crop, need])
                money -= cost
    return orders[:10]


def priv_seeds(obs):
    return obs["private"]["seeds"]


def seed_target(obs, crop):
    """Wanted seed stock for this crop."""
    d = game_day(obs)
    if d >= NO_PLANT_AFTER:
        return 0
    free = len(free_tiles(obs))
    if crop == "WHEAT":
        return min(3, free) if n_animals(obs) > 0 else 0
    if crop == "CARROT":
        return min(4, max(0, free - 1)) if d < 12 else 0
    if crop == "MELON":
        money = obs["farms"][obs["player"]]["money"]
        if 5 <= d <= 17 and money >= CROPS["MELON"]["seed"] + 100:
            return max(1, min(free, (money - 150) // CROPS["MELON"]["seed"]))
    return 0


# ================================================================ dispatcher
def decide_unit(obs, pos):
    """One unit's action: current-tile priority ladder, else move to nearest unclaimed need."""
    me = obs["farms"][obs["player"]]
    d = game_day(obs)
    fx, fy = pos
    tile = me["tiles"][fy][fx]

    # 0. HARVEST (decay = 1 unit / 2 turns once lifespan ends)
    if is_plant(tile):
        cd = CROPS[tile["crop"]]
        age = d - tile["planted_day"]
        if tile["yield_units"] > 0 and age >= cd["first"]:
            return ["HARVEST"]
    if is_animal(tile) and tile["yield_units"] > 0:
        return ["HARVEST"]

    # 1. WATER (daily, mandatory)
    if is_plant(tile) and not tile["watered_today"]:
        return ["WATER"]

    # 2. FEED hungry animal (escape = sunk loss)
    if is_animal(tile) and not tile["fed_today"] and inv_count(obs, "WHEAT") > 0:
        return ["FEED"]

    # 2b. COLLECT_FERTILIZER (1/day/animal, non-cumulative)
    if is_animal(tile) and tile["fertilizer_available"]:
        return ["COLLECT_FERTILIZER"]

    # 3. FERTILIZE inside bonus window (fertilizer in inventory)
    if is_plant(tile):
        cd = CROPS[tile["crop"]]
        age = d - tile["planted_day"]
        ws = (cd["maxyd"] + 1) // 2
        if ws <= age <= cd["maxyd"] and tile.get("fertilized_until_day", -1) < d \
                and inv_count(obs, "FERTILIZER") > 0:
            return ["FERTILIZE"]

    # 4. PLANT on empty tile / PLACE animal on empty structure
    if tile is None:
        crop = pick_crop(obs)
        if crop and priv_seeds(obs).get(crop, 0) > 0:
            return ["PLANT", crop]
    elif not isinstance(tile, str) and tile.get("kind") in ("COOP", "PASTURE") \
            and "animal" not in tile:
        a = STRUCT_TO_ANIMAL.get(tile["kind"])
        if a and obs["private"]["inventories"][0].get(a, 0):
            return ["PLACE", a]

    # 5. CARE (banks large production bonus)
    if is_animal(tile) and not tile["cared_today"]:
        return ["CARE"]

    # 6. DIG weed
    if isinstance(tile, dict) and tile.get("kind") == "WEED":
        return ["DIG"]

    # 7. PICKUP needed items if at shed (batched: 1 turn for n items)
    if at_shed(obs, (fx, fy)):
        pu = pickup_op(obs)
        if pu:
            return pu

    # 8. MOVE to nearest unclaimed need
    need = nearest_unclaimed(obs, (fx, fy))
    if need:
        return [step_towards((fx, fy), need["pos"])]
    return ["PASS"]


def nearest_unclaimed(obs, pos):
    best, best_key = None, None
    for need in build_needs(obs):
        if need["pos"] in CLAIMED:
            continue
        key = (need["prio"], manhattan(need["pos"], pos))
        if best is None or key < best_key:
            best, best_key = need, key
    if best is not None:
        CLAIMED.add(best["pos"])
    return best


def at_shed(obs, pos):
    me = obs["farms"][obs["player"]]
    half = len(me["tiles"]) // 2
    return tuple(pos) in {(half - 1, half - 1), (half, half - 1),
                          (half - 1, half), (half, half)}


def pickup_op(obs):
    """Batched pickup from shed (1 turn for n items)."""
    if fert_want(obs) > 0 and shed_count(obs, "FERTILIZER") > 0:
        return ["PICKUP", "FERTILIZER", min(3, shed_count(obs, "FERTILIZER"))]
    if n_animals(obs) > 0 and shed_count(obs, "WHEAT") > 0 and inv_count(obs, "WHEAT") == 0:
        return ["PICKUP", "WHEAT", min(3, shed_count(obs, "WHEAT"))]
    for a in ("GOOSE", "COW", "SHEEP"):
        if shed_count(obs, a) > 0 and empty_structures(obs, a):
            return ["PICKUP", a, 1]
    return None


def empty_structures(obs, animal):
    kind = ANIMALS[animal]["structure"]
    return [(x, y) for x, y, t in tiles(obs)
            if isinstance(t, dict) and t.get("kind") == kind and "animal" not in t]


# ================================================================ agent entry
CLAIMED = set()


def agent(obs):
    """Kaggle entry point — the last function defined in main.py."""
    global CLAIMED
    CLAIMED = set()
    try:
        me = obs["farms"][obs["player"]]
        market = market_orders(obs)
        farmer = decide_unit(obs, list(me["farmer"]))
        hand_actions = [decide_unit(obs, list(h)) for h in me["hands"]]
        return {"farmer": farmer, "hands": hand_actions, "market": market}
    except Exception:
        import sys, traceback
        traceback.print_exc(file=sys.stderr)
        return {"farmer": ["PASS"], "hands": [], "market": []}
