"""ChistaAgent v1.2 — rule-based agent, proven-constants edition.

Kaggle submission: single main.py with `agent(obs)` as the LAST function.
Every constant is empirically proven (docs/replays Analysis/000-003, 006, 007).

Layout model (003 tile modes):
  - wheat backbone on most tiles (feed + sell; matures in 2 days)
  - strawberry cluster in EAST columns (x>=8) — ongoing income
  - pasture pair at (4,2)/(5,2) — center row
  - planting NEVER stops: wheat to the last day (harvest day+2, sell day+2)

Market model (001/002):
  - accumulate then DUMP (no drip); forced full sale days 28-30
  - wheat BUY_PRODUCT to feed animals (6207/6269 winners do this)
  - land: NE at day 6, SW at day 11 (universal rule)
  - shops: no measurable influence — ignore
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
FORCE_SELL_DAYS = {28, 29}
SEASON_LATE = 27
MAX_HANDS = 8
HIRE_FRACTION = 0.1
NO_PLANT_AFTER = 26
NO_STRAWBERRY_BEFORE = 3
NO_STRAWBERRY_AFTER = 20

# proven parameters (sweep + replay analysis)
DAY0_TILES = 16
CREW_RATE = 6
LAND_DAYS = [6, 11]            # NE@day6, SW@day11 (universal winner schedule)
MELON_START_DAY = 3
BUY_WHEAT_NOT_PLANT = True     # winners buy wheat 3.4M units; don't waste tiles on feed

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


def tile_crop_for(x, y, d):
    """Proven spatial layout (docs/replays Analysis/003 §4):
    strawberry in east columns (x>=8), wheat everywhere else."""
    if x >= 8:
        return "STRAWBERRY"
    return "WHEAT"


# ------------------------------------------------------------------ crop choice
def crew_capacity(obs):
    """Tiles the crew can water per day (5/unit/day, walking included)."""
    if game_day(obs) == 0:
        return DAY0_TILES
    return (1 + len(obs["farms"][obs["player"]]["hands"])) * CREW_RATE


def feed_reserve(obs):
    n = n_animals(obs)
    return min(5 * n, max(0, SEASON_LATE - game_day(obs)) * n)


def fert_want(obs):
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


def empty_structures(obs, animal):
    kind = ANIMALS[animal]["structure"]
    return [(x, y) for x, y, t in tiles(obs)
            if isinstance(t, dict) and t.get("kind") == kind and "animal" not in t]


# ================================================================ needs
def build_needs(obs):
    """All pending needs: {pos, prio (0=highest), op, ...}."""
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
    # planting needs (capped by crew capacity, spatial layout aware)
    seeds = priv["seeds"]
    planted = sum(1 for _, _, t in tiles(obs) if is_plant(t))
    room = max(0, crew_capacity(obs) - planted)
    for x, y in free_tiles(obs):
        if room <= 0:
            break
        crop = pick_crop(obs, x, y)
        if crop and seeds.get(crop, 0) > 0:
            needs.append({"pos": (x, y), "prio": 4, "op": "PLANT", "crop": crop})
            room -= 1
    return needs


def pick_crop(obs, x=None, y=None):
    """Melon-core: melon on free tiles days 3-17 (money crop), wheat backbone."""
    d = game_day(obs)
    seeds = obs["private"]["seeds"]
    if MELON_START_DAY <= d <= 17 and seeds.get("MELON", 0) > 0:
        return "MELON"
    if seeds.get("WHEAT", 0) > 0:
        return "WHEAT"
    if seeds.get("CARROT", 0) > 0 and d < 10:
        return "CARROT"
    return None


def animals(obs, animal=None):
    return [(x, y) for x, y, t in tiles(obs)
            if is_animal(t) and (animal is None or t["animal"] == animal)]


def free_tiles(obs):
    return [(x, y) for x, y, t in tiles(obs) if t is None]


# ================================================================ market layer
def market_orders(obs):
    """SELL (accumulate-dump model) -> batched HIRE -> investments."""
    d = game_day(obs)
    me = obs["farms"][obs["player"]]
    money = me["money"]
    orders = []

    # 1. SELL: winners accumulate & dump. Faster pace late-season (26% of
    # winner revenue lands in days 25-30); forced full sale days 29-30.
    for item, n in obs["private"]["shed"].items():
        if n <= 0 or item in ANIMALS:
            continue
        if d in FORCE_SELL_DAYS:
            orders.append(["SELL", item, n])
        else:
            pace = SAFE_PACE.get(item, 66) * (2 if d >= 20 else 1)
            sell = min(n, pace)
            if sell > 0:
                orders.append(["SELL", item, sell])

    # 2. HIRE batch: aggressive from day 0 (8 hires = $64 total)
    load = len(build_needs(obs))
    if d < SEASON_LATE - 2:
        while True:
            hired_today = me["hires_today"] + sum(1 for o in orders if o[0] == "HIRE")
            if hired_today >= MAX_HANDS:
                break
            cost = FIB[min(me["hires_today"], len(FIB) - 1)]
            if cost > money * HIRE_FRACTION or load <= (1 + hired_today) * 8:
                break
            orders.append(["HIRE"])
            money -= cost

    # 3. Investments, replay-proven schedule:
    #    NE@day6, SW@day11 (universal); goose for fertilizer; BUY wheat for feed
    #    (6207/6269 winners buy wheat instead of planting it).
    reserve = 40 + n_animals(obs) * 10
    unlocked_extra = len(me["unlocked_quadrants"]) - 1
    for qidx, buy_day in enumerate(LAND_DAYS):
        if unlocked_extra == qidx and d >= buy_day and money >= LAND_PRICES[qidx] + 100:
            orders.append(["BUY_LAND"])
            money -= LAND_PRICES[qidx]
    if 3 <= d <= 8 and n_animals(obs, "GOOSE") == 0 and shed_count(obs, "GOOSE") == 0 \
            and money >= 300 + 100:
        orders.append(["BUY_ANIMAL", "GOOSE", 1])
        money -= 300
    if n_animals(obs) > 0 and available(obs, "WHEAT") < n_animals(obs) * 2 and d < SEASON_LATE:
        need_w = min(20, max(0, n_animals(obs) * 2 - available(obs, "WHEAT")))
        if need_w > 0 and money >= need_w * CROPS["WHEAT"]["seed"] + 40:
            orders.append(["BUY_PRODUCT", "WHEAT", need_w])
            money -= need_w * CROPS["WHEAT"]["seed"]
    for crop in ("STRAWBERRY", "CARROT", "WHEAT", "MELON"):
        want = seed_target(obs, crop)
        have = obs["private"]["seeds"].get(crop, 0)
        need = max(0, want - have)
        if need > 0:
            cost = CROPS[crop]["seed"] * need
            if money >= cost + 40:
                orders.append(["BUY_SEED", crop, need])
                money -= cost
    return orders[:10]


def seed_target(obs, crop):
    """Wanted seed stock per crop. Melon = money crop (days 3-15)."""
    d = game_day(obs)
    if d >= NO_PLANT_AFTER:
        return 0
    if crop == "MELON":
        if MELON_START_DAY <= d <= 15:
            money = obs["farms"][obs["player"]]["money"]
            if money >= 100:
                return max(1, min(12, (money - 150) // CROPS["MELON"]["seed"]))
        return 0
    if crop == "WHEAT":
        return 12 if d < SEASON_LATE else 0
    if crop == "CARROT":
        return 4 if d < 6 else 0
    return 0


# ================================================================ dispatcher
def pickup_op(obs):
    """Batched pickup from shed (1 turn for n items)."""
    if fert_want(obs) > 0 and shed_count(obs, "FERTILIZER") > 0:
        return ["PICKUP", "FERTILIZER", min(3, shed_count(obs, "FERTILIZER"))]
    for a in ("GOOSE", "COW", "SHEEP"):
        if shed_count(obs, a) > 0 and empty_structures(obs, a):
            return ["PICKUP", a, 1]
    return None


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


def decide_unit(obs, pos):
    """One unit's action: current-tile priority ladder, else move to nearest need."""
    me = obs["farms"][obs["player"]]
    d = game_day(obs)
    fx, fy = pos
    tile = me["tiles"][fy][fx]

    # 0. HARVEST (decay = 1 unit / 2 turns)
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

    # 3. FERTILIZE in bonus window
    if is_plant(tile):
        cd = CROPS[tile["crop"]]
        age = d - tile["planted_day"]
        ws = (cd["maxyd"] + 1) // 2
        if ws <= age <= cd["maxyd"] and tile.get("fertilized_until_day", -1) < d \
                and inv_count(obs, "FERTILIZER") > 0:
            return ["FERTILIZE"]

    # 4. PLANT on empty tile / PLACE animal on empty structure
    if tile is None:
        crop = pick_crop(obs, fx, fy)
        if crop and obs["private"]["seeds"].get(crop, 0) > 0:
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

    # 7. PICKUP needed items if at shed (batched)
    if at_shed(obs, (fx, fy)):
        pu = pickup_op(obs)
        if pu:
            return pu

    # 8. MOVE to nearest unclaimed need
    need = nearest_unclaimed(obs, (fx, fy))
    if need:
        return [step_towards((fx, fy), need["pos"])]
    return ["PASS"]


# ================================================================ agent entry
CLAIMED = set()


def agent(obs):
    """Kaggle entry point — last function defined in main.py."""
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
