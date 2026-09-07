"""L0 mechanics — single source of truth for Kaggriculture rules.

Stateless rule tables + query functions. No state is stored here; every
function takes a state-like mapping (dict/dataclass view of the observation)
and answers a question. Values are lab-verified against the engine source
(kaggriculture.py) and docs/research/006-mechanics-verification.md.

Rule provenance (engine source lines):
- CROPS/ANIMALS tables: kaggriculture.py L12-22
- one-time crop watering bonus: L440-441 (window = ceil(max_yield_day/2)..max)
- ongoing crop production: doubled if watered AND fertilized that day
- animal production: (day - placed - first) >= 0 and % interval == 0; base is
  unconditional, fed_today gates only the care bank (L~828)
- decay: one-time crops lose 1 yield every other turn after max_lifespan
- escape/weed: 2 consecutive unfed/unwatered days
"""
from __future__ import annotations

CROPS: dict[str, dict] = {
    "WHEAT":      {"seed": 10,  "first": 2,  "maxyd": 4,  "interval": 0, "max_yield": 6, "ongoing": False, "base": 25},
    "CARROT":     {"seed": 20,  "first": 2,  "maxyd": 3,  "interval": 0, "max_yield": 4, "ongoing": False, "base": 35},
    "TOMATO":     {"seed": 50,  "first": 8,  "maxyd": 8,  "interval": 1, "max_yield": 4, "ongoing": True,  "base": 60},
    "STRAWBERRY": {"seed": 100, "first": 10, "maxyd": 10, "interval": 2, "max_yield": 4, "ongoing": True,  "base": 120},
    "MELON":      {"seed": 80,  "first": 10, "maxyd": 12, "interval": 0, "max_yield": 6, "ongoing": False, "base": 250},
}

ANIMALS: dict[str, dict] = {
    "GOOSE": {"cost": 300, "structure": "COOP",    "first": 4, "interval": 1, "max_held": 4, "product": "EGG",  "product_base": 50},
    "COW":   {"cost": 400, "structure": "PASTURE", "first": 8, "interval": 2, "max_held": 6, "product": "MILK", "product_base": 160},
    "SHEEP": {"cost": 500, "structure": "PASTURE", "first": 6, "interval": 3, "max_held": 6, "product": "WOOL", "product_base": 200},
}

PRODUCT_BASE = {"WHEAT": 25, "CARROT": 35, "TOMATO": 60, "STRAWBERRY": 120,
                "MELON": 250, "EGG": 50, "MILK": 160, "WOOL": 200, "FERTILIZER": 100}

SELLABLE = list(PRODUCT_BASE)

SEASON_DAYS = 30
TURNS_PER_DAY = 24
BOARD = 10          # 10x10 grid, four 5x5 quadrants
SHED_CAP = 100
MARKET_ORDERS_PER_TURN = 10
FARM_HAND_COST_MULT = 1   # nth hire costs mult * fib(n), resets daily
UNLOCK_COST = {"NE": 1000, "SW": 2000, "SE": 4000}
WEED_CHANCE = 0.005

# structure kinds an animal may live in
ANIMAL_STRUCTURE = {"GOOSE": "COOP", "COW": "PASTURE", "SHEEP": "PASTURE"}

# harvest policy: one-time crops must be harvested at/after maxyd (full yield).
ONE_TIME_CROPS = tuple(c for c, d in CROPS.items() if not d["ongoing"])
ONGOING_CROPS = tuple(c for c, d in CROPS.items() if d["ongoing"])


# ---------------------------------------------------------------- queries

def water_bonus_window(crop: str) -> tuple[int, int]:
    """Inclusive day-range in which watering a one-time crop adds +1 yield/day."""
    cd = CROPS[crop]
    start = (cd["maxyd"] + 1) // 2
    return start, cd["maxyd"]


def is_plant(tile) -> bool:
    return isinstance(tile, dict) and tile.get("kind") == "PLANT"


def is_animal_tile(tile) -> bool:
    return bool(isinstance(tile, dict) and tile.get("kind") in ("COOP", "PASTURE")
                and tile.get("animal"))


def plant_age(tile, day: int) -> int:
    return day - tile.get("planted_day", day)


def crop_of(tile):
    return tile.get("crop") if is_plant(tile) else None


def one_time_yield_at(crop: str, age: int, watered_days: int, fert_days: int) -> int:
    """Yield of a one-time crop if harvested at `age` with the given
    watered/fertilized days inside the bonus window. Fertilizer doubles the
    watering bonus (never the base). Never exceeds max_yield."""
    cd = CROPS[crop]
    if age < cd["first"]:
        return 0
    lo, hi = water_bonus_window(crop)
    bonus_days = max(0, min(watered_days, hi - lo + 1))
    if fert_days:
        return min(1 + 2 * bonus_days, cd["max_yield"])
    return min(1 + bonus_days, cd["max_yield"])


def plant_needs_water(tile, day: int) -> bool:
    """A living plant needs water today unless already watered; decayed plants
    and weeds never do. Plants die after 2 consecutive unwatered days."""
    if not is_plant(tile):
        return False
    if tile.get("watered_today"):
        return False
    return tile.get("consecutive_unwatered", 0) < 2


def plant_is_lost(tile) -> bool:
    """2 consecutive unwatered days -> weed (dead)."""
    return is_plant(tile) and tile.get("consecutive_unwatered", 0) >= 2


def plant_mature(tile, day: int) -> bool:
    """Harvest policy: one-time crops at/after max_yield_day (full yield);
    ongoing crops whenever yield_units > 0."""
    if not is_plant(tile):
        return False
    crop = crop_of(tile)
    cd = CROPS.get(crop)
    if cd is None:
        return False
    age = plant_age(tile, day)
    if not cd["ongoing"]:
        return age >= cd["maxyd"]
    return tile.get("yield_units", 0) > 0


def animal_production_due(tile, day: int) -> bool:
    """(day - placed - first) >= 0 and % interval == 0 (engine L~828)."""
    if not is_animal_tile(tile):
        return False
    a = ANIMALS[tile["animal"]]
    since = day - tile.get("placed_day", day) - a["first"]
    return since >= 0 and since % a["interval"] == 0


def animal_pending_yield(tile) -> int:
    """Base (unconditional) + care bank (paid only on fed production days)."""
    if not is_animal_tile(tile):
        return 0
    a = ANIMALS[tile["animal"]]
    bank = tile.get("pending_care_bonus", 0) if tile.get("fed_today") else 0
    return min(a["max_held"], 1 + bank)


def animal_needs_feed(tile) -> bool:
    return is_animal_tile(tile) and not tile.get("fed_today")


def animal_needs_care(tile) -> bool:
    return is_animal_tile(tile) and not tile.get("cared_today")


def animal_is_lost(tile) -> bool:
    return is_animal_tile(tile) and tile.get("consecutive_unfed", 0) >= 2


def animal_fertilizer_ready(tile) -> bool:
    return is_animal_tile(tile) and bool(tile.get("fertilizer_available"))


def hire_cost(hires_today: int) -> int:
    """Cost of the NEXT hire given hires made today: fib(1,1,2,3,5,...)."""
    a, b = 1, 1
    for _ in range(hires_today):
        a, b = b, a + b
    return FARM_HAND_COST_MULT * a


def sellable_items() -> tuple:
    return tuple(SELLABLE)
