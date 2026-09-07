"""L0 mechanics — single source of truth for Kaggriculture rules.

**Derived from the engine, never hand-copied.** The rule tables and the price
function are imported from `kaggle_environments.envs.kaggriculture` at import
time, so this module can never drift from the environment the agent actually
runs in. Query functions add lab-verified *interpretations* (harvest policy,
care-bank payout, escape thresholds) on top of those engine tables.

Provenance of interpretation functions (engine source, kaggriculture.py):
- animal production gate: L~828 `(day - placed - first) >= 0 and % interval == 0`;
  base yield unconditional, fed_today gates only the care bank
- one-time crop watering bonus: L440-441 (window ceil(maxyd/2)..maxyd)
- lab evidence: docs/research/006-mechanics-verification.md + lab/_g*.py runs
"""
from __future__ import annotations

from kaggle_environments.envs.kaggriculture.kaggriculture import (
    ANIMALS as _ENGINE_ANIMALS,
    CROPS as _ENGINE_CROPS,
    FARM_HAND_COST_MULT as _ENGINE_FARM_HAND_COST_MULT,
    LAND_PRICES as _ENGINE_LAND_PRICES,
    MARKET_I0 as _ENGINE_MARKET_I0,
    MARKET_PARAMS as _ENGINE_MARKET_PARAMS,
    PRICE_FLOOR as _ENGINE_PRICE_FLOOR,
    PRODUCTS as _ENGINE_PRODUCTS,
    market_price as _engine_market_price,
)

# ------------------------------------------------------------------
# Tables straight from the engine (aliased, not copied).
# ------------------------------------------------------------------

CROPS: dict[str, dict] = _ENGINE_CROPS          # seed/first/maxyd/interval/max_yield/ongoing
ANIMALS: dict[str, dict] = _ENGINE_ANIMALS      # cost/structure/first/interval/max_held/product
PRODUCTS: tuple[str, ...] = tuple(_ENGINE_PRODUCTS)
MARKET_I0: int = _ENGINE_MARKET_I0
PRICE_FLOOR: int = _ENGINE_PRICE_FLOOR
LAND_PRICES: tuple[int, int, int] = tuple(_ENGINE_LAND_PRICES)   # (NE, SW, SE)
FARM_HAND_COST_MULT: int = _ENGINE_FARM_HAND_COST_MULT

PRODUCT_BASE: dict[str, int] = {p: _ENGINE_MARKET_PARAMS[p]["base"] for p in PRODUCTS}
SELLABLE: tuple[str, ...] = PRODUCTS            # engine lets every product be sold

SEASON_DAYS = 30
TURNS_PER_DAY = 24
BOARD = 10          # 10x10 grid, four 5x5 quadrants
SHED_CAP = 100
MARKET_ORDERS_PER_TURN = 10
UNLOCK_COST = dict(zip(("NE", "SW", "SE"), LAND_PRICES))
WEED_CHANCE = 0.005

ONE_TIME_CROPS: tuple[str, ...] = tuple(c for c, d in CROPS.items() if not d["ongoing"])
ONGOING_CROPS: tuple[str, ...] = tuple(c for c, d in CROPS.items() if d["ongoing"])

ANIMAL_STRUCTURE = {a: d["structure"] for a, d in ANIMALS.items()}


# ------------------------------------------------------------------
# Price — re-export the ENGINE function itself (zero drift by construction).
# ------------------------------------------------------------------

market_price = _engine_market_price


# ------------------------------------------------------------------
# Interpretation queries (stateless; take tiles/state, answer questions).
# These encode lab-verified semantics ON TOP of engine tables.
# ------------------------------------------------------------------

def water_bonus_window(crop: str) -> tuple[int, int]:
    """Inclusive day-range in which watering a one-time crop adds +1 yield/day
    (engine L440-441: window_start = (max_yield_day + 1) // 2)."""
    cd = CROPS[crop]
    return (cd["max_yield_day"] + 1) // 2, cd["max_yield_day"]


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
    """Yield of a one-time crop harvested at `age` with watered/fertilized days
    inside the bonus window. Water bonus doubles under fertilizer (engine
    doubling applies to the watering bonus, not the base); capped at max_yield."""
    cd = CROPS[crop]
    if age < cd["first_yield_day"]:
        return 0
    lo, hi = water_bonus_window(crop)
    bonus_days = max(0, min(watered_days, hi - lo + 1))
    if fert_days:
        return min(1 + 2 * bonus_days, cd["max_yield"])
    return min(1 + bonus_days, cd["max_yield"])


def plant_needs_water(tile, day: int) -> bool:
    """A living plant needs water today unless already watered. Dead plants
    (2 consecutive unwatered days -> weed) never do."""
    if not is_plant(tile):
        return False
    if tile.get("watered_today"):
        return False
    return tile.get("consecutive_unwatered", 0) < 2


def plant_is_lost(tile) -> bool:
    return is_plant(tile) and tile.get("consecutive_unwatered", 0) >= 2


def plant_mature(tile, day: int) -> bool:
    """Harvest policy: one-time crops at/after max_yield_day (full yield,
    lab-verified — early harvest throws away yield); ongoing crops whenever
    yield_units > 0."""
    if not is_plant(tile):
        return False
    cd = CROPS.get(crop_of(tile))
    if cd is None:
        return False
    if not cd["ongoing"]:
        return plant_age(tile, day) >= cd["max_yield_day"]
    return tile.get("yield_units", 0) > 0


def animal_production_due(tile, day: int) -> bool:
    """(day - placed - first) >= 0 and % interval == 0 (engine L~828)."""
    if not is_animal_tile(tile):
        return False
    a = ANIMALS[tile["animal"]]
    since = day - tile.get("placed_day", day) - a["first_yield_day"]
    return since >= 0 and since % a["interval"] == 0


def animal_pending_yield(tile) -> int:
    """Base (unconditional) + care bank, capped at max_held on the tile.
    The bank pays out only on fed production days (engine L~828)."""
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
    """Cost of the NEXT hire: mult * fib(1,1,2,3,5,...), resets daily."""
    a, b = 1, 1
    for _ in range(hires_today):
        a, b = b, a + b
    return FARM_HAND_COST_MULT * a


def sellable_items() -> tuple:
    return SELLABLE
