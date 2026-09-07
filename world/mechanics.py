"""L0 mechanics — the engine's rule tables + guard-mirroring queries.

EVERY function here is one of:
  - a direct import/alias of the engine's own object (zero drift), or
  - an ENGINE RULE with a source-line citation in its docstring, written in
    the same shape as the engine's guard.

Anything else (policies, derived models, "when to harvest" opinions) does
not belong in L0 — that's L1/L2 territory. Enforced by
tests/test_source_citations.py.
"""
from __future__ import annotations

from functools import lru_cache

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
# Tables — imported from the engine, never copied.
# ------------------------------------------------------------------

CROPS: dict[str, dict] = _ENGINE_CROPS          # seed/first_yield_day/max_yield_day/interval/max_yield/ongoing
ANIMALS: dict[str, dict] = _ENGINE_ANIMALS      # cost/structure/first_yield_day/interval/max_held/product
PRODUCTS: tuple[str, ...] = tuple(_ENGINE_PRODUCTS)
MARKET_I0: int = _ENGINE_MARKET_I0
PRICE_FLOOR: int = _ENGINE_PRICE_FLOOR
LAND_PRICES: tuple[int, int, int] = tuple(_ENGINE_LAND_PRICES)   # (NE, SW, SE) — engine L96
FARM_HAND_COST_MULT: int = _ENGINE_FARM_HAND_COST_MULT

PRODUCT_BASE: dict[str, int] = {p: _ENGINE_MARKET_PARAMS[p]["base"] for p in PRODUCTS}
SELLABLE: tuple[str, ...] = PRODUCTS

SEASON_DAYS = 30
TURNS_PER_DAY = 24
BOARD = 10
SHED_CAP = 100
MARKET_ORDERS_PER_TURN = 10
WEED_CHANCE = 0.005

ONE_TIME_CROPS: tuple[str, ...] = tuple(c for c, d in CROPS.items() if not d["ongoing"])
ONGOING_CROPS: tuple[str, ...] = tuple(c for c, d in CROPS.items() if d["ongoing"])

ANIMAL_STRUCTURE = {a: d["structure"] for a, d in ANIMALS.items()}

# The engine's market price function itself — re-exported verbatim.
market_price = _engine_market_price


# ------------------------------------------------------------------
# Guard-mirroring queries. Each carries its engine citation.
# ------------------------------------------------------------------

def is_plant(tile) -> bool:
    """ENGINE RULE (L341, L410): `isinstance(tile, dict) and tile.get("kind") == "PLANT"`."""
    return isinstance(tile, dict) and tile.get("kind") == "PLANT"


def is_animal_tile(tile) -> bool:
    """ENGINE RULE (L811): `isinstance(tile, dict) and "animal" in tile`."""
    return bool(isinstance(tile, dict) and "animal" in tile)


def plant_age(tile, day: int) -> int:
    """ENGINE RULE (L439, L453): `day - tile["planted_day"]`."""
    return day - tile.get("planted_day", day)


def water_bonus_window(crop: str) -> tuple[int, int]:
    """ENGINE RULE (L440-441): window_start = (max_yield_day + 1) // 2;
    window = [window_start .. max_yield_day]. Cached: pure function of the
    crop table."""
    return _water_bonus_window_cached(crop)


@lru_cache(maxsize=None)
def _water_bonus_window_cached(crop: str) -> tuple[int, int]:
    cd = CROPS[crop]
    return (cd["max_yield_day"] + 1) // 2, cd["max_yield_day"]


def animal_production_due(tile, day: int) -> bool:
    """ENGINE RULE (L828-829): `(day + 1 - placed_day - first_yield_day) >= 0
    and % interval == 0`, evaluated at END-of-day refresh for next_day."""
    if not is_animal_tile(tile):
        return False
    a = ANIMALS[tile["animal"]]
    return _production_due_cached(tile["animal"], tile.get("placed_day", day), day,
                                  a["first_yield_day"], a["interval"])


@lru_cache(maxsize=None)
def _production_due_cached(animal: str, placed_day: int, day: int,
                           first: int, interval: int) -> bool:
    since = (day + 1) - placed_day - first
    return since >= 0 and since % interval == 0


def animal_pending_yield(tile) -> int:
    """ENGINE RULE (L831-833): on a production day,
    yield_units = min(max_held, yield_units + 1 + bonus) where bonus is the
    pending care bank, consumed only if fed_today. This predicts that payout."""
    if not is_animal_tile(tile):
        return 0
    a = ANIMALS[tile["animal"]]
    bank = tile.get("pending_care_bonus", 0) if tile.get("fed_today") else 0
    return min(a["max_held"], tile.get("yield_units", 0) + 1 + bank)


def animal_needs_feed(tile) -> bool:
    """ENGINE RULE (L505): `tile["fed_today"]` -> FEED returns (inert)."""
    return is_animal_tile(tile) and not tile.get("fed_today")


def animal_needs_care(tile) -> bool:
    """ENGINE RULE (L519): `tile["cared_today"]` -> CARE returns (inert)."""
    return is_animal_tile(tile) and not tile.get("cared_today")


def animal_fertilizer_ready(tile) -> bool:
    """ENGINE RULE (L512): `tile["fertilizer_available"]` gate on COLLECT_FERTILIZER."""
    return is_animal_tile(tile) and bool(tile.get("fertilizer_available"))


def hire_cost(hires_today: int) -> int:
    """ENGINE RULE (L690-691): mult * fib(n_already_today), fib(0)=1, fib(1)=1.
    Cached: pure function of an int."""
    return _hire_cost_cached(hires_today)


@lru_cache(maxsize=None)
def _hire_cost_cached(hires_today: int) -> int:
    a, b = 1, 1
    for _ in range(hires_today):
        a, b = b, a + b
    return FARM_HAND_COST_MULT * a


def sellable_items() -> tuple:
    return SELLABLE
