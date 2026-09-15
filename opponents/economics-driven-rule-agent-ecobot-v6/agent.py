"""Kaggle Submission — Single File Bundle."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable

# ---------------------------------------------------------------------------
# MODULE: constants.py
# ---------------------------------------------------------------------------

TOTAL_DAYS: int = 30
TURNS_PER_DAY: int = 24
MAX_MARKET_ORDERS: int = 10
BOARD_SIZE: int = 10
I0: int = 10_000

# Interleaved by distance rings from shed junction across NW, NE, SW (total 18 pasture slots)
PASTURE_CLUSTER: list[tuple[int, int]] = [
    # Ring 0: Shed corner tiles (dist 0)
    (4, 4), (5, 4), (4, 5),
    # Ring 1: Immediate shed neighbors (dist 1)
    (4, 3), (3, 4), (5, 3), (6, 4), (3, 5), (4, 6),
    # Ring 2: Secondary cluster tiles (dist 2)
    (3, 3), (4, 2), (2, 4), (6, 3), (5, 2), (7, 4), (2, 5), (3, 6), (4, 7),
]
SHED_TILES: list[tuple[int, int]] = [(4, 4), (5, 4), (4, 5), (5, 5)]
SHED_TILES_SET: frozenset[tuple[int, int]] = frozenset(SHED_TILES)

MILK_SUPPORT_SHOPS: tuple[str, ...] = ("PIZZA_SHOP", "ICE_CREAM_SHOP", "SMOOTHIE_SHOP")
EGG_SUPPORT_SHOPS: tuple[str, ...] = ("BAKERY", "BRUNCH_SPOT")

SELLABLE_ITEMS: tuple[str, ...] = (
    "FERTILIZER", "MILK", "WOOL", "EGG",
    "MELON", "STRAWBERRY", "CARROT", "TOMATO", "WHEAT",
)

SHOP_DEMANDS: dict[str, dict[str, float]] = {
    "BAKERY":         {"EGG": 6.0, "WHEAT": 6.0},
    "PIZZA_SHOP":     {"MILK": 6.0, "TOMATO": 6.0, "WHEAT": 6.0},
    "BRUNCH_SPOT":    {"EGG": 6.0, "WHEAT": 6.0, "STRAWBERRY": 6.0},
    "YARN_STORE":     {"WOOL": 12.0},
    "ICE_CREAM_SHOP": {"STRAWBERRY": 6.0, "MILK": 6.0, "WHEAT": 6.0},
    "PET_CAFE":       {"CARROT": 12.0},
    "SMOOTHIE_SHOP":  {"STRAWBERRY": 6.0, "MILK": 6.0},
    "FARMERS_MARKET": {"WHEAT": 6.0, "CARROT": 6.0, "TOMATO": 6.0, "STRAWBERRY": 6.0},
}

TOWN_CENTER_FLAT: float = 1.0
LIQUIDATION_DAY: int = 28
MELON_LAST_PLANT_DAY: int = 18
STRAWBERRY_LAST_PLANT_DAY: int = 14
STRAWBERRY_LAND_HORIZON_DAYS: int = 3
WHEAT_LAST_PLANT_DAY: int = 26
PASTURE_RESERVATION_LAST_DAY: int = 16
STRAWBERRY_BASE_TARGET_BUSHES: int = 15
STRAWBERRY_BOOST_TARGET_BUSHES: int = 42
MAX_TOTAL_HERD: int = 18
MIN_PAYBACK_DAYS: int = 8
HAND_COST_PER_ANIMAL_DAY: float = 2.0
SEED_PURCHASE_CASH_RESERVE: float = 200.0
FEED_WHEAT_RESERVE: int = 4
FEED_WHEAT_RESERVE_MIN_RATIO: int = 2
FEED_WHEAT_SELL_RATIO: int = 4
N_FERTILIZER_HANDS: int = 2
FERTILIZER_PICKUP_BATCH: int = 4

CROPS: dict[str, dict[str, Any]] = {
    "WHEAT": {
        "seed_cost": 10, "base_price": 25,
        "first_yield_day": 2, "max_yield_day": 4,
        "max_units_base": 4, "max_units_fert": 6,
        "bonus_start_day": 2, "is_ongoing": False,
    },
    "CARROT": {
        "seed_cost": 20, "base_price": 35,
        "first_yield_day": 2, "max_yield_day": 3,
        "max_units_base": 3, "max_units_fert": 4,
        "bonus_start_day": 2, "is_ongoing": False,
    },
    "TOMATO": {
        "seed_cost": 50, "base_price": 60,
        "first_yield_day": 8, "max_yield_day": 11,
        "is_ongoing": True,
        "bonus_start_day": 8, "yield_days": [8, 9, 10, 11],
    },
    "STRAWBERRY": {
        "seed_cost": 100, "base_price": 120,
        "first_yield_day": 10, "max_yield_day": 16,
        "is_ongoing": True,
        "bonus_start_day": 10, "yield_days": [10, 12, 14, 16],
    },
    "MELON": {
        "seed_cost": 80, "base_price": 250,
        "first_yield_day": 10, "max_yield_day": 10,
        "is_ongoing": False,
        "bonus_start_day": 6, "max_units_base": 6, "max_units_fert": 6,
    },
}

ANIMALS: dict[str, dict[str, Any]] = {
    "GOOSE": {"cost": 300, "base_price": 50, "structure": "COOP", "first_yield_day": 4, "interval": 1, "product": "EGG"},
    "COW":   {"cost": 400, "base_price": 160, "structure": "PASTURE", "first_yield_day": 8, "interval": 2, "product": "MILK"},
    "SHEEP": {"cost": 500, "base_price": 200, "structure": "PASTURE", "first_yield_day": 6, "interval": 3, "product": "WOOL"},
}

MARKET_PARAMS: dict[str, dict[str, Any]] = {
    "WHEAT":       {"base": 25,  "T": 400, "below_func": "sqrt",  "below_target": 0.80, "above_func": "log",    "above_target": 0.20},
    "CARROT":      {"base": 35,  "T": 450, "below_func": "hinge", "below_target": 1.00, "above_func": "sqrt",   "above_target": 0.70},
    "TOMATO":      {"base": 60,  "T": 200, "below_func": "hinge", "below_target": 0.40, "above_func": "sqrt",   "above_target": 0.60},
    "STRAWBERRY":  {"base": 120, "T": 100, "below_func": "sqrt",  "below_target": 0.70, "above_func": "linear", "above_target": 1.60},
    "MELON":       {"base": 250, "T": 300, "below_func": "log",   "below_target": 0.20, "above_func": "sq",     "above_target": 3.60},
    "EGG":         {"base": 50,  "T": 332, "below_func": "hinge", "below_target": 0.40, "above_func": "log",    "above_target": 0.20},
    "MILK":        {"base": 160, "T": 122, "below_func": "sqrt",  "below_target": 0.60, "above_func": "linear", "above_target": 1.60},
    "WOOL":        {"base": 200, "T": 105, "below_func": "log",   "below_target": 0.20, "above_func": "sq",     "above_target": 3.20},
    "FERTILIZER":  {"base": 100, "T": 200, "below_func": "linear","below_target": 0.40, "above_func": "linear", "above_target": 0.40},
}

FIBONACCI_HIRE_COSTS: tuple[int, ...] = (
    1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597, 2584, 4181, 6765,
)

class _Constants:
    def __getattr__(self, name: str) -> Any:
        try:
            return globals()[name]
        except KeyError:
            raise AttributeError(f'Constant {name!r} is not defined') from None

c = _Constants()


# ---------------------------------------------------------------------------
# MODULE: market.py
# ---------------------------------------------------------------------------

def total_hire_cost(count: int, already_hired: int) -> int:
    if count < 0 or already_hired < 0 or already_hired + count > len(c.FIBONACCI_HIRE_COSTS):
        raise ValueError(
            f"Invalid hire range: count={count}, already_hired={already_hired}, max_limit={len(c.FIBONACCI_HIRE_COSTS)}"
        )
    return sum(c.FIBONACCI_HIRE_COSTS[already_hired : already_hired + count])


def _shape(func: str, x: float, t: float) -> float:
    if func == "linear":
        return x
    if func == "sq":
        return x * x
    if func == "sqrt":
        return math.sqrt(max(0.0, x))
    if func == "log":
        return math.log(1.0 + max(0.0, x))
    if func == "hinge":
        u = x / max(1.0, t)
        return u + 8.0 * max(0.0, u - 1.0) ** 2
    raise ValueError(f"Unknown shape function: {func!r}")


def market_price(item: str, inv: int) -> int:
    if item not in c.MARKET_PARAMS:
        raise ValueError(f"Unknown market item: {item!r}")
    p = c.MARKET_PARAMS[item]
    base = float(p["base"])
    t = float(p["T"])
    if inv == c.I0:
        return int(round(base))
    if inv < c.I0:
        x = float(c.I0 - inv)
        denom = _shape(p["below_func"], t, t)
        amp = (p["below_target"] * base) / denom if denom > 0 else 0.0
        return max(1, int(round(base + amp * _shape(p["below_func"], x, t))))
    x = float(inv - c.I0)
    denom = _shape(p["above_func"], t, t)
    amp = (p["above_target"] * base) / denom if denom > 0 else 0.0
    return max(1, int(round(base - amp * _shape(p["above_func"], x, t))))


def get_price(item: str, market_inv: dict[str, int]) -> int:
    return market_price(item, market_inv.get(item, c.I0))


# ---------------------------------------------------------------------------
# MODULE: navigation.py
# ---------------------------------------------------------------------------

def manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def quad_of(pos: tuple[int, int]) -> str:
    half = c.BOARD_SIZE // 2
    if pos[1] < half:
        return "NE" if pos[0] >= half else "NW"
    return "SE" if pos[0] >= half else "SW"


def is_on_shed_tile(pos: tuple[int, int]) -> bool:
    return pos in c.SHED_TILES_SET


def closest_shed_pos(pos: tuple[int, int]) -> tuple[int, int]:
    return min(c.SHED_TILES, key=lambda s: manhattan(pos, s))


def bfs_step(start: tuple[int, int], goal: tuple[int, int]) -> str | None:
    if start == goal:
        return None
    dx, dy = goal[0] - start[0], goal[1] - start[1]
    if abs(dx) >= abs(dy):
        return "EAST" if dx > 0 else "WEST"
    return "SOUTH" if dy > 0 else "NORTH"


def go_shed_or_drop(pos: tuple[int, int]) -> list[Any]:
    if is_on_shed_tile(pos):
        return ["DROP"]
    return [bfs_step(pos, closest_shed_pos(pos)) or "PASS"]


# ---------------------------------------------------------------------------
# MODULE: farm.py
# ---------------------------------------------------------------------------

@dataclass(slots=True, frozen=True)
class AnimalCensus:
    field_cows: int
    field_sheep: int
    field_geese: int
    shed_cows: int
    shed_sheep: int
    shed_geese: int
    carried_cows: int
    carried_sheep: int
    carried_geese: int

    @property
    def total_cows(self) -> int:
        return self.field_cows + self.shed_cows + self.carried_cows

    @property
    def total_sheep(self) -> int:
        return self.field_sheep + self.shed_sheep + self.carried_sheep

    @property
    def total_geese(self) -> int:
        return self.field_geese + self.shed_geese + self.carried_geese

    @property
    def total(self) -> int:
        return self.total_cows + self.total_sheep

    @property
    def total_all(self) -> int:
        return self.total + self.total_geese

    @property
    def on_field(self) -> int:
        return self.field_cows + self.field_sheep + self.field_geese

    @property
    def in_shed(self) -> int:
        return self.shed_cows + self.shed_sheep + self.shed_geese

    @property
    def n_feedable(self) -> int:
        return self.on_field + self.in_shed


def count_animal_census(
    farm_state: dict[str, Any],
    shed: dict[str, int],
    inventories: list[dict[str, int]],
) -> AnimalCensus:
    return AnimalCensus(
        field_cows=sum(1 for a in farm_state["animals"] if a["animal"] == "COW"),
        field_sheep=sum(1 for a in farm_state["animals"] if a["animal"] == "SHEEP"),
        field_geese=sum(1 for a in farm_state["animals"] if a["animal"] == "GOOSE"),
        shed_cows=shed.get("COW", 0),
        shed_sheep=shed.get("SHEEP", 0),
        shed_geese=shed.get("GOOSE", 0),
        carried_cows=sum(inv.get("COW", 0) for inv in inventories),
        carried_sheep=sum(inv.get("SHEEP", 0) for inv in inventories),
        carried_geese=sum(inv.get("GOOSE", 0) for inv in inventories),
    )


def kept_feedable_count(census: AnimalCensus, retained_caps: dict[str, int] | None) -> int:
    """On-field + in-shed animals that survive a cull's retained_caps, per species."""
    caps = retained_caps if retained_caps is not None else {}
    return (
        min(census.field_cows + census.shed_cows, caps.get("COW", 999))
        + min(census.field_sheep + census.shed_sheep, caps.get("SHEEP", 999))
        + min(census.field_geese + census.shed_geese, caps.get("GOOSE", 999))
    )


def wheat_feed_thresholds(
    census: AnimalCensus, retained_caps: dict[str, int] | None, day: int
) -> tuple[int, int]:
    """Buy floor and sell ceiling for shed wheat, with a deliberate dead zone."""
    if day >= c.LIQUIDATION_DAY:
        return 0, 0
    n_kept = kept_feedable_count(census, retained_caps)
    floor = max(c.FEED_WHEAT_RESERVE, n_kept * c.FEED_WHEAT_RESERVE_MIN_RATIO)
    ceiling = max(c.FEED_WHEAT_RESERVE * c.FEED_WHEAT_SELL_RATIO, n_kept * c.FEED_WHEAT_SELL_RATIO)
    return floor, ceiling


def parse_farm_state(tiles: list[list[Any]], day: int) -> dict[str, Any]:
    state: dict[str, Any] = {
        "animals": [],
        "empty_pastures": [],
        "empty_coops": [],
        "plants": [],
        "weeds": [],
        "empty_tiles": [],
        "unlocked_count": 0,
    }

    for y in range(c.BOARD_SIZE):
        for x in range(c.BOARD_SIZE):
            tile = tiles[y][x]
            pos = (x, y)

            if tile == "LOCKED":
                continue
            state["unlocked_count"] += 1

            if tile is None:
                state["empty_tiles"].append(pos)
                continue

            if not isinstance(tile, dict):
                raise TypeError(f"Unexpected tile structure at {pos}: {tile!r}")

            kind = tile.get("kind")
            if kind == "WEED":
                state["weeds"].append(pos)
            elif kind in ("PASTURE", "COOP"):
                animal = tile.get("animal")
                if animal is None:
                    if kind == "COOP":
                        state["empty_coops"].append(pos)
                    else:
                        state["empty_pastures"].append(pos)
                else:
                    state["animals"].append({
                        "pos": pos,
                        "animal": animal,
                        "fed_today": tile.get("fed_today", False),
                        "consecutive_unfed": tile.get("consecutive_unfed", 0),
                        "cared_today": tile.get("cared_today", False),
                        "fertilizer_available": tile.get("fertilizer_available", False),
                        "yield_units": tile.get("yield_units", 0),
                    })
            elif kind == "PLANT":
                crop = tile.get("crop")
                if crop is None or crop not in c.CROPS:
                    raise ValueError(f"Unknown or missing crop {crop!r} at {pos}: {tile}")
                planted_day = tile.get("planted_day", day)
                age = day - planted_day
                spec = c.CROPS[crop]
                yu = tile.get("yield_units", 0)
                watered = tile.get("watered_today", False)
                fert_until = tile.get("fertilized_until_day", -1)

                if spec["is_ongoing"]:
                    is_expired = age >= spec["yield_days"][-1] + 1
                else:
                    is_expired = age >= spec["max_yield_day"] + 1

                fert_due = False
                if spec["is_ongoing"]:
                    upcoming_ages = [d for d in spec["yield_days"] if d >= age]
                    if upcoming_ages:
                        days_to_yield = upcoming_ages[0] - age
                        yield_day = day + days_to_yield
                        fert_due = days_to_yield <= 1 and fert_until < yield_day

                state["plants"].append({
                    "pos": pos,
                    "crop": crop,
                    "age": age,
                    "watered_today": watered,
                    "yield_units": yu,
                    "fertilize_due": fert_due,
                    "is_expired": is_expired,
                    "is_ongoing": spec["is_ongoing"],
                    "first_yield_day": spec["first_yield_day"],
                    "max_yield_day": spec["max_yield_day"],
                })
            else:
                raise ValueError(f"Unknown tile kind {kind!r} at {pos}: {tile}")

    return state


def planted_count(farm_state: dict[str, Any], crop: str) -> int:
    return sum(1 for p in farm_state["plants"] if p["crop"] == crop)


def committed_count(farm_state: dict[str, Any], seeds: dict[str, int], crop: str) -> int:
    return planted_count(farm_state, crop) + seeds.get(crop, 0)


def count_fertilize_due(farm_state: dict[str, Any]) -> int:
    return sum(1 for p in farm_state["plants"] if p["fertilize_due"])


def live_reserved_pastures(
    cluster: list[tuple[int, int]],
    unlocked_quads: list[str],
    target_slots: int,
    day: int = 0,
) -> frozenset[tuple[int, int]]:
    if day > c.PASTURE_RESERVATION_LAST_DAY:
        return frozenset()
    quad_cap = 4 if len(unlocked_quads) == 1 else min(c.MAX_TOTAL_HERD, len(unlocked_quads) * 6)
    effective_target = max(target_slots, quad_cap)
    reserved: set[tuple[int, int]] = set()
    for pos in cluster:
        if len(reserved) >= effective_target:
            break
        if quad_of(pos) in unlocked_quads:
            reserved.add(pos)
    return frozenset(reserved)


# ---------------------------------------------------------------------------
# MODULE: planner.py
# ---------------------------------------------------------------------------

@dataclass(slots=True, frozen=True)
class DevelopmentPlan:
    cows: int = 0
    sheep: int = 0
    geese: int = 0
    crop_targets: dict[str, int] = field(default_factory=dict)
    target_quads: int = 1

    @property
    def total_herd(self) -> int:
        return self.cows + self.sheep + self.geese

    @property
    def total_grazers(self) -> int:
        return self.cows + self.sheep


@dataclass(slots=True)
class ReplanState:
    day: int = -1
    shop_count: int = -1
    quad_count: int = -1
    prices: dict[str, int] = field(default_factory=dict)


@dataclass(slots=True)
class DriftState:
    prev_inv: dict[str, int] = field(default_factory=dict)
    prev_day: int = -1
    observed: dict[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class CandidateScore:
    kind: str
    item: str
    cost: float
    daily_profit: float
    npv: float


def compute_daily_demand(unlocked_shops: list[str]) -> dict[str, float]:
    demand: dict[str, float] = {item: c.TOWN_CENTER_FLAT for item in c.SELLABLE_ITEMS}
    demand.pop("FERTILIZER", None)
    for shop in unlocked_shops:
        for item, qty in c.SHOP_DEMANDS.get(shop, {}).items():
            demand[item] = demand.get(item, 0.0) + qty
    return demand


def update_drift_tracker(day: int, market_inv: dict[str, int], state: DriftState) -> dict[str, float]:
    if day != state.prev_day:
        if state.prev_day >= 0:
            for item in c.SELLABLE_ITEMS:
                if item in state.prev_inv and item in market_inv:
                    state.observed[item] = float(state.prev_inv[item] - market_inv[item])
        state.prev_inv = dict(market_inv)
        state.prev_day = day
    return state.observed


def effective_drift(demand: dict[str, float], observed: dict[str, float]) -> dict[str, float]:
    return {
        item: max(demand.get(item, 0.0), 0.6 * observed.get(item, 0.0))
        for item in c.SELLABLE_ITEMS
    }


def herd_feed_cost(
    herd: int, drift: dict[str, float], market_inv: dict[str, int], horizon: int = 8
) -> float:
    w_drift = drift.get("WHEAT", 0.0) + float(herd)
    projected = max(1, market_inv.get("WHEAT", c.I0) - int(w_drift * horizon))
    return float(market_price("WHEAT", projected))


def expected_price(
    item: str,
    lag: int,
    market_inv: dict[str, int],
    drift: dict[str, float],
    additional_supply_units: float = 0.0,
) -> int:
    net_drift = drift.get(item, 0.0) - additional_supply_units
    projected_inv = market_inv.get(item, c.I0) - int(net_drift * lag)
    return market_price(item, max(1, projected_inv))


def demand_multiplier(item: str, demand: dict[str, float], market_inv: dict[str, int]) -> float:
    d = demand.get(item, 0.0)
    scarcity = c.I0 - market_inv.get(item, c.I0)
    if d >= 12.0:
        return 1.6
    if d >= 6.0:
        return 1.3
    if scarcity > 0.6 * c.MARKET_PARAMS[item]["T"]:
        return 1.4
    return 1.0



# --- Mathematical Asset Valuation Functions ---

def evaluate_animal(
    species: str,
    day: int,
    total_herd: int,
    current_species_count: int,
    drift: dict[str, float],
    market_inv: dict[str, int],
    demand: dict[str, float],
) -> CandidateScore:
    spec = c.ANIMALS[species]
    cost = float(spec["cost"])
    item = spec["product"]
    lag = spec["first_yield_day"]
    interval = spec["interval"]
    days_left = c.TOTAL_DAYS - day - lag

    if days_left <= 0:
        return CandidateScore(species, item, cost, 0.0, -1.0)

    harvests = days_left // interval
    lifetime_units = harvests * interval  # Care banks +1/day so average yield is 1 unit/day
    own_supply = current_species_count * 1.0
    p = expected_price(item, lag, market_inv, drift, own_supply)
    revenue = lifetime_units * p * demand_multiplier(item, demand, market_inv)

    fert_price = float(get_price("FERTILIZER", market_inv))
    fert_revenue = (c.TOTAL_DAYS - day) * fert_price

    feed_price = herd_feed_cost(total_herd + 1, drift, market_inv)
    feed_cost = (c.TOTAL_DAYS - day) * feed_price
    labor_cost = (c.TOTAL_DAYS - day) * c.HAND_COST_PER_ANIMAL_DAY
    npv = revenue + fert_revenue - cost - feed_cost - labor_cost

    daily_profit = npv / max(1, c.TOTAL_DAYS - day)
    return CandidateScore(species, item, cost, daily_profit, npv)


def evaluate_fixed_crop(
    crop: str,
    day: int,
    drift: dict[str, float],
    market_inv: dict[str, int],
    demand: dict[str, float],
    current_crop_units: float = 0.0,
) -> CandidateScore:
    spec = c.CROPS[crop]
    cost = float(spec["seed_cost"])
    yield_days = spec["yield_days"]
    last_plant_day = c.STRAWBERRY_LAST_PLANT_DAY if crop == "STRAWBERRY" else c.TOTAL_DAYS - yield_days[0] - 2

    if day > last_plant_day:
        return CandidateScore(crop, crop, cost, 0.0, -1.0)

    valid_harvests = sum(1 for k in yield_days if day + k <= c.TOTAL_DAYS - 1)
    if valid_harvests <= 0:
        return CandidateScore(crop, crop, cost, 0.0, -1.0)

    expected_units = valid_harvests * 1.5
    lag = yield_days[0]
    p = expected_price(crop, lag, market_inv, drift, current_crop_units)
    revenue = expected_units * p * demand_multiplier(crop, demand, market_inv)
    npv = revenue - cost

    lifespan = min(c.TOTAL_DAYS - day, yield_days[-1] + 1)
    daily_profit = npv / max(1, lifespan)
    return CandidateScore(crop, crop, cost, daily_profit, npv)


def evaluate_replant_crop(
    crop: str,
    day: int,
    drift: dict[str, float],
    market_inv: dict[str, int],
    demand: dict[str, float],
    current_crop_units: float = 0.0,
) -> CandidateScore:
    spec = c.CROPS[crop]
    cost = float(spec["seed_cost"])
    grow_days = spec["max_yield_day"]
    yield_units = float(spec["max_units_base"])

    if crop == "MELON" and day > c.MELON_LAST_PLANT_DAY:
        return CandidateScore(crop, crop, cost, 0.0, -1.0)
    if crop == "WHEAT" and day > c.WHEAT_LAST_PLANT_DAY:
        return CandidateScore(crop, crop, cost, 0.0, -1.0)
    if day > c.TOTAL_DAYS - grow_days - 1:
        return CandidateScore(crop, crop, cost, 0.0, -1.0)

    p = expected_price(crop, grow_days, market_inv, drift, current_crop_units)
    revenue = yield_units * p * demand_multiplier(crop, demand, market_inv)
    profit_per_cycle = revenue - cost

    daily_profit = profit_per_cycle / max(1, grow_days)
    npv = daily_profit * (c.TOTAL_DAYS - day)
    return CandidateScore(crop, crop, cost, daily_profit, npv)


def build_development_plan(
    day: int,
    money: float,
    demand: dict[str, float],
    drift: dict[str, float],
    market_inv: dict[str, int],
    farm_state: dict[str, Any],
    census: AnimalCensus,
    unlocked_quads: list[str],
    unlocked_shops: list[str],
    retained_caps: dict[str, int] | None = None,
    downsized: frozenset[str] = frozenset(),
) -> DevelopmentPlan:
    # 1. Base initialization from current live census
    acc_cows = census.total_cows
    acc_sheep = census.total_sheep
    acc_geese = census.total_geese
    acc_crops: dict[str, int] = {}

    # Base plan capacity on ACTUAL unlocked quads (do not speculatively assume extra land upfront)
    target_quad_count = len(unlocked_quads)
    planned_pasture_cap = sum(1 for p in c.PASTURE_CLUSTER if quad_of(p) in unlocked_quads)
    planned_coop_cap = target_quad_count * 2
    planned_tile_capacity = target_quad_count * 25

    # Base straw allocation
    straw_demand = demand.get("STRAWBERRY", 0.0)
    has_high_straw_demand = straw_demand >= 7.0
    straw_target_base = c.STRAWBERRY_BOOST_TARGET_BUSHES if has_high_straw_demand else c.STRAWBERRY_BASE_TARGET_BUSHES
    if day <= c.STRAWBERRY_LAST_PLANT_DAY:
        acc_crops["STRAWBERRY"] = straw_target_base

    # 2. Capacity-driven Greedy Asset Allocation Loop
    max_herd_limit = c.MAX_TOTAL_HERD
    egg_demand = demand.get("EGG", 0.0)
    wool_demand = demand.get("WOOL", 0.0)
    milk_demand = demand.get("MILK", 0.0)

    for _ in range(60):
        candidates: list[CandidateScore] = []
        total_herd = acc_cows + acc_sheep + acc_geese
        total_grazers = acc_cows + acc_sheep

        # Evaluate Animals (if capacity, downsize rules, and caps allow)
        if total_grazers < planned_pasture_cap and total_herd < max_herd_limit:
            if "COW" not in downsized:
                cow_cap = retained_caps.get("COW", 999) if retained_caps is not None else 999
                milk_shops_count = sum(1 for s in unlocked_shops if s in c.MILK_SUPPORT_SHOPS)
                if milk_shops_count == 0:
                    max_cows_allowed = 2
                elif milk_shops_count == 1:
                    max_cows_allowed = 6
                else:
                    max_cows_allowed = min(14, 2 + milk_shops_count * 4)

                if acc_cows < cow_cap and acc_cows < max_cows_allowed and day <= 11:
                    c_cow = evaluate_animal("COW", day, total_herd, acc_cows, drift, market_inv, demand)
                    if c_cow.npv > 0.0:
                        candidates.append(c_cow)

            if "SHEEP" not in downsized:
                sheep_cap = retained_caps.get("SHEEP", 999) if retained_caps is not None else 999
                yarn_shops_count = sum(1 for s in unlocked_shops if s == "YARN_STORE")
                if yarn_shops_count == 0:
                    max_sheep_base = 2
                elif yarn_shops_count == 1:
                    max_sheep_base = 4
                else:
                    max_sheep_base = min(8, yarn_shops_count * 4)

                # Phase 2 (Days 12-15): after Cow deadline passes, if there is active wool demand/price,
                # expand sheep cap to fill any remaining unused pasture slots up to max total herd limit (18)
                if 12 <= day <= 15 and (yarn_shops_count > 0 or wool_demand >= 6.0):
                    max_sheep_allowed = c.MAX_TOTAL_HERD
                else:
                    max_sheep_allowed = max_sheep_base

                if acc_sheep < sheep_cap and acc_sheep < max_sheep_allowed and day <= 15:
                    c_sheep = evaluate_animal("SHEEP", day, total_herd, acc_sheep, drift, market_inv, demand)
                    if c_sheep.npv > 0.0:
                        candidates.append(c_sheep)

        egg_shops_count = sum(1 for s in unlocked_shops if s in c.EGG_SUPPORT_SHOPS)
        if egg_shops_count > 0 and acc_geese < planned_coop_cap and total_herd < max_herd_limit:
            if "GOOSE" not in downsized:
                goose_cap = retained_caps.get("GOOSE", 999) if retained_caps is not None else 999
                if acc_geese < goose_cap and acc_geese < int(egg_demand):
                    c_goose = evaluate_animal("GOOSE", day, total_herd, acc_geese, drift, market_inv, demand)
                    if c_goose.npv > 0.0:
                        candidates.append(c_goose)

        # Evaluate Crops
        wheat_tiles_needed = max(15, int((acc_cows + acc_sheep + acc_geese) * 1.0))
        available_crop_tiles = max(0, planned_tile_capacity - planned_pasture_cap - planned_coop_cap - wheat_tiles_needed)

        if day <= c.STRAWBERRY_LAST_PLANT_DAY:
            straw_count = acc_crops.get("STRAWBERRY", 0)
            c_straw = evaluate_fixed_crop(
                "STRAWBERRY", day, drift, market_inv, demand, current_crop_units=float(straw_count) * 0.46
            )
            if c_straw.npv > 0.0 and straw_count < (c.STRAWBERRY_BOOST_TARGET_BUSHES if has_high_straw_demand else c.STRAWBERRY_BASE_TARGET_BUSHES):
                candidates.append(c_straw)

        total_secondary_crops = sum(v for k, v in acc_crops.items() if k != "STRAWBERRY")

        if demand.get("TOMATO", 0.0) >= 6.0 and total_secondary_crops < available_crop_tiles:
            tomato_count = acc_crops.get("TOMATO", 0)
            c_tomato = evaluate_fixed_crop(
                "TOMATO", day, drift, market_inv, demand, current_crop_units=float(tomato_count) * 0.5
            )
            if c_tomato.npv > 0.0 and tomato_count < 8:
                candidates.append(c_tomato)

        if demand.get("CARROT", 0.0) >= 6.0 and total_secondary_crops < available_crop_tiles:
            carrot_count = acc_crops.get("CARROT", 0)
            c_carrot = evaluate_replant_crop(
                "CARROT", day, drift, market_inv, demand, current_crop_units=float(carrot_count) * 1.17
            )
            if c_carrot.npv > 0.0 and carrot_count < 8:
                candidates.append(c_carrot)

        if not candidates:
            break

        # Best asset by daily profit per tile-day (marginal tile return)
        best = max(candidates, key=lambda c_cand: c_cand.daily_profit)
        if best.daily_profit <= 0.0:
            break

        if best.kind == "COW":
            acc_cows += 1
        elif best.kind == "SHEEP":
            acc_sheep += 1
        elif best.kind == "GOOSE":
            acc_geese += 1
        else:
            acc_crops[best.kind] = acc_crops.get(best.kind, 0) + 1

    # 3. Land Purchase Target Calculation — strictly capacity- and demand-driven
    straw_days_left = c.STRAWBERRY_LAST_PLANT_DAY - day
    if straw_days_left <= c.STRAWBERRY_LAND_HORIZON_DAYS:
        straw_land_need = planted_count(farm_state, "STRAWBERRY")
    else:
        straw_land_need = acc_crops.get("STRAWBERRY", 0)
    other_crop_tiles = sum(v for k, v in acc_crops.items() if k != "STRAWBERRY")

    target_quads = len(unlocked_quads)
    wheat_basis = max(15, int((acc_cows + acc_sheep + acc_geese) * 1.0))
    total_needed_tiles = (
        acc_cows + acc_sheep + acc_geese + other_crop_tiles + straw_land_need + wheat_basis
    )
    current_tile_capacity = len(unlocked_quads) * 25
    quad_limit = 3

    if len(unlocked_quads) < quad_limit:
        next_cost = 1000.0 if len(unlocked_quads) == 1 else (2000.0 if len(unlocked_quads) == 2 else 4000.0)
        has_land_budget = money >= next_cost
        real_empty_tiles = len(farm_state.get("empty_tiles", []))
        is_capacity_constrained = (
            total_needed_tiles >= current_tile_capacity
            or real_empty_tiles <= 2
            or (acc_cows + acc_sheep) > sum(1 for p in c.PASTURE_CLUSTER if quad_of(p) in unlocked_quads)
            or acc_geese > len(unlocked_quads) * 2
        )
        if has_land_budget and is_capacity_constrained:
            target_quads = len(unlocked_quads) + 1

    return DevelopmentPlan(
        cows=acc_cows,
        sheep=acc_sheep,
        geese=acc_geese,
        crop_targets=acc_crops,
        target_quads=target_quads,
    )


@dataclass(slots=True)
class CullState:
    """Hysteresis tracker for economic downsizing of unprofitable species."""
    last_day: int = -1
    negative_days: dict[str, int] = field(default_factory=dict)
    downsized: set[str] = field(default_factory=set)
    retained_caps: dict[str, int] = field(default_factory=dict)


_CULL_CONSECUTIVE_DAYS: int = 2
_CULL_SPECIES: tuple[str, ...] = ("COW", "SHEEP", "GOOSE")


def update_cull_state(
    day: int,
    drift: dict[str, float],
    market_inv: dict[str, int],
    census: AnimalCensus,
    state: CullState,
) -> dict[str, int]:
    if day == state.last_day:
        return dict(state.retained_caps)
    state.last_day = day

    if day >= c.LIQUIDATION_DAY:
        for species in _CULL_SPECIES:
            state.retained_caps[species] = 0
        return dict(state.retained_caps)

    counts = {
        "COW": census.total_cows,
        "SHEEP": census.total_sheep,
        "GOOSE": census.total_geese,
    }

    for species in _CULL_SPECIES:
        spec = c.ANIMALS[species]
        item = spec["product"]
        lag = spec["first_yield_day"]
        rate = 1.0 / float(spec["interval"])
        projected = max(1, market_inv.get(item, c.I0) - int(drift.get(item, 0.0) * lag))
        fert_price = float(get_price("FERTILIZER", market_inv))
        revenue_day = (market_price(item, projected) * rate) + fert_price
        cost_day = float(get_price("WHEAT", market_inv)) + c.HAND_COST_PER_ANIMAL_DAY
        profit = revenue_day - cost_day

        if species in state.downsized:
            if species not in state.retained_caps:
                state.retained_caps[species] = max(1, math.ceil(counts[species] * 0.5))
            continue

        if counts[species] <= 0:
            state.negative_days[species] = 0
            state.retained_caps[species] = 999
            continue

        if profit < 0.0:
            state.negative_days[species] = state.negative_days.get(species, 0) + 1
            if state.negative_days[species] >= _CULL_CONSECUTIVE_DAYS:
                state.retained_caps[species] = max(1, math.ceil(counts[species] * 0.5))
                state.downsized.add(species)
                state.negative_days[species] = 0
            else:
                state.retained_caps[species] = 999
        else:
            state.negative_days[species] = 0
            state.retained_caps[species] = 999

    return dict(state.retained_caps)


def needs_replan(
    day: int,
    unlocked_shops: list[str],
    unlocked_quads: list[str],
    market_inv: dict[str, int],
    replan: ReplanState,
) -> bool:
    if (
        day != replan.day
        or len(unlocked_shops) != replan.shop_count
        or len(unlocked_quads) != replan.quad_count
    ):
        return True
    for item in c.SELLABLE_ITEMS:
        p_old = replan.prices.get(item, 0)
        p_new = get_price(item, market_inv)
        if p_old > 0 and abs(p_new - p_old) / p_old > 0.25:
            return True
    return False


def mark_replanned(
    day: int,
    unlocked_shops: list[str],
    unlocked_quads: list[str],
    market_inv: dict[str, int],
    replan: ReplanState,
) -> None:
    replan.day = day
    replan.shop_count = len(unlocked_shops)
    replan.quad_count = len(unlocked_quads)
    replan.prices = {item: get_price(item, market_inv) for item in c.SELLABLE_ITEMS}


# ---------------------------------------------------------------------------
# MODULE: market_orders.py
# ---------------------------------------------------------------------------

_WORK_PER_ANIMAL: float = 4.0
_WORK_PER_PLANT: float = 2.5
_WORK_PER_PLANTING: float = 3.5
_WORK_PER_DEPLOY: float = 7.0
_WORK_PER_PENDING_HARVEST: float = 1.5

_THROTTLED_SELL_ITEMS: frozenset[str] = frozenset(("MELON", "WOOL", "STRAWBERRY", "WHEAT"))
_SELL_PRICE_DROP_LIMIT: float = 0.15


def _throttled_qty(item: str, qty: int, market_inv: dict[str, int], day: int) -> int:
    if day >= c.LIQUIDATION_DAY:
        return qty
    if item not in _THROTTLED_SELL_ITEMS:
        return qty
    p_now = get_price(item, market_inv)
    if p_now <= 1:
        return qty
    inv_now = market_inv.get(item, c.I0)
    lo, hi, best = 1, qty, 1
    while lo <= hi:
        mid = (lo + hi) // 2
        p_after = get_price(item, {**market_inv, item: inv_now + mid})
        if (p_now - p_after) / p_now <= _SELL_PRICE_DROP_LIMIT:
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1
    return best


def get_desired_hires(
    day: int,
    n_animals: int,
    n_plants: int,
    seeds_to_plant: int,
    animals_to_deploy: int,
    ready_harvest_tiles: int,
    unlocked_quads_count: int,
) -> int:
    """Calculate workforce needed today from physical task volume and farm layout size."""
    if day < 0 or day >= c.TOTAL_DAYS:
        raise ValueError(f"Invalid day: {day}")
    if day == 0:
        return 5

    work = (
        _WORK_PER_ANIMAL * n_animals
        + _WORK_PER_PLANT * n_plants
        + _WORK_PER_PLANTING * seeds_to_plant
        + _WORK_PER_DEPLOY * animals_to_deploy
        + _WORK_PER_PENDING_HARVEST * ready_harvest_tiles
    )

    efficiency = 11.0 if unlocked_quads_count == 1 else (9.0 if unlocked_quads_count == 2 else 7.5)
    desired = int(math.ceil(work / efficiency))

    if day <= 2:
        desired = max(2, min(desired, 4))
    elif day == c.LIQUIDATION_DAY:
        desired = min(desired, 9)
    elif day > c.LIQUIDATION_DAY:
        desired = min(desired, 8)

    max_hands = 11 if unlocked_quads_count < 3 else 12
    return max(2, min(max_hands, desired))


def _plan_sells(
    day: int,
    hour: int,
    shed: dict[str, int],
    farm_state: dict[str, Any],
    market_inv: dict[str, int],
    inventories: list[dict[str, int]],
    census: AnimalCensus,
    retained_caps: dict[str, int] | None = None,
) -> tuple[list[list[Any]], float]:
    orders: list[list[Any]] = []
    revenue: float = 0.0

    carried_sells: dict[str, int] = defaultdict(int)
    if day >= 29 and hour >= 15:
        for inv in inventories:
            for item, qty in inv.items():
                if qty > 0 and item not in ("COW", "SHEEP", "GOOSE"):
                    carried_sells[item] += qty

    fert_due_count = count_fertilize_due(farm_state)
    fert_in_shed = shed.get("FERTILIZER", 0) + carried_sells.get("FERTILIZER", 0)
    fert_capacity = c.N_FERTILIZER_HANDS * c.FERTILIZER_PICKUP_BATCH
    fert_reserve = 0 if day >= c.LIQUIDATION_DAY else min(fert_due_count, fert_capacity)
    fert_to_sell = max(0, fert_in_shed - fert_reserve)
    if fert_to_sell > 0:
        orders.append(["SELL", "FERTILIZER", fert_to_sell])
        revenue += fert_to_sell * get_price("FERTILIZER", market_inv)

    for item in ("MELON", "STRAWBERRY", "MILK", "WOOL", "EGG", "CARROT", "TOMATO"):
        qty = shed.get(item, 0) + carried_sells.get(item, 0)
        if qty > 0:
            sell_qty = _throttled_qty(item, qty, market_inv, day)
            if sell_qty > 0:
                orders.append(["SELL", item, sell_qty])
                revenue += sell_qty * get_price(item, market_inv)

    wheat_in_shed = shed.get("WHEAT", 0) + carried_sells.get("WHEAT", 0)
    _, sell_ceiling = wheat_feed_thresholds(census, retained_caps, day)
    if wheat_in_shed > sell_ceiling and day > 0:
        excess = wheat_in_shed - sell_ceiling
        sell_qty = _throttled_qty("WHEAT", excess, market_inv, day)
        if sell_qty > 0:
            orders.append(["SELL", "WHEAT", sell_qty])
            revenue += sell_qty * get_price("WHEAT", market_inv)

    return orders, revenue


def _plan_seed_buy(
    orders: list[list[Any]], crop: str, want: int, budget: float, cost: float
) -> tuple[float, float, int]:
    if want <= 0:
        return budget, cost, 0
    price = c.CROPS[crop]["seed_cost"]
    bought = min(want, int(budget // price))
    if bought <= 0:
        return budget, cost, 0
    orders.append(["BUY_SEED", crop, bought])
    spend = bought * price
    return budget - spend, cost + spend, bought


def plan_market_orders(
    day: int,
    hour: int,
    money: float,
    shed: dict[str, int],
    seeds: dict[str, int],
    market_inv: dict[str, int],
    unlocked_quads: list[str],
    farm_state: dict[str, Any],
    hires_today: int,
    inventories: list[dict[str, int]],
    needed_pastures: list[tuple[int, int]],
    census: AnimalCensus,
    plan: DevelopmentPlan,
    target_crops_today: list[str],
    retained_caps: dict[str, int] | None = None,
    downsized: frozenset[str] = frozenset(),
) -> list[list[Any]]:
    orders: list[list[Any]] = []
    budget = money

    target_quadrants = plan.target_quads

    if day == 0 and hour <= 1:
        if hires_today < 5:
            for _ in range(5 - hires_today):
                orders.append(["HIRE"])
        if shed.get("SHEEP", 0) == 0 and len(farm_state["animals"]) == 0:
            orders.append(["BUY_ANIMAL", "SHEEP", 2])
            orders.append(["BUY_ANIMAL", "COW", 2])
            orders.append(["BUY_SEED", "MELON", 12])
            orders.append(["BUY_SEED", "WHEAT", 7])
            orders.append(["BUY_PRODUCT", "WHEAT", 4])
        return orders[:c.MAX_MARKET_ORDERS]

    sell_orders, revenue = _plan_sells(
        day, hour, shed, farm_state, market_inv, inventories, census, retained_caps=retained_caps
    )
    orders.extend(sell_orders)
    budget += revenue

    # Feed backup — floor only, never the sell ceiling: buying back up to the
    # ceiling every time normal feeding dips below it is exactly the cycle
    # this split is meant to avoid.
    n_kept_feedable = kept_feedable_count(census, retained_caps)
    if n_kept_feedable > 0:
        buy_floor, _ = wheat_feed_thresholds(census, retained_caps, day)
        wheat_avail = shed.get("WHEAT", 0) + sum(inv.get("WHEAT", 0) for inv in inventories)
        if day < c.LIQUIDATION_DAY and wheat_avail < buy_floor:
            wp = get_price("WHEAT", market_inv)
            can_buy = min(buy_floor - wheat_avail, int(budget // max(1, wp)))
            if can_buy > 0:
                orders.append(["BUY_PRODUCT", "WHEAT", can_buy])
                budget -= float(wp * can_buy)

    # Hires — unkept animals need no labour, so they do not count toward workforce demand
    ready_harvest = sum(1 for p in farm_state["plants"] if p["yield_units"] > 0)
    ready_harvest += sum(1 for a in farm_state["animals"] if a["yield_units"] > 0)
    seeds_to_plant = min(
        sum(seeds.get(c_crop, 0) for c_crop in target_crops_today), len(farm_state["empty_tiles"])
    )
    n_kept_animals = (
        min(census.field_cows, retained_caps.get("COW", 999) if retained_caps is not None else 999)
        + min(census.field_sheep, retained_caps.get("SHEEP", 999) if retained_caps is not None else 999)
        + min(census.field_geese, retained_caps.get("GOOSE", 999) if retained_caps is not None else 999)
    )
    desired = get_desired_hires(
        day,
        n_kept_animals,
        len(farm_state["plants"]),
        seeds_to_plant,
        census.in_shed + len(needed_pastures),
        ready_harvest,
        len(unlocked_quads),
    )
    new_hires = max(0, desired - hires_today)
    while new_hires > 0:
        hcost = total_hire_cost(new_hires, hires_today)
        if budget >= hcost:
            orders.extend([["HIRE"] for _ in range(new_hires)])
            budget -= float(hcost)
            break
        new_hires -= 1

    # Animals: buy planned livestock FIRST so urgent early deadlines (e.g. Cows day < 8) are funded
    animal_orders: list[list[Any]] = []
    straw_target = plan.crop_targets.get("STRAWBERRY", 0)
    straw_ready = (
        committed_count(farm_state, seeds, "STRAWBERRY") >= straw_target
        or day > c.STRAWBERRY_LAST_PLANT_DAY
    )
    total_bought_this_tick = 0
    total_grazers_bought_this_tick = 0
    current_grazers = census.total_cows + census.total_sheep
    current_total = census.total_all

    if day >= 3:
        animal_targets = (("COW", plan.cows), ("SHEEP", plan.sheep), ("GOOSE", plan.geese))
        for species, target in animal_targets:
            if species in downsized:
                continue
            cap = retained_caps.get(species, 999) if retained_caps is not None else 999
            if species == "SHEEP" and day < 3:
                continue
            have = census.total_geese if species == "GOOSE" else (census.total_sheep if species == "SHEEP" else census.total_cows)
            effective_target = min(target, cap)
            deficit = effective_target - have
            if deficit <= 0:
                continue
            price = c.ANIMALS[species]["cost"]
            payback = c.MIN_PAYBACK_DAYS
            if day > c.TOTAL_DAYS - c.ANIMALS[species]["first_yield_day"] - payback:
                continue
            remaining_herd_slots = max(0, c.MAX_TOTAL_HERD - (current_total + total_bought_this_tick))
            if species in ("COW", "SHEEP"):
                remaining_grazer_slots = max(0, c.MAX_TOTAL_HERD - (current_grazers + total_grazers_bought_this_tick))
                max_buyable = min(remaining_herd_slots, remaining_grazer_slots)
            else:
                max_buyable = remaining_herd_slots
            want = min(2, deficit, max_buyable, int((budget - 100) // price))
            if want > 0:
                animal_orders.append(["BUY_ANIMAL", species, want])
                budget -= float(want * price)
                total_bought_this_tick += want
                if species in ("COW", "SHEEP"):
                    total_grazers_bought_this_tick += want

    # Seeds: buy seeds with remaining budget (protect $500 reserve so seeds never starve cow purchases)
    needed_cow_deficit = max(0, plan.cows - census.total_cows)
    if day <= 11 and needed_cow_deficit > 0:
        cash_reserve = 500.0
    elif 4 <= day < c.LIQUIDATION_DAY:
        cash_reserve = c.SEED_PURCHASE_CASH_RESERVE
    else:
        cash_reserve = 0.0

    spendable = max(0.0, budget - cash_reserve)
    seed_cost = 0.0
    res_pasture = live_reserved_pastures(c.PASTURE_CLUSTER, unlocked_quads, plan.total_grazers, day=day)
    needed_structs = set(needed_pastures)
    empty_count = sum(
        1 for pos in farm_state["empty_tiles"]
        if pos not in res_pasture and pos not in needed_structs
    )

    # 1. Base feed wheat guarantee
    if 4 <= day <= c.WHEAT_LAST_PLANT_DAY:
        curr_wheat = committed_count(farm_state, seeds, "WHEAT")
        if curr_wheat < census.n_feedable and empty_count > 0:
            spendable, seed_cost, bought = _plan_seed_buy(
                orders, "WHEAT", min(census.n_feedable - curr_wheat, empty_count), spendable, seed_cost
            )
            empty_count = max(0, empty_count - bought)

    # 2. Strawberry priority
    total_straw = committed_count(farm_state, seeds, "STRAWBERRY")
    if day <= c.STRAWBERRY_LAST_PLANT_DAY and total_straw < straw_target and empty_count > 0:
        spendable, seed_cost, bought = _plan_seed_buy(
            orders, "STRAWBERRY", min(straw_target - total_straw, empty_count), spendable, seed_cost
        )
        empty_count = max(0, empty_count - bought)

    # 3. Secondary crops (Carrot / Tomato from crop_targets)
    if plan.crop_targets:
        straw_done = committed_count(farm_state, seeds, "STRAWBERRY")
        for crop, target in plan.crop_targets.items():
            if crop == "STRAWBERRY" or crop not in c.CROPS or target <= 0:
                continue
            if day <= 10 and straw_done < straw_target:
                continue
            spec = c.CROPS[crop]
            lag = spec["first_yield_day"] if spec["is_ongoing"] else spec["max_yield_day"]
            if day > c.TOTAL_DAYS - lag - 1:
                continue
            curr = committed_count(farm_state, seeds, crop)
            if curr < target:
                spendable, seed_cost, bought = _plan_seed_buy(orders, crop, target - curr, spendable, seed_cost)
                empty_count = max(0, empty_count - bought)

    # 4. Continuous wheat flow
    if 4 <= day <= c.WHEAT_LAST_PLANT_DAY:
        if day < c.WHEAT_LAST_PLANT_DAY or (day == c.WHEAT_LAST_PLANT_DAY and hour < 12):
            wheat_buffer = 15
            want_wheat = max(0, empty_count + wheat_buffer - seeds.get("WHEAT", 0))
            spendable, seed_cost, _ = _plan_seed_buy(orders, "WHEAT", want_wheat, spendable, seed_cost)
    elif 1 <= day <= 3:
        want_wheat = max(0, empty_count - seeds.get("WHEAT", 0))
        spendable, seed_cost, _ = _plan_seed_buy(orders, "WHEAT", want_wheat, spendable, seed_cost)

    budget -= seed_cost

    # --- Land: calculate on REMAINING budget (after animals) ---
    land_orders: list[list[Any]] = []
    want_land = target_quadrants > len(unlocked_quads)
    if "NE" not in unlocked_quads and day >= 5 and want_land and budget >= 1000:
        land_orders.append(["BUY_LAND"])
        budget -= 1000.0
    elif "SW" not in unlocked_quads and "NE" in unlocked_quads and day >= 9 and want_land and budget >= 2000:
        land_orders.append(["BUY_LAND"])
        budget -= 2000.0
    elif "SE" not in unlocked_quads and "SW" in unlocked_quads and day >= 12 and target_quadrants >= 4 and budget >= 4000:
        land_orders.append(["BUY_LAND"])
        budget -= 4000.0

    # --- Order Queue: Land appended before Animals for truncation priority ---
    orders.extend(land_orders)
    orders.extend(animal_orders)

    return orders[:c.MAX_MARKET_ORDERS]


# ---------------------------------------------------------------------------
# MODULE: dispatch.py
# ---------------------------------------------------------------------------

_VALUABLE_ITEMS: frozenset[str] = frozenset(("MELON",))
_PREMIUM_RUSH_CROPS: frozenset[str] = frozenset(("MELON",))
_CARGO_RETURN_THRESHOLD: int = 10


def _plant_ready_to_harvest(p: dict[str, Any], day: int) -> bool:
    return p["yield_units"] > 0 and (
        p["is_ongoing"]
        or p["age"] >= p["max_yield_day"]
        or day >= 28
    )


def _has_sellable_cargo(inv: dict[str, int], animals_unfed: bool, fert_due: bool) -> bool:
    if animals_unfed and inv.get("WHEAT", 0) > 0:
        return False
    if fert_due and inv.get("FERTILIZER", 0) > 0:
        return False
    return sum(inv.get(item, 0) for item in _VALUABLE_ITEMS) >= _CARGO_RETURN_THRESHOLD


def _should_return_to_shed(
    inv: dict[str, int],
    animals_unfed: bool,
    fert_due: bool,
    has_empty_structures: bool,
) -> bool:
    if sum(inv.values()) == 0:
        return False
    if animals_unfed and inv.get("WHEAT", 0) > 0:
        return False
    if fert_due and inv.get("FERTILIZER", 0) > 0:
        return False
    if has_empty_structures and any(inv.get(a, 0) > 0 for a in ("COW", "SHEEP", "GOOSE")):
        return False
    return True


@dataclass(slots=True)
class Task:
    priority: int
    pos: tuple[int, int]
    action: list[Any]
    need: tuple[str, ...] | None = None


def select_target_crops(day: int, seeds_stock: dict[str, int]) -> list[str]:
    if day > c.WHEAT_LAST_PLANT_DAY:
        return []
    if 21 <= day <= c.WHEAT_LAST_PLANT_DAY:
        return ["WHEAT"]
    crops: list[str] = []
    if day <= 12 and seeds_stock.get("MELON", 0) > 0:
        crops.append("MELON")
    if day <= c.STRAWBERRY_LAST_PLANT_DAY and seeds_stock.get("STRAWBERRY", 0) > 0:
        crops.append("STRAWBERRY")
    if 13 <= day <= c.MELON_LAST_PLANT_DAY and seeds_stock.get("MELON", 0) > 0:
        crops.append("MELON")
    crops.append("WHEAT")
    return crops


def compute_needed_pastures(
    farm_state: dict[str, Any],
    census: AnimalCensus,
) -> list[tuple[int, int]]:
    needed: list[tuple[int, int]] = []
    planted_positions = {p["pos"]: p for p in farm_state["plants"]}
    occupied_pastures = census.field_cows + census.field_sheep
    for p_pos in c.PASTURE_CLUSTER:
        if len(needed) + occupied_pastures + len(farm_state["empty_pastures"]) >= census.total:
            break
        if (
            p_pos in farm_state["empty_tiles"]
            or p_pos in farm_state["weeds"]
            or p_pos in planted_positions
        ):
            needed.append(p_pos)
    return needed


def _add_pickup_tasks(
    tasks: list[Task],
    priority: int,
    item: str,
    qty: int,
    shed_subset: list[tuple[int, int]] | None = None,
) -> None:
    for shed_pos in (shed_subset if shed_subset is not None else c.SHED_TILES):
        tasks.append(Task(priority, shed_pos, ["PICKUP", item, qty]))


def _resolve_structure_action(
    pos: tuple[int, int],
    farm_state: dict[str, Any],
    planted_positions: dict[tuple[int, int], dict[str, Any]],
    build_action: str,
) -> list[Any] | None:
    if pos in farm_state["weeds"]:
        return ["DIG"]
    if pos in planted_positions:
        p = planted_positions[pos]
        if p["yield_units"] > 0 or (not p["is_ongoing"] and p["age"] >= p["first_yield_day"] - 1):
            return None
        return ["DIG"]
    if pos in farm_state["empty_tiles"]:
        return [build_action]
    return None


def _build_tasks(
    farm_state: dict[str, Any],
    shed: dict[str, int],
    seeds_stock: dict[str, int],
    needed_pastures: list[tuple[int, int]],
    all_units: list[tuple[tuple[int, int], dict[str, int]]],
    day: int,
    hour: int,
    plan: DevelopmentPlan,
    unlocked_quads: list[str],
    retained_caps: dict[str, int] | None = None,
) -> list[Task]:
    tasks: list[Task] = []

    # 0. Identify kept animal positions based on retained_caps (closest to shed / fed today)
    kept_animals: set[tuple[int, int]] = set()
    for species in ("COW", "SHEEP", "GOOSE"):
        cap = retained_caps.get(species, 999) if retained_caps is not None else 999
        if cap > 0:
            species_animals = [a for a in farm_state["animals"] if a["animal"] == species]
            species_animals.sort(key=lambda a: (not a["fed_today"], manhattan(a["pos"], (4, 4))))
            for a in species_animals[:cap]:
                kept_animals.add(a["pos"])

    # 1. Animal care
    for a in farm_state["animals"]:
        pos = a["pos"]
        if pos not in kept_animals:
            # Unkept/downsized animal: do NOT feed/care so it leaves over 2 turns,
            # but collect any pending yield or free fertilizer.
            if a["yield_units"] > 0:
                tasks.append(Task(205, pos, ["HARVEST"]))
            if a["fertilizer_available"]:
                tasks.append(Task(210, pos, ["COLLECT_FERTILIZER"]))
            continue
        if not a["fed_today"]:
            feed_pri = 350 if a["consecutive_unfed"] >= 1 else 260
            tasks.append(Task(feed_pri, pos, ["FEED"], need=("WHEAT",)))
        if not a["cared_today"]:
            tasks.append(Task(230, pos, ["CARE"]))
        if a["fertilizer_available"]:
            tasks.append(Task(210, pos, ["COLLECT_FERTILIZER"]))
        if a["yield_units"] > 0:
            tasks.append(Task(205, pos, ["HARVEST"]))

    # 2. Animal placement & pickups
    n_cow_shed, n_sheep_shed = shed.get("COW", 0), shed.get("SHEEP", 0)
    n_goose_shed = shed.get("GOOSE", 0)
    cow_cap = retained_caps.get("COW", 999) if retained_caps is not None else 999
    sheep_cap = retained_caps.get("SHEEP", 999) if retained_caps is not None else 999
    goose_cap = retained_caps.get("GOOSE", 999) if retained_caps is not None else 999

    field_cows = sum(1 for a in farm_state["animals"] if a["animal"] == "COW")
    field_sheep = sum(1 for a in farm_state["animals"] if a["animal"] == "SHEEP")
    field_geese = sum(1 for a in farm_state["animals"] if a["animal"] == "GOOSE")

    has_carried_cow = any(inv.get("COW", 0) > 0 for _, inv in all_units) and field_cows < cow_cap
    has_carried_sheep = any(inv.get("SHEEP", 0) > 0 for _, inv in all_units) and field_sheep < sheep_cap
    carried_goose_count = sum(1 for _, inv in all_units if inv.get("GOOSE", 0) > 0) if field_geese < goose_cap else 0
    geese_waiting = (n_goose_shed if field_geese < goose_cap else 0) + carried_goose_count

    keep_cow_shed = n_cow_shed > 0 and field_cows < cow_cap
    keep_sheep_shed = n_sheep_shed > 0 and field_sheep < sheep_cap
    keep_goose_shed = n_goose_shed > 0 and field_geese < goose_cap

    if (keep_cow_shed or keep_sheep_shed or has_carried_cow or has_carried_sheep) and farm_state["empty_pastures"]:
        for pos in farm_state["empty_pastures"]:
            tasks.append(Task(300, pos, ["PLACE"], need=("COW", "SHEEP")))

    if (keep_goose_shed or carried_goose_count > 0) and farm_state["empty_coops"]:
        for pos in farm_state["empty_coops"]:
            tasks.append(Task(300, pos, ["PLACE"], need=("GOOSE",)))

    if (keep_cow_shed or keep_sheep_shed) and farm_state["empty_pastures"]:
        _add_pickup_tasks(tasks, 185, "COW" if keep_cow_shed else "SHEEP", 1)

    if keep_goose_shed and farm_state["empty_coops"]:
        _add_pickup_tasks(tasks, 185, "GOOSE", 1)

    # 3. Crops Harvesting (Priority 205 — same as animal harvest)
    for p in farm_state["plants"]:
        if p["is_expired"]:
            tasks.append(Task(220, p["pos"], ["DIG"]))
        if _plant_ready_to_harvest(p, day):
            pri = 305 if p["crop"] in _PREMIUM_RUSH_CROPS else 205
            tasks.append(Task(pri, p["pos"], ["HARVEST"]))
        if not p["watered_today"] and not p["is_expired"]:
            tasks.append(Task(200, p["pos"], ["WATER"]))
        if p["fertilize_due"]:
            tasks.append(Task(180, p["pos"], ["FERTILIZE"], need=("FERTILIZER",)))

    # 4. Shed Drop Tasks (Priority 160 for travel; Priority 210 if worker is already standing on a shed tile)
    standing_shed_positions = {pos for pos, _ in all_units if pos in c.SHED_TILES_SET}
    for shed_pos in c.SHED_TILES:
        tasks.append(Task(160, shed_pos, ["DROP"]))
        if shed_pos in standing_shed_positions:
            tasks.append(Task(210, shed_pos, ["DROP"]))

    # 5. Fertilizer Pickup (targeted to needed carrier count only)
    fert_due_count = count_fertilize_due(farm_state)
    if fert_due_count > 0 and shed.get("FERTILIZER", 0) > 0:
        carried_fert = sum(inv.get("FERTILIZER", 0) for _, inv in all_units)
        if carried_fert < fert_due_count:
            needed_carriers = min(4, max(1, math.ceil((fert_due_count - carried_fert) / 4.0)))
            _add_pickup_tasks(tasks, 178, "FERTILIZER", 4, shed_subset=c.SHED_TILES[:needed_carriers])

    # 6. Weeds
    if day <= c.WHEAT_LAST_PLANT_DAY:
        for pos in farm_state["weeds"]:
            tasks.append(Task(150, pos, ["DIG"]))

    # 7. Crop planting
    if hour < c.TURNS_PER_DAY - 1:
        res_pasture = live_reserved_pastures(c.PASTURE_CLUSTER, unlocked_quads, plan.total_grazers, day=day)
        needed_structs = set(needed_pastures)
        plantable = [
            pos for pos in farm_state["empty_tiles"]
            if pos not in res_pasture and pos not in needed_structs
        ]
        plantable.sort(key=lambda p: (manhattan(p, (4, 4)), p[1], p[0]))

        target_crops = select_target_crops(day, seeds_stock)

        # 1. MELON gets highest priority for plantable land (day 0-12 rush / high value)
        if "MELON" in target_crops and plantable:
            n_melon = min(len(plantable), seeds_stock.get("MELON", 0))
            if n_melon > 0:
                for pos in plantable[:n_melon]:
                    tasks.append(Task(140, pos, ["PLANT", "MELON"]))
                plantable = plantable[n_melon:]

        # 2. STRAWBERRY gets second priority before secondary pivot crops
        if "STRAWBERRY" in target_crops and plantable:
            n_straw = min(len(plantable), seeds_stock.get("STRAWBERRY", 0))
            if n_straw > 0:
                for pos in plantable[:n_straw]:
                    tasks.append(Task(140, pos, ["PLANT", "STRAWBERRY"]))
                plantable = plantable[n_straw:]

        # 3. Pivot crops (Carrot / Tomato) from plan.crop_targets
        if plan.crop_targets:
            for pivot_crop in ("CARROT", "TOMATO"):
                if not plantable:
                    break
                target = plan.crop_targets.get(pivot_crop, 0)
                available = seeds_stock.get(pivot_crop, 0)
                if target <= 0 or available <= 0:
                    continue
                n_pivot = min(len(plantable), available, target)
                for pos in plantable[:n_pivot]:
                    tasks.append(Task(138, pos, ["PLANT", pivot_crop]))
                plantable = plantable[n_pivot:]

        # 4. Remaining target crops (e.g. Wheat)
        for crop in target_crops:
            if crop in ("MELON", "STRAWBERRY"):
                continue
            if not plantable:
                break
            n_plant = min(len(plantable), seeds_stock.get(crop, 0))
            if n_plant <= 0:
                continue
            for pos in plantable[:n_plant]:
                tasks.append(Task(140, pos, ["PLANT", crop]))
            plantable = plantable[n_plant:]

    # 8. Pastures & Coops construction
    planted_positions = {p["pos"]: p for p in farm_state["plants"]}
    p_pri = 200 if (n_cow_shed > 0 or n_sheep_shed > 0) else 120
    for pos in needed_pastures:
        act = _resolve_structure_action(pos, farm_state, planted_positions, "BUILD_PASTURE")
        if act is not None:
            tasks.append(Task(p_pri, pos, act))

    needed_new_coops = max(0, geese_waiting - len(farm_state["empty_coops"]))
    if needed_new_coops > 0:
        res_pasture = live_reserved_pastures(c.PASTURE_CLUSTER, unlocked_quads, plan.total_grazers, day=day)
        needed_structs = set(needed_pastures)
        coop_candidates = [
            pos for pos in farm_state["empty_tiles"]
            if pos not in res_pasture and pos not in needed_structs
        ]
        coop_candidates.sort(key=lambda p: manhattan(p, (4, 4)))
        for pos in coop_candidates[:needed_new_coops]:
            tasks.append(Task(200, pos, ["BUILD_COOP"]))

    # 9. Feed pickup (only feed kept animals)
    unfed = sum(1 for a in farm_state["animals"] if not a["fed_today"] and a["pos"] in kept_animals)
    if unfed > 0 and shed.get("WHEAT", 0) > 0:
        carried_feeders = sum(1 for _, inv in all_units if inv.get("WHEAT", 0) > 0)
        needed_feeders = min(4, max(1, math.ceil(unfed / 4.0)))
        if carried_feeders < needed_feeders:
            _add_pickup_tasks(tasks, 250, "WHEAT", 4, shed_subset=c.SHED_TILES[: needed_feeders - carried_feeders])

    return tasks


def _assign_tasks(
    active_units: list[tuple[int, tuple[int, int], dict[str, int]]],
    tasks: list[Task],
    animals_unfed: bool,
    fert_due: bool,
) -> dict[int, Task]:
    def eligible(inv: dict[str, int], task: Task) -> bool:
        if task.action[0] == "DROP":
            return _has_sellable_cargo(inv, animals_unfed, fert_due)
        if sum(inv.get(i, 0) for i in _VALUABLE_ITEMS) >= _CARGO_RETURN_THRESHOLD:
            return False
        if task.action[0] == "PICKUP" and len(task.action) >= 2:
            item = task.action[1]
            if inv.get(item, 0) > 0:
                return False
            if item in ("COW", "SHEEP", "GOOSE") and any(inv.get(a, 0) > 0 for a in ("COW", "SHEEP", "GOOSE")):
                return False
        return task.need is None or any(inv.get(item, 0) > 0 for item in task.need)

    assigned: dict[int, Task] = {}
    claimed_units: set[int] = set()
    claimed_pos: set[tuple[int, int]] = set()

    for ui, pos, inv in active_units:
        standing = [t for t in tasks if t.pos == pos and t.pos not in claimed_pos and eligible(inv, t)]
        if standing:
            best_t = max(standing, key=lambda t: t.priority)
            assigned[ui] = best_t
            claimed_units.add(ui)
            claimed_pos.add(pos)

    for ui, pos, inv in active_units:
        if ui in claimed_units:
            continue
        available = [t for t in tasks if t.pos not in claimed_pos and eligible(inv, t)]
        if not available:
            continue
        best_t = max(available, key=lambda t: (t.priority // 20, -manhattan(pos, t.pos), t.priority))
        assigned[ui] = best_t
        claimed_units.add(ui)
        claimed_pos.add(best_t.pos)

    return assigned


def _to_action(
    pos: tuple[int, int],
    inv: dict[str, int],
    task: Task,
    step_to: Callable[[tuple[int, int]], list[Any]],
) -> list[Any]:
    if pos != task.pos:
        return step_to(task.pos)
    if task.action[0] == "PLACE":
        candidate_animals = task.need if task.need is not None else ("COW", "SHEEP", "GOOSE")
        for animal in candidate_animals:
            if inv.get(animal, 0) > 0:
                return ["PLACE", animal]
        return ["PASS"]
    return task.action


def dispatch_units(
    all_units: list[tuple[tuple[int, int], dict[str, int]]],
    farm_state: dict[str, Any],
    shed: dict[str, int],
    seeds: dict[str, int],
    day: int,
    hour: int,
    needed_pastures: list[tuple[int, int]],
    plan: DevelopmentPlan,
    unlocked_quads: list[str],
    retained_caps: dict[str, int] | None = None,
) -> list[list[Any]]:
    evac_actions: dict[int, list[Any]] = {}
    if day >= 29 and hour >= 15:
        for ui, (pos, inv) in enumerate(all_units):
            if any(inv.get(item, 0) > 0 for item in c.SELLABLE_ITEMS):
                evac_actions[ui] = go_shed_or_drop(pos)

    tasks = _build_tasks(
        farm_state, shed, dict(seeds), needed_pastures,
        all_units, day, hour, plan=plan, unlocked_quads=unlocked_quads,
        retained_caps=retained_caps,
    )
    active_units = [
        (ui, pos, inv)
        for ui, (pos, inv) in enumerate(all_units)
        if ui not in evac_actions
    ]

    kept_animals: set[tuple[int, int]] = set()
    for species in ("COW", "SHEEP", "GOOSE"):
        cap = retained_caps.get(species, 999) if retained_caps is not None else 999
        if cap > 0:
            species_animals = [a for a in farm_state["animals"] if a["animal"] == species]
            species_animals.sort(key=lambda a: (not a["fed_today"], manhattan(a["pos"], (4, 4))))
            for a in species_animals[:cap]:
                kept_animals.add(a["pos"])

    animals_unfed = any(
        not a["fed_today"] and a["pos"] in kept_animals for a in farm_state["animals"]
    )
    fert_due = count_fertilize_due(farm_state) > 0

    assigned = _assign_tasks(active_units, tasks, animals_unfed, fert_due)

    actions: list[list[Any]] = []
    for ui, (pos, inv) in enumerate(all_units):
        if ui in evac_actions:
            actions.append(evac_actions[ui])
            continue

        task = assigned.get(ui)
        if task is not None:
            step_to = lambda target, _p=pos: [bfs_step(_p, target) or "PASS"]
            actions.append(_to_action(pos, inv, task, step_to))
            continue

        has_empty_structures = bool(farm_state["empty_pastures"] or farm_state["empty_coops"])
        if _should_return_to_shed(inv, animals_unfed, fert_due, has_empty_structures):
            actions.append(go_shed_or_drop(pos))
            continue
        actions.append(["PASS"])

    return actions


# ---------------------------------------------------------------------------
# MODULE: main.py
# ---------------------------------------------------------------------------

"""Kaggle Agent Entry Point — EcoBot v3 Modular Runner."""





@dataclass(slots=True)
class AgentMemory:
    plan: DevelopmentPlan = field(default_factory=DevelopmentPlan)
    replan: ReplanState = field(default_factory=ReplanState)
    drift: DriftState = field(default_factory=DriftState)
    cull: CullState = field(default_factory=CullState)


_MEMORY = AgentMemory()


def agent(obs: dict[str, Any]) -> dict[str, Any]:
    global _MEMORY
    if obs.get("step", 0) == 0:
        _MEMORY = AgentMemory()

    player = obs["player"]
    day = obs["day"]
    hour = obs["hour"]

    me = obs["farms"][player]
    private = obs["private"]
    market = obs["market"]
    unlocked_shops = obs["town"]["unlocked_shops"]

    money = float(me["money"])
    tiles = me["tiles"]
    unlocked_quads = me["unlocked_quadrants"]
    shed = private["shed"]
    seeds = private["seeds"]
    market_inv = market["inventory"]
    inventories = private["inventories"]
    hires_today = me["hires_today"]

    # 1. State parsing & census
    farm_state = parse_farm_state(tiles, day)
    census = count_animal_census(farm_state, shed, inventories)

    # 2. Economy & Replan (Pure functions with explicit state tracking)
    demand = compute_daily_demand(unlocked_shops)
    observed = update_drift_tracker(day, market_inv, _MEMORY.drift)
    drift = effective_drift(demand, observed)

    # Economic downsizing: species whose feed+labour cost exceeds the value of
    # their production are downsized by 50% once per season to save grain & labour.
    retained_caps = update_cull_state(day, drift, market_inv, census, _MEMORY.cull)
    downsized_species = frozenset(_MEMORY.cull.downsized)

    if needs_replan(day, unlocked_shops, unlocked_quads, market_inv, _MEMORY.replan):
        _MEMORY.plan = build_development_plan(
            day, money, demand, drift, market_inv, farm_state, census,
            unlocked_quads, unlocked_shops, retained_caps=retained_caps,
            downsized=downsized_species,
        )
        mark_replanned(day, unlocked_shops, unlocked_quads, market_inv, _MEMORY.replan)

    # 3. Market Orders
    needed_pastures = compute_needed_pastures(farm_state, census)
    target_crops_today = select_target_crops(day, seeds)

    market_orders = plan_market_orders(
        day, hour, money, shed, seeds, market_inv,
        unlocked_quads, farm_state, hires_today, inventories,
        needed_pastures, census, _MEMORY.plan, target_crops_today,
        retained_caps=retained_caps,
        downsized=downsized_species,
    )

    # 4. Units & Field Actions
    farmer_pos = (me["farmer"][0], me["farmer"][1])
    all_units = [(farmer_pos, inventories[0])]
    all_units.extend(((h[0], h[1]), inventories[idx + 1]) for idx, h in enumerate(me["hands"]))

    actions = dispatch_units(
        all_units, farm_state, shed, seeds, day, hour, needed_pastures,
        plan=_MEMORY.plan, unlocked_quads=unlocked_quads,
        retained_caps=retained_caps,
    )

    return {
        "farmer": actions[0],
        "hands": actions[1:],
        "market": market_orders,
    }
