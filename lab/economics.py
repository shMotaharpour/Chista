"""Crop / animal / land economics for Kaggriculture.

All math from README tables + kaggriculture source (docs/kaggriculture-source.md).
Outputs the numbers that become agent v1 rule constants.
"""
from __future__ import annotations

from lab.prices import price, MARKET_I0

CROPS = {
    "WHEAT":      {"seed": 10,  "first_yield_day": 2,  "max_yield_day": 4,  "interval": 0, "max_yield": 6, "ongoing": False, "unfert_yield": 4},
    "CARROT":     {"seed": 20,  "first_yield_day": 2,  "max_yield_day": 3,  "interval": 0, "max_yield": 4, "ongoing": False, "unfert_yield": 3},
    "TOMATO":     {"seed": 50,  "first_yield_day": 8,  "max_yield_day": 8,  "interval": 1, "max_yield": 4, "ongoing": True,  "unfert_yield": 4},
    "STRAWBERRY": {"seed": 100, "first_yield_day": 10, "max_yield_day": 10, "interval": 2, "max_yield": 4, "ongoing": True,  "unfert_yield": 4},
    "MELON":      {"seed": 80,  "first_yield_day": 10, "max_yield_day": 12, "interval": 0, "max_yield": 6, "ongoing": False, "unfert_yield": 4},
}

ANIMALS = {
    "GOOSE": {"cost": 300, "structure": "COOP",    "first_yield_day": 4, "interval": 1, "max_held": 4, "product": "EGG"},
    "COW":   {"cost": 400, "structure": "PASTURE", "first_yield_day": 8, "interval": 2, "max_held": 6, "product": "MILK"},
    "SHEEP": {"cost": 500, "structure": "PASTURE", "first_yield_day": 6, "interval": 3, "max_held": 6, "product": "WOOL"},
}

BASE_PRICES = {k: v["base"] for k, v in price.__globals__["MARKET_PARAMS"].items()}


def crop_cycle(crop: str, fertilized: bool = False, start_day: int = 0, season_days: int = 30) -> dict:
    """Optimal single-tile cycle. Returns yield, revenue, days occupied, profit."""
    c = CROPS[crop]
    seed_cost = c["seed"]
    if c["ongoing"]:
        # Scheduled productions at day first_yield + k*interval (1-indexed count).
        productions = []
        count = 0
        day = start_day + c["first_yield_day"]
        while count < c["max_yield"]:
            productions.append(day - start_day)
            day += c["interval"]
            count += 1
        # Fertilized AND watered on production day → 2 per production. Unfertilized → 1.
        per = 2 if fertilized else 1
        total_yield = per * c["max_yield"]
        last_day = productions[-1]
        occupancy = last_day + 1  # dig/replant next day
    else:
        # Watering from bonus window start (ceil(max_yield_day/2)) adds 1/day (2 if fert).
        window_start = (c["max_yield_day"] + 1) // 2
        bonus_days = c["max_yield_day"] - window_start + 1
        rate = 2 if fertilized else 1
        base_yield = 1  # planting day itself yields 1
        total_yield = min(c["max_yield"], base_yield + rate * bonus_days) if fertilized else \
                      min(c["unfert_yield"], base_yield + rate * bonus_days)
        last_day = c["max_yield_day"]
        occupancy = last_day + 1
    revenue = total_yield * BASE_PRICES[crop]
    profit = revenue - seed_cost
    return {
        "crop": crop, "fertilized": fertilized, "yield": total_yield,
        "days": occupancy, "revenue": revenue, "seed_cost": seed_cost,
        "profit": profit, "profit_per_day": profit / occupancy,
        "cycles_in_season": season_days // occupancy,
    }


def animal_economics(animal: str, start_day: int = 0, season_days: int = 30, wheat_price: int = 25) -> dict:
    """Steady-state animal economics incl. feed cost, care bonus, fertilizer byproduct."""
    a = ANIMALS[animal]
    prod = a["product"]
    # Production days: first at first_yield_day, then every interval.
    production_days = list(range(a["first_yield_day"], season_days + 1, a["interval"]))
    base_yield = len(production_days)
    # CARE every day (banked +1 per day, paid on production): approx +1 per production day
    care_bonus = base_yield  # upper bound assuming perfect care
    total_yield = min(a["max_held"], 1) + 0  # placeholder; real: yield accrues to max_held cap between harvests
    # Simplify: harvest daily when possible → yield ≈ base + care bonus, capped by harvest frequency.
    total = base_yield + care_bonus
    feed_cost = wheat_price * season_days  # 1 wheat/day
    fertilizer_revenue = season_days * price("FERTILIZER", MARKET_I0)  # 1/day byproduct
    revenue = total * BASE_PRICES[prod]
    setup = a["cost"]  # structure build cost 1 action (no direct coin cost)
    profit = revenue + fertilizer_revenue - feed_cost - setup
    days = season_days - start_day
    return {
        "animal": animal, "product": prod, "productions": base_yield,
        "total_yield": total, "revenue": revenue,
        "feed_cost": feed_cost, "fertilizer_revenue": fertilizer_revenue,
        "setup_cost": setup, "profit": profit, "profit_per_day": profit / days,
        "breakeven_day": setup / max(1, (BASE_PRICES[prod] + price("FERTILIZER", MARKET_I0) - wheat_price) / max(1, a["interval"])),
    }


def land_value(days_left: int, best_profit_per_day: float) -> dict:
    """Breakeven analysis for BUY_LAND quadrants."""
    out = []
    for name, cost in zip(["NE", "SW", "SE"], [1000, 2000, 4000]):
        tiles = 25
        revenue = tiles * best_profit_per_day * days_left
        out.append({"quadrant": name, "cost": cost, "days_left": days_left,
                    "revenue_potential": revenue, "profit": revenue - cost,
                    "breakeven_days": cost / (tiles * best_profit_per_day) if best_profit_per_day else float("inf")})
    return out


if __name__ == "__main__":
    print("=== CROPS (unfertilized / fertilized) ===")
    for crop in CROPS:
        for fert in (False, True):
            r = crop_cycle(crop, fert)
            print(f"{crop:<11} fert={fert!s:<5} yield={r['yield']} days={r['days']:>2} "
                  f"profit=${r['profit']:>5} per_day=${r['profit_per_day']:.2f} cycles={r['cycles_in_season']}")
    print("\n=== ANIMALS (wheat @$25, full care, 30d) ===")
    for animal in ANIMALS:
        r = animal_economics(animal)
        print(f"{animal:<6} yield={r['total_yield']:>3} revenue=${r['revenue']:>5} "
              f"feed=${r['feed_cost']} fert_rev=${r['fertilizer_revenue']:>4} "
              f"profit=${r['profit']:>5} per_day=${r['profit_per_day']:.2f}")
    print("\n=== LAND (best crop profit/day, day 0) ===")
    best = max(crop_cycle(c, True)["profit_per_day"] for c in CROPS)
    for r in land_value(30, best):
        print(f"{r['quadrant']} cost=${r['cost']} breakeven={r['breakeven_days']:.1f}d profit30d=${r['profit']:.0f}")
