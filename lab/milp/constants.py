"""CP-SAT season planner for Kaggriculture (v1.3).

Solves the 30-day season as a constraint program. All-integer model.
Output: day-by-day plan (plant/sell/buy/hire/land).

Usage: python -m lab.milp.cpsat_model [--time 60] [--out plan.json]
"""
from __future__ import annotations

from ortools.sat.python import cp_model

# ---------------------------------------------------------------- constants
DAYS = 30
TURNS_PER_DAY = 24
CROPS = {
    "WHEAT":      {"seed": 10,  "first": 2,  "maxyd": 4,  "interval": 0, "ongoing": False, "base": 25,  "yield_base": 4},
    "CARROT":     {"seed": 20,  "first": 2,  "maxyd": 3,  "ongoing": False, "base": 35,  "yield_base": 3},
    "TOMATO":     {"seed": 50,  "first": 8,  "maxyd": 8,  "interval": 1, "ongoing": True,  "base": 60},
    "STRAWBERRY": {"seed": 100, "first": 10, "maxyd": 10, "interval": 2, "ongoing": True,  "base": 120},
    "MELON":      {"seed": 80,  "first": 10, "maxyd": 12, "ongoing": False, "base": 250, "yield_base": 6},
}
SELLABLE = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER"]
ANIMALS = {
    "GOOSE": {"cost": 300, "first": 4, "interval": 1, "max_held": 4, "product": "EGG",  "base": 50},
    "COW":   {"cost": 400, "first": 8, "interval": 2, "max_held": 6, "product": "MILK", "base": 160},
    "SHEEP": {"cost": 500, "first": 6, "interval": 3, "max_held": 6, "product": "WOOL", "base": 200},
}
LAND_PRICES = [1000, 2000, 4000]
LAND_DAYS = [6, 11, 20]
STARTING_MONEY = 3000
SHED_CAP = 100
QUADRANT_TILES = 25
FORCE_SELL_START = 28  # days 28-29 (0-indexed) → forced full sell
