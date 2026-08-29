"""Market pricing module — exact port of kaggriculture MARKET_PARAMS / market_price.

Source: docs/kaggriculture-source.md. Verified against live env in tests.
"""
from __future__ import annotations

import math

MARKET_I0 = 10_000
PRICE_FLOOR = 1
HINGE_GAIN = 8.0

MARKET_PARAMS = {
    "WHEAT":      {"base":  25, "T": 400, "below_func": "sqrt",   "below_target": 0.80, "above_func": "log",    "above_target": 0.20},
    "CARROT":     {"base":  35, "T": 450, "below_func": "hinge",  "below_target": 1.00, "above_func": "sqrt",   "above_target": 0.70},
    "TOMATO":     {"base":  60, "T": 200, "below_func": "hinge",  "below_target": 0.40, "above_func": "sqrt",   "above_target": 0.60},
    "STRAWBERRY": {"base": 120, "T": 100, "below_func": "sqrt",   "below_target": 0.70, "above_func": "linear", "above_target": 1.60},
    "MELON":      {"base": 250, "T": 300, "below_func": "log",    "below_target": 0.20, "above_func": "sq",     "above_target": 3.60},
    "EGG":        {"base":  50, "T": 332, "below_func": "hinge",  "below_target": 0.40, "above_func": "log",    "above_target": 0.20},
    "MILK":       {"base": 160, "T": 122, "below_func": "sqrt",   "below_target": 0.60, "above_func": "linear", "above_target": 1.60},
    "WOOL":       {"base": 200, "T": 105, "below_func": "log",    "below_target": 0.20, "above_func": "sq",     "above_target": 3.20},
    "FERTILIZER": {"base": 100, "T": 200, "below_func": "linear", "below_target": 0.40, "above_func": "linear", "above_target": 0.40},
}

PRODUCTS = list(MARKET_PARAMS)


def _shape(func: str, x: float, T: float | None = None) -> float:
    x = max(0.0, x)
    if func == "linear":
        return x
    if func == "sq":
        return x * x
    if func == "sqrt":
        return math.sqrt(x)
    if func == "log":
        return math.log(1.0 + x)
    if func == "log10":
        return math.log10(1.0 + x)
    if func == "hinge":
        if not T or T <= 0:
            return x
        u = x / T
        return u + HINGE_GAIN * max(0.0, u - 1.0) ** 2
    return x


def price(item: str, inventory: int, params=None) -> int:
    """Exact port of env market_price(). Floor $1, rounded to int."""
    p = (params or MARKET_PARAMS)[item]
    base, I0, T = p["base"], MARKET_I0, p["T"]
    if inventory < I0:
        f = p["below_func"]
        amp = p["below_target"] * base / _shape(f, T, T)
        val = base + amp * _shape(f, I0 - inventory, T)
    else:
        f = p["above_func"]
        amp = p["above_target"] * base / _shape(f, T, T)
        val = base - amp * _shape(f, inventory - I0, T)
    return max(PRICE_FLOOR, int(round(val)))


def quantity_to_floor(item: str, params=None) -> int:
    """Units sold (added to inventory) from I0 until price hits the $1 floor."""
    inv = MARKET_I0
    while price(item, inv, params) > PRICE_FLOOR:
        inv += 1
        if inv > MARKET_I0 * 10:
            break
    return inv - MARKET_I0


def sell_revenue(item: str, n_units: int, start_inventory: int = MARKET_I0) -> int:
    """Exact env mechanics: one unit at a time, price quoted pre-sell per unit.

    Units sold at $1 do NOT add to inventory (floor stays responsive).
    """
    inv = start_inventory
    total = 0
    for _ in range(n_units):
        p = price(item, inv)
        total += p
        if p > 1:
            inv += 1
    return total
