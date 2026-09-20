"""The market's price function and the town's drain — the engine's arithmetic.

`market_price` (kaggriculture.py:192-206) and `_shape` (:61-74) are the authority;
this is the same formula written once so a plan can be priced without the engine in
the loop. The town's consumption cadence (`_town_consume`, :728-749) is what moves
inventory when nobody trades, and it is the only drain that does not depend on the
opponent.
"""

from __future__ import annotations

import math

import numpy as np

from agent.world.model import PRODUCTS
from agent.world.rules import SHOPS, TOWN_CENTER_PRODUCTS

#: kaggriculture.py:38-39
MARKET_I0: int = 10_000
PRICE_FLOOR: int = 1
#: The quadratic gain of the `hinge` shape (:58).
HINGE_GAIN: float = 8.0

#: MARKET_PARAMS (kaggriculture.py:41-51), verbatim: base price, the anchor
#: throughput T, and an independent shape + target move for each side of I0.
MARKET_PARAMS: dict[str, dict] = {
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


def shape(func: str, x: float, t: float | None = None) -> float:
    """kaggriculture.py:61-74. `log` is ln(1 + x), so every shape has f(0) = 0."""
    x = max(0.0, float(x))
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
        if not t or t <= 0:
            return x
        u = x / t
        return u + HINGE_GAIN * max(0.0, u - 1.0) ** 2
    return x


def price(item: str, inventory: float) -> int:
    """The sale price of one unit at this market inventory, kaggriculture.py:192-206.

    `base` at I0; rises as inventory falls, falls as inventory grows; floored at $1
    and rounded to the nearest dollar.
    """
    p = MARKET_PARAMS[item]
    base = p["base"]
    if inventory < MARKET_I0:
        f = p["below_func"]
        amp = p["below_target"] * base / shape(f, p["T"], p["T"])
        value = base + amp * shape(f, MARKET_I0 - inventory, p["T"])
    else:
        f = p["above_func"]
        amp = p["above_target"] * base / shape(f, p["T"], p["T"])
        value = base - amp * shape(f, inventory - MARKET_I0, p["T"])
    return max(PRICE_FLOOR, int(round(value)))


def prices(inventory: dict[str, int]) -> dict[str, int]:
    """Every product's price at one market inventory."""
    return {item: price(item, inventory[item]) for item in PRODUCTS}


def price_of(item: str, inventory: float) -> int:
    """Scalar alias of `price`, for call sites that hold the item name."""
    return price(item, inventory)


def price_table(inventory: np.ndarray) -> np.ndarray:
    """The (9,) quote vector at one (9,) inventory, in PRODUCTS order."""
    return np.array([price(item, float(x))
                     for item, x in zip(PRODUCTS, inventory, strict=True)],
                    dtype=np.int64)


def price_vec(item: str, inventories: np.ndarray) -> np.ndarray:
    """`price(item, x)` over a vector of inventories, one product at a time."""
    return np.array([price(item, float(x)) for x in np.asarray(inventories).ravel()],
                    dtype=np.int64)


#: How an order is quoted, kaggriculture.py:596-605. Both players are quoted from the
#: same pre-commit inventory, so a buy followed by a sell of one unit against an
#: unchanged market nets exactly zero (README, "Buying inventory from the market").
QUOTE: dict[str, str] = {
    "SELL": "price(item, inventory)",
    "BUY_PRODUCT": "price(item, inventory - 1)",
    "BUY_SEED": "the crop's seed price, fixed",
    "BUY_ANIMAL": "the animal's cost, fixed",
}

#: A sale at the floor does not add supply (:658-660), so the floor stays responsive
#: to the next buy.
FLOOR_SALES_ADD_NO_SUPPLY: bool = True


def town_drain(shops: tuple[str, ...] | list[str], step: int,
               shop_interval: int = 4, center_interval: int = 24) -> dict[str, int]:
    """What the town takes out of the market on one turn (kaggriculture.py:728-749).

    `shops` is `town["unlocked_shops"]` — with replacement, so a name may repeat and
    each copy consumes on its own. A single-product shop consumes 2.
    """
    drain: dict[str, int] = {item: 0 for item in PRODUCTS}
    if step % shop_interval == 0:
        for name in shops:
            products = SHOPS[name]
            each = 2 if len(products) == 1 else 1
            for item in products:
                drain[item] += each
    if step % center_interval == 0:
        for item in TOWN_CENTER_PRODUCTS:
            drain[item] += 1
    return drain


def drain_per_day(shops: tuple[str, ...] | list[str],
                  shop_interval: int = 4, center_interval: int = 24,
                  turns_per_day: int = 24) -> dict[str, int]:
    """The same drain over one whole day, for a plan that thinks in days."""
    total: dict[str, int] = {item: 0 for item in PRODUCTS}
    for step in range(turns_per_day):
        for item, units in town_drain(shops, step, shop_interval, center_interval).items():
            total[item] += units
    return total
