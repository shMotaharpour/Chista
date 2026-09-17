"""The engine's price function, vectorised once.

`kaggriculture.market_price` is the authority: this module is the same
arithmetic over numpy arrays, so a caller that needs a whole curve pays one
pass instead of a Python loop. Derived from the engine, never transcribed
(R002), and pinned by `tests/test_market_analyzer.py::test_price_parity`,
which compares it against `K.market_price` over a wide inventory grid.

The curve: `base ± amp·shape(|I - I0|)` with `I0 = 10000` neutral, floored at
`PRICE_FLOOR = 1`. Below `I0` the town is short and the quote rises; above it
the quote falls. Only `SELL` and `BUY_PRODUCT` read it (F034, F035).
"""

from __future__ import annotations

import numpy as np

from kaggle_environments.envs.kaggriculture import kaggriculture as K


def shape(func: str, x: np.ndarray, T: float | None = None) -> np.ndarray:
    """`kaggriculture._shape`, over arrays."""
    x = np.maximum(0.0, np.asarray(x, dtype=np.float64))
    if func == "linear":
        return x
    if func == "sq":
        return x * x
    if func == "sqrt":
        return np.sqrt(x)
    if func == "log":
        return np.log1p(x)
    if func == "log10":
        return np.log10(1.0 + x)
    if func == "hinge":
        if not T or T <= 0:
            return x
        u = x / T
        return u + K.HINGE_GAIN * np.maximum(0.0, u - 1.0) ** 2
    return x


def price_vec(item: str, inventory: np.ndarray | float,
              params: dict | None = None) -> np.ndarray:
    """Price(s) for `item` at the given market inventory. int64, engine rounding."""
    p = (params or K.MARKET_PARAMS)[item]
    I = np.asarray(inventory, dtype=np.float64)
    base, I0, T = p["base"], p["I0"], p["T"]
    below_amp = p["below_target"] * base / shape(p["below_func"], T, T)
    above_amp = p["above_target"] * base / shape(p["above_func"], T, T)
    below = base + below_amp * shape(p["below_func"], I0 - I, T)
    above = base - above_amp * shape(p["above_func"], I - I0, T)
    price = np.where(I < I0, below, above)
    return np.maximum(K.PRICE_FLOOR, np.round(price)).astype(np.int64)


def price_of(item: str, inventory: float) -> int:
    """Scalar price, for the call sites that quote one unit."""
    return int(price_vec(item, np.array(float(inventory)))[()])


def price_table(inventory: np.ndarray) -> np.ndarray:
    """The (9,) price vector for a whole inventory vector, in `PRODUCTS` order."""
    return np.array([price_of(g, float(inventory[i]))
                     for i, g in enumerate(K.PRODUCTS)], dtype=np.int64)
