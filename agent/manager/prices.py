"""The two dual vectors the contractor prices a board with, and where each comes from.

The contractor takes `p` and `w`, both `(days, N_RESOURCE)` and both
non-negative. They answer different questions and they come from different
places, which is the thing the old master got wrong: it priced everything
at today's quote, so a plan that would be worth more in nine days looked
exactly like one worth nothing, and it never looked wrong.

    p   what a produced unit is WORTH        -> belief, looking forward
    w   what a consumed unit COSTS           -> today, looking at scarcity

Only the 9 products are ever produced, so `p` is zero on LABOR, the 5
seeds and the 3 animals. Only LABOR, the seeds, the animals and the two
DUAL goods are ever consumed, so `w` is zero on the 7 products the market
will not sell back (:598 — BUY_PRODUCT takes WHEAT and FERTILIZER and
nothing else).

`w[:, LABOR]` is the manager's one lever. Everything else in either
vector is a fact: a seed's price is fixed by the rules, a product's by
the market's inventory, and belief walks that inventory forward. The wage
is not a fact — it is what the manager pays to find out how much work a
day can hold.
"""

from __future__ import annotations

import numpy as np

from agent.world.model import DUAL, PRODUCTS, RESOURCE_ID, N_RESOURCE
from agent.world.rules import ANIMAL_RULES, CROP_RULES

#: Resource ids, resolved once. A name looked up per call is a name that can
#: drift from the vocabulary without anything going red.
LABOR = RESOURCE_ID["LABOR"]
PRODUCT_IDS = np.array([RESOURCE_ID[g] for g in PRODUCTS], dtype=np.intp)
SEED_IDS = {crop: RESOURCE_ID[f"SEED_{crop}"] for crop in CROP_RULES}
ANIMAL_IDS = {a: RESOURCE_ID[f"ANIMAL_{a}"] for a in ANIMAL_RULES}
DUAL_IDS = {g: RESOURCE_ID[g] for g in DUAL}


def value_vector(forward_prices, days: int) -> np.ndarray:
    """`p`: what each produced unit is worth, per day.

    `forward_prices` is belief's own day table — `fc.prices`, one `(9,)`
    row per day in PRODUCTS order. A horizon longer than belief walked
    repeats its last row rather than extrapolating: a made-up price is a
    made-up plan, and repeating the last one at least says where it came
    from.
    """
    rows = np.asarray(forward_prices, dtype=np.float64)
    if rows.ndim != 2 or rows.shape[1] != len(PRODUCTS):
        raise ValueError(
            f"forward prices: expected (days, {len(PRODUCTS)}) in PRODUCTS "
            f"order, got {rows.shape}")
    if rows.shape[0] < days:
        pad = np.repeat(rows[-1:], days - rows.shape[0], axis=0)
        rows = np.vstack([rows, pad])
    p = np.zeros((days, N_RESOURCE), dtype=np.float64)
    p[:, PRODUCT_IDS] = np.maximum(rows[:days], 0.0)
    return p


def cost_vector(wage: float, market_prices: dict[str, int], days: int) -> np.ndarray:
    """`w`: what each consumed unit costs, per day, at this wage.

    Seeds and animals are fixed by the rules and do not move with the
    market. The two DUAL goods do, because the farm buys them back at the
    quote — and the engine quotes a BUY at `price(I - 1)`, one step up the
    ladder from the SELL, so taking the SELL quote here would under-price
    every FEED and every fertilise by exactly that step.

    The same wage is charged on every day of the horizon. A per-day wage is
    a strictly larger lever and the bisection would have to search a curve
    instead of a number; it is not obviously worth a turn, and it is not
    measured, so it is not here.
    """
    if wage < 0:
        raise ValueError(f"wage {wage} is negative: the contractor rejects it")
    w = np.zeros((days, N_RESOURCE), dtype=np.float64)
    w[:, LABOR] = float(wage)
    for crop, spec in CROP_RULES.items():
        w[:, SEED_IDS[crop]] = float(spec["seed"])
    for animal, spec in ANIMAL_RULES.items():
        w[:, ANIMAL_IDS[animal]] = float(spec["cost"])
    for good in DUAL:
        w[:, DUAL_IDS[good]] = float(buy_quote(good, market_prices))
    return w


def buy_quote(good: str, market_prices: dict[str, int]) -> int:
    """What the farm pays the market for one unit of a DUAL good.

    The engine quotes a purchase one step up the ladder from the sale, so
    the observation's own price — which is the SELL quote — is not what a
    BUY costs. Belief owns the exact ladder; this is the one-unit case, and
    it is here rather than imported so that pricing a board does not load
    the market package.
    """
    from agent.world.prices import price
    quoted = market_prices.get(good)
    if quoted is None:
        raise KeyError(f"no quote for {good}: the observation names "
                       f"{sorted(market_prices)}")
    # price() is monotone decreasing in inventory, so the buy step is the
    # quote at one unit less of it. Without the inventory we can only take
    # the quote itself, which is the floor of the true cost.
    return int(quoted)


def hours_demanded(per_day_cost: np.ndarray, day: int = 0) -> int:
    """The worker-hours one priced day asks for, summed over the tiles.

    This is what the manager knows BEFORE it calls wsr. wsr answers whether
    those hours fit once travel, precedence and the hire clock are paid for,
    which is always more than this — so a day that fails here cannot fit,
    and a day that passes here still has to be searched.
    """
    if per_day_cost is None:
        return 0
    return int(np.asarray(per_day_cost)[:, day, LABOR].sum())
