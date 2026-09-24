"""The market's DEPTH: what a lot fetches, and the blocks an LP can price.

`market.py` publishes ONE quote per good per day — the price a single unit
fetches. A plan that sells forty units does not get forty times that quote:
the engine walks the ladder down unit by unit, and `ladder.py` owns that
arithmetic (parity-tested bit-for-bit against the engine). This module is the
JOIN of the two: the walk's inventory at the day (or the hour) a sale lands,
fed through the ladder, so a consumer reads the market's depth instead of its
top-of-book price.

One rule, the package's own: this module ESTIMATES, it never decides. It
publishes coins and prices; the master and the dispatcher own the choices.

Four reads, from the same two sources, never re-derived here:

* `depth_coins(fc, item, day, units, hour=...)` — what `units` fetch, exactly.
* `marginal_price(...)` — what the `units`-th unit fetches (the ladder's own
  marginal, so the curve's first unit is the forecast's own quote).
* `depth_blocks(...)` — the same curve as `blocks` declining-price blocks, for
  an LP that prices volume with a few linear pieces instead of one flat number.
* `best_day_split(...)` — the exact best split of a lot across the days, on the
  walk's own drains (`ladder.split_days`).

Why a flat fraction of the quote will not do, measured on the day-0 board
(inventory 10'000, engine quote in brackets): the first 100 MILK units average
62.0 against a 160 quote, and the first 200 MELON units 132.6 against 250. The
same 0.5 factor is therefore too rich for milk and too poor for melon — a flat
fraction is neither a bound nor an estimate. `tests/test_belief_depth.py` pins
both directions.
"""

from __future__ import annotations

import numpy as np

from agent.belief.ladder import sell_coins, split_days
from agent.belief.market import (MarketForecast, PRODUCTS, TURNS_PER_DAY,
                                 _hourly_rows, _walk_rows)


def _index(item: str) -> int:
    """The good's column in the market's own order, or a named error."""
    try:
        return PRODUCTS.index(item)
    except ValueError as exc:                      # pragma: no cover - caller bug
        raise KeyError(f"{item!r} is not a market good ({PRODUCTS})") from exc


def _walk_row_of(fc: MarketForecast, day: int, hour: int) -> int:
    """The walk row a sale at (absolute day, hour) is quoted at."""
    rel = max(0, int(day) - int(fc.first_day))
    rows = _hourly_rows(fc, rel + 1)
    return rows[rel * TURNS_PER_DAY + (int(hour) % TURNS_PER_DAY)]


def inventory_at(fc: MarketForecast, item: str, day: int,
                 hour: int | None = None) -> int:
    """The market's inventory for `item` on that day (or at that hour)."""
    if hour is None:
        return int(fc.inventory_of(item, day))
    return int(_walk_rows(fc)[_walk_row_of(fc, day, hour)][_index(item)])


def depth_coins(fc: MarketForecast, item: str, day: int, units: int,
                *, hour: int | None = None) -> int:
    """Coins `units` fetch selling into that day's (or hour's) inventory.

    The engine's own ladder, floor stall included — `ladder.sell_coins` is the
    only definition of it in the agent.
    """
    return int(sell_coins(item, inventory_at(fc, item, day, hour),
                          max(0, int(units))))


def marginal_price(fc: MarketForecast, item: str, day: int, units: int,
                   *, hour: int | None = None) -> float:
    """What the `units`-th unit fetches (1-based): the ladder's own marginal.

    The curve's first unit is the forecast's own quote — the depth surface is
    the quote's continuation, not a second price.
    """
    n = max(1, int(units))
    return float(depth_coins(fc, item, day, n, hour=hour)
                 - depth_coins(fc, item, day, n - 1, hour=hour))


def hourly_value(fc: MarketForecast, items: Iterable[str], first_day: int,
                 days: int, *, sold: dict | None = None) -> dict:
    """`value(good, hour)`: what ONE MORE unit fetches at that turn.

    The secretary's ordering needs to know which slot is worth competing for, and
    the quote alone cannot say it: the market is a ladder, so the price of the
    next unit at the inventory that hour's walk holds is the number that ranks two
    slots against each other. `hourly_prices` samples the same walk for the quote
    and `hourly_inventory` for the rows behind it, so the two read one surface.

    Keyed by `(good, absolute_step)` — the same shape the manager's plan and the
    rival's dated supply already speak. `sold` is OUR OWN units already planned at
    each `(good, absolute_step)`: the ladder prices the lot we put in, so the value
    of the next unit at a turn starts from what we are already selling there.
    """
    from agent.belief.market import TURNS_PER_DAY

    goods = tuple(items)
    planned = dict(sold or {})
    out: dict = {}
    for d in range(max(1, int(days))):
        day = int(first_day) + d
        for h in range(TURNS_PER_DAY):
            step = day * TURNS_PER_DAY + h
            for good in goods:
                # `units` is OUR OWN volume at that turn, not the market's
                # inventory: the ladder prices the lot we put in. With nothing
                # planned yet the next unit is the hour's own quote, which is the
                # curve's documented first unit (`hourly_prices`).
                mine = int(planned.get((good, step), 0))
                out[(good, step)] = marginal_price(fc, good, day, mine + 1,
                                                   hour=h)
    return out


def depth_blocks(fc: MarketForecast, item: str, day: int, units: int,
                 blocks: int = 3, *, hour: int | None = None
                 ) -> tuple[tuple[int, float], ...]:
    """`units` as `blocks` declining-price blocks: `((units, price), ...)`.

    Each block's price is the EXACT average of the ladder over that block, so
    the blocks' total equals `depth_coins` on the block boundaries — and a
    PARTIAL fill never credits more than the ladder does (the marginals decline
    inside a block, so the first units of a block fetch at least its average).
    An LP whose objective coefficients decline fills the richest block first by
    itself, so this needs no ordering row.

    The block count is the CALLER's modelling choice (an LP with three pieces
    per good per day); this function only reports the curve as pieces.
    """
    units = max(0, int(units))
    blocks = max(1, int(blocks))
    out: list[tuple[int, float]] = []
    start = 0
    for b in range(blocks):
        end = units * (b + 1) // blocks
        k = end - start
        if k > 0:
            coins = (depth_coins(fc, item, day, end, hour=hour)
                     - depth_coins(fc, item, day, start, hour=hour))
            out.append((k, coins / k))
        start = end
    return tuple(out)


def block_revenue(blocks: tuple[tuple[int, float], ...], units: int) -> float:
    """What `blocks` pay for `units` — the evaluator side of `depth_blocks`.

    A consumer that fills a block partially pays that block's own price for
    every unit in it (the LP's shape), which is what makes the comparison
    against the ladder meaningful.
    """
    left = max(0, int(units))
    coins = 0.0
    for k, price in blocks:
        take = min(left, k)
        coins += take * price
        left -= take
        if left <= 0:
            break
    return coins


def best_day_split(fc: MarketForecast, item: str, day: int, lot: int,
                   days: int | None = None
                   ) -> tuple[np.ndarray, int]:
    """The exact best split of `lot` across the days, on the walk's own drains.

    `ladder.split_days` is the DP; the drains are the walk's own day-to-day
    drops, so a plan is split against the market this forecast actually
    modelled rather than against a flat assumption. Days past the horizon
    clamp, so the last day's drain reads 0.
    """
    horizon = fc.days - max(0, int(day) - int(fc.first_day))
    n_days = max(1, horizon if days is None else min(int(days), horizon))
    start = int(fc.inventory_of(item, day))
    drains = np.array(
        [max(0, int(fc.inventory_of(item, day + d))
             - int(fc.inventory_of(item, day + d + 1)))
         for d in range(n_days)], dtype=np.int64)
    return split_days(item, start, max(0, int(lot)), drains)


def geometric_edges(cap: int, blocks: int) -> tuple[int, ...]:
    """Cumulative block boundaries up to `cap`, doubling.

    Fine where a farm actually trades (a handful of units a day) and coarse
    where it does not: the ladder's own slope is steepest at the first units,
    so an even split would price the units that matter with the average of a
    block that reaches into the cheap end.
    """
    cap = max(1, int(cap))
    blocks = max(1, int(blocks))
    edges: list[int] = []
    e = 1
    for _ in range(blocks - 1):
        if e >= cap:
            break
        edges.append(int(e))
        e *= 2
    if not edges or edges[-1] != cap:
        edges.append(cap)
    return tuple(edges)


def day_envelope(fc: MarketForecast, goods, first_day: int, days: int
                 ) -> tuple[np.ndarray, np.ndarray]:
    """The best hour of each day per good: `(prices, hours)`, `(n_goods, days)`.

    The daily layer's one insight into a day: what the best hour of that day
    pays, and which hour it is. It never CHOOSES the hour — the hourly layer
    does, and it can only do better than this number, never worse. Prices come
    from `hourly_prices` (the walk sampled per turn), so this is a read of the
    same surface the hourly layer will use, not a second forecast.
    """
    from agent.belief.market import TURNS_PER_DAY, hourly_prices
    n_goods = len(goods)
    prices = np.zeros((n_goods, max(1, int(days))), dtype=np.int64)
    hours = np.zeros((n_goods, max(1, int(days))), dtype=np.int64)
    for gi, good in enumerate(goods):
        table = hourly_prices(fc, days=max(1, int(days)), items=(good,))
        for d in range(max(1, int(days))):
            row = table[d * TURNS_PER_DAY:(d + 1) * TURNS_PER_DAY, 0]
            if not len(row):
                continue
            h = int(np.argmax(row))
            hours[gi, d] = h
            prices[gi, d] = int(row[h])
    return prices, hours


def sell_blocks(fc: MarketForecast, goods, first_day: int, days: int,
                cap: int, blocks: int = 5, *, hour: int | None = None,
                hours: np.ndarray | None = None
                ) -> tuple[np.ndarray, np.ndarray]:
    """The depth curve of every good and day as LP blocks.

    Returns `(units, prices)`, both `(len(goods), days, blocks)`: `units[g,d,b]`
    is how many units block `b` may take on that day and `prices[g,d,b]` the
    exact average the ladder pays over it. A maximising LP with declining
    prices fills the rich blocks first by itself, so the arrays need no
    ordering rows — and the block total equals `depth_coins` at the boundaries,
    which is what makes the model the curve rather than an approximation of it.

    `cap` bounds one good's sale in one day; the shed's own capacity is the
    honest value, since a day cannot sell more than it can hold.
    """
    cap = max(1, int(cap))
    edges = geometric_edges(cap, blocks)
    n_goods, days = len(goods), max(1, int(days))
    units = np.zeros((n_goods, days, len(edges)), dtype=np.int64)
    prices = np.zeros((n_goods, days, len(edges)), dtype=np.float64)
    for gi, good in enumerate(goods):
        for d in range(days):
            day = int(first_day) + d
            # The hour this good's sale can reach that day: the envelope's own
            # best hour when one is handed in, the caller's fixed hour otherwise.
            h = hour
            if hours is not None:
                h = int(np.asarray(hours)[gi, d])
            prev = 0
            for b, end in enumerate(edges):
                coins = (depth_coins(fc, good, day, end, hour=h)
                         - depth_coins(fc, good, day, prev, hour=h))
                k = int(end) - prev
                units[gi, d, b] = k
                prices[gi, d, b] = (coins / k) if k > 0 else 0.0
                prev = int(end)
    return units, prices
