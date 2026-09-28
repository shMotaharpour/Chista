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

from agent.belief.ladder import sell_coins, sell_coins_vec, split_days
from agent.belief.market import (MarketForecast, PRODUCTS, TURNS_PER_DAY,
                                 _hourly_rows, _walk_rows)


def _index(item: str) -> int:
    """The good's column in the market's own order, or a named error."""
    try:
        return PRODUCTS.index(item)
    except ValueError as exc:                      # pragma: no cover - caller bug
        raise KeyError(f"{item!r} is not a market good ({PRODUCTS})") from exc


def _walk_row_of(fc: MarketForecast, day: int, hour: int) -> int:
    """The walk row a sale at (absolute day, hour) is quoted at.

    `_hourly_rows`'s own expression resolved for ONE cell instead of built as a
    table: row `rel * 24 + hour` of it is
    `max(0, (first_day + rel) * 24 - step + hour)`, clamped to the walk's last
    row (rel = the day's offset from the forecast's first day; for rel 0 the
    expression reduces to `max(0, hour - hour_now)`, the snapshot rule the
    docstring of `_hourly_rows` states). The depth surface asks this per cell,
    so rebuilding the table per cell was the table's own cost paid 200k times;
    `tests/test_market_hourly.py` pins the two against each other.
    """
    rel = max(0, int(day) - int(fc.first_day))
    row = ((int(fc.first_day) + rel) * TURNS_PER_DAY - int(fc.walk_step)
           + (int(hour) % TURNS_PER_DAY))
    return max(0, min(row, len(fc.walk_inventory) - 1))


def inventory_at(fc: MarketForecast, item: str, day: int,
                 hour: int | None = None) -> int:
    """The market's inventory for `item` on that day (or at that hour)."""
    if hour is None:
        return int(fc.inventory_of(item, day))
    return int(_walk_rows(fc)[_walk_row_of(fc, day, hour)][_index(item)])


def depth_coins(fc: MarketForecast, item: str, day: int, units: int,
                *, hour: int | None = None, pad: float = 0.0) -> int:
    """Coins `units` fetch selling into that day's (or hour's) inventory.

    The engine's own ladder, floor stall included — `ladder.sell_coins` is the
    only definition of it in the agent.

    `pad` walks the same ladder from a FULLER market: `opponent.quantile_price_floor`
    names risk as exactly this — the drain falling `z·sd` short is the curve read
    `z·sd` units higher. Zero is the mean ladder, bit-identical to the model
    before this parameter existed.
    """
    return int(sell_coins(item, inventory_at(fc, item, day, hour) + int(pad),
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

    One pass per good: the cells are the walk's own rows, sampled once
    (`_hourly_rows`), and the marginal is `ladder.marginal_coins_vec` over that
    surface — a cell at a time was 5 ms per day of horizon, this is the same
    numbers without the per-cell call.
    """
    from agent.belief.ladder import marginal_coins_vec
    from agent.belief.market import TURNS_PER_DAY

    goods = tuple(items)
    planned = dict(sold or {})
    days = max(1, int(days))
    rows = _hourly_rows(fc, days)
    walk = _walk_rows(fc)
    steps = (int(first_day) * TURNS_PER_DAY
             + np.arange(days * TURNS_PER_DAY, dtype=np.int64))
    out: dict = {}
    for good in goods:
        inv = walk[np.asarray(rows), _index(good)]
        if planned:
            mine = np.array([int(planned.get((good, int(s)), 0)) for s in steps],
                            dtype=np.int64)
        else:
            mine = np.zeros(len(steps), dtype=np.int64)
        vals = marginal_coins_vec(good, inv, mine)
        out.update({(good, int(s)): float(v) for s, v in zip(steps, vals)})
    return out


def rival_risk(supply: dict, items: Iterable[str], first_day: int,
               days: int) -> dict:
    """`risk(good, hour)`: the units the RIVAL is expected to put in that turn.

    The same shape as `hourly_value`, and the other half of the ranking: the
    engine quotes both players at one index before committing either, so a sale
    the rival is also making that turn shares the price and is worth less than the
    quote says. `supply` is the manager's dated rival supply —
    `{absolute_step: {good: units}}`, already gated to the days their board says
    have goods — and a turn they are absent from is zero risk, which is the honest
    reading of an empty board rather than a gap for priors to fill.
    """
    goods = set(items)
    out: dict = {}
    for d in range(max(1, int(days))):
        day = int(first_day) + d
        for h in range(24):
            step = day * 24 + h
            basket = (supply or {}).get(step) or {}
            for good, units in basket.items():
                if good in goods:
                    out[(good, step)] = int(units)
    return out


def _cell_inventories(fc: MarketForecast, item: str, first_day: int, days: int,
                      *, hour: int | None = None, hours=None, gi: int = 0
                      ) -> np.ndarray:
    """The day's inventory for one good over `days` days, `(days,)`.

    The same read `inventory_at(fc, item, day, hour)` makes, for every day at
    once: the day-start table when no hour is asked for, else the walk row of
    `_walk_row_of` at that day's hour (`hours[gi, d]` when an envelope is handed
    in). Values are truncated to whole units exactly as `inventory_at` does.
    """
    day_ix = [int(day) for day in range(int(first_day), int(first_day) + days)]
    if hours is None and hour is None:
        return np.array([int(fc.inventory_of(item, day)) for day in day_ix],
                        dtype=np.float64)
    if hours is not None:
        per_day = [int(np.asarray(hours)[gi, d]) for d in range(days)]
    else:
        per_day = [int(hour)] * days
    walk = _walk_rows(fc)
    i = _index(item)
    return np.array([int(walk[_walk_row_of(fc, day, h)][i])
                     for day, h in zip(day_ix, per_day)], dtype=np.float64)


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
    goods = tuple(goods)
    n_goods = len(goods)
    nd = max(1, int(days))
    prices = np.zeros((n_goods, nd), dtype=np.int64)
    hours = np.zeros((n_goods, nd), dtype=np.int64)
    if not n_goods:
        return prices, hours
    # ONE table for every good, then one argmax over the hour axis. The old
    # shape asked `hourly_prices` for a good at a time — nine walks of the same
    # market and nine Python passes over the days — and `np.argmax` takes the
    # FIRST maximum, which is the hour the per-day loop picked.
    table = hourly_prices(fc, days=nd, items=goods)
    per_day = table.reshape(nd, TURNS_PER_DAY, n_goods)
    hours = np.argmax(per_day, axis=1).astype(np.int64)          # (nd, n_goods)
    prices = np.take_along_axis(per_day, hours[:, None, :], axis=1)[:, 0, :]
    # The caller's own layout is (n_goods, days): the table is day-major.
    return prices.T.copy(), hours.T.copy()


def sell_blocks(fc: MarketForecast, goods, first_day: int, days: int,
                cap: int, blocks: int = 5, *, hour: int | None = None,
                hours: np.ndarray | None = None,
                pad: np.ndarray | None = None
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

    `pad` is one number per good (the caller's own order) added to the market's
    inventory before the ladder is walked: the conservative ladder. It is how a
    risk-averse price reaches the model without a second price anywhere — the
    curve stays `depth_coins`, read from a fuller market. None is the mean ladder.
    """
    cap = max(1, int(cap))
    edges = geometric_edges(cap, blocks)
    n_goods, days = len(goods), max(1, int(days))
    ends = np.asarray(edges, dtype=np.int64)
    prevs = np.concatenate([np.zeros(1, dtype=np.int64), ends[:-1]])
    units = np.zeros((n_goods, days, len(edges)), dtype=np.int64)
    units[:, :] = ends - prevs                    # one split, every cell
    prices = np.zeros((n_goods, days, len(edges)), dtype=np.float64)
    for gi, good in enumerate(goods):
        # the day's inventory for this good at the hour the sell can reach it,
        # walked higher by this good's risk pad: the drain falling z sd short is
        # the ladder walked z sd units higher, which is what depth_coins' own
        # `pad` said before the surface became one vectorised read.
        inv = _cell_inventories(fc, good, first_day, days, hour=hour,
                                hours=hours, gi=gi)
        if pad is not None:
            inv = inv + float(np.asarray(pad)[gi])
        flow = (sell_coins_vec(good, inv[:, None], ends[None, :])
                - sell_coins_vec(good, inv[:, None], prevs[None, :]))
        prices[gi] = flow / np.maximum(units[gi], 1)
    return units, prices
