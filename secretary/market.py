"""Secretary B: the market forecast — inventory forward, price through the engine.

Issue #15. `price(inventory)` is a known function with known per-good
parameters (F034), so forecasting the price is forecasting **market
inventory**, and its drivers are nearly all observable:

- **town consumption** — a deterministic cadence: every unlocked shop
  instance consumes its products every `townShopSellInterval` turns (a
  single-product shop at multiplier 2), and the town centre consumes
  every product but fertilizer every `townCenterSellInterval` turns
  (F037). Measured against the engine over a full episode (probe
  `bench/bench_market_forecast.py`): the cadence model reproduces every
  observed inventory delta exactly — 6,462/6,462 item-step comparisons
  (seed 7, no weeds, PASS policies). It is not an approximation.
- **our sales** — the caller's own schedule, passed in.
- **the opponent's sales** — not directly observable; carried as a
  per-item residual (#16). Default zero, and named: a forecast that
  ignores the other seat's sells is biased toward higher prices.

The one genuinely unknown draw is the *next* shop unlock
(`townShopUnlockInterval` days, drawn with replacement from the engine's
own shop table, capped at `MAX_SHOP_INSTANCES`); the currently unlocked
list is public (`obs["town"]["unlocked_shops"]`). Two policies, both
measurable: `"none"` (the observed set stays — under-counts future
consumption, so it is biased toward *higher* prices) and `"mean"` (the
mean shop's demand is added on each unlock day until the cap). **`mean` is
the default because it is measured better at every horizon past day 3**:
20 seeds, worst item, `bench/bench_market_forecast.py --error` — at day 10
the inventory error is 1.14 % of I0 (none: 1.32 %) and the price error 30
coins (none: 52); at day 20, 2.87 % (none: 3.72 %) and 78 coins (none:
115).

The price itself is never modelled here: `market_price` is imported from
the engine (R002) and called on the forecast inventory exactly as
`_refresh_prices` does after each turn's consumption.

**Turn-order contract, probed and not assumed** (`bench/bench_market_forecast.py
--probe`): the observation at step `s` shows the inventory *after* turn
`s-1`, and the shop set shown there is the set that consumes during turn
`s`. A turn runs unit actions, then `_process_market` (our sells land),
then `_town_consume` (shops, centre, `_refresh_prices`). So the forecast
consumes from the current turn onwards and samples one price per day at
hour 0 — the price a sale at the start of that day faces.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from kaggle_environments.envs.kaggriculture import kaggriculture as K

# The vocabulary and the tables all come from the engine (R002): a shop
# table or a product list edited there must move this module with it.
PRODUCTS: tuple[str, ...] = tuple(K.PRODUCTS)
SHOPS: dict[str, tuple[str, ...]] = {name: tuple(items)
                                     for name, items in K.SHOPS.items()}
TOWN_CENTER_PRODUCTS: tuple[str, ...] = tuple(K.TOWN_CENTER_PRODUCTS)
MAX_SHOP_INSTANCES: int = int(K.MAX_SHOP_INSTANCES)
PRICE_FLOOR: int = int(K.PRICE_FLOOR)

TURNS_PER_DAY = 24          # engine default; checked by verify_engine_timings
UNLOCK_POLICIES = ("none", "mean")

_PROD_INDEX = {item: i for i, item in enumerate(PRODUCTS)}


def _get(config: Any, key: str, default: Any) -> Any:
    """The engine's own configuration lookup (`K.get`), applied to ours."""
    if config is None:
        return default
    try:
        return config.get(key, default)
    except AttributeError:
        return default


def shop_demand(shops: Iterable[str]) -> dict[str, int]:
    """One consumption tick of a shop set: product -> units removed.

    A single-product shop consumes at multiplier 2 (engine `_town_consume`;
    each instance consumes independently, so a repeated shop counts twice).
    """
    out: dict[str, int] = {}
    for name in shops:
        items = SHOPS[name]
        mult = 2 if len(items) == 1 else 1
        for item in items:
            out[item] = out.get(item, 0) + mult
    return out


def town_deltas(shops: Sequence[str], step: int, shop_interval: int,
                center_interval: int) -> dict[str, int]:
    """The town's consumption during turn `step`, exactly as the engine runs it."""
    out: dict[str, int] = {}
    if shop_interval > 0 and step % shop_interval == 0:
        for item, n in shop_demand(shops).items():
            out[item] = out.get(item, 0) + n
    if center_interval > 0 and step % center_interval == 0:
        for item in TOWN_CENTER_PRODUCTS:
            out[item] = out.get(item, 0) + 1
    return out


def mean_shop_demand() -> dict[str, float]:
    """Mean per-INSTANCE demand of one shop draw (the `"mean"` unlock policy).

    Every shop in the engine's table is equally likely (drawn with
    replacement from `sorted(SHOPS)`), so the expected instance is the
    unweighted mean over the table.
    """
    names = sorted(SHOPS)
    total: dict[str, float] = {}
    for name in names:
        for item, n in shop_demand([name]).items():
            total[item] = total.get(item, 0.0) + n
    return {item: n / len(names) for item, n in total.items()}


@dataclass(frozen=True)
class MarketForecast:
    """One forward pass over the market: inventory and price per day start.

    `inventory[d]` / `prices[d]` are tuples in `PRODUCTS` order. Row 0 is
    `first_day` — the season day the forecast was made on — and every
    later row is that day's hour 0. A plan made mid-day is therefore still
    indexed by the day it is IN: row 0 is the state the agent is looking
    at, not tomorrow's day start. `day` arguments are ABSOLUTE season days
    (see `_row`) and are clamped to the horizon, so a caller asking about
    a day past it gets the last modelled day instead of an error.
    """

    start_step: int
    horizon_days: int
    first_day: int                # the season day row 0 belongs to
    inventory: tuple[tuple[int, ...], ...]
    prices: tuple[tuple[int, ...], ...]
    unlock_policy: str = "mean"
    residual: Mapping[str, float] = field(default_factory=dict)
    assumptions: tuple[str, ...] = ()

    @property
    def days(self) -> int:
        return len(self.inventory)

    def _index(self, day: int) -> int:
        """Absolute season day -> row, clamped to the modelled horizon."""
        return min(max(int(day) - int(self.first_day), 0), self.days - 1)

    def _row(self, day: int) -> tuple[int, ...]:
        return self.inventory[self._index(day)]

    def inventory_of(self, item: str, day: int) -> int:
        return self._row(day)[_PROD_INDEX[item]]

    def price_of(self, item: str, day: int) -> int:
        return self.prices[self._index(day)][_PROD_INDEX[item]]

    def price_path(self, item: str) -> tuple[int, ...]:
        return tuple(row[_PROD_INDEX[item]] for row in self.prices)

    def describe(self) -> str:
        return (f"forecast step={self.start_step} days={self.days} "
                f"unlocks={self.unlock_policy} "
                f"residual={len(self.residual)} items")


def _step_of(obs: Any) -> int:
    """The observation's absolute turn index (day * 24 + hour)."""
    if isinstance(obs, dict) and obs.get("step") is not None:
        return int(obs["step"])
    day = int(obs.get("day", 0)) if isinstance(obs, dict) else 0
    hour = int(obs.get("hour", 0)) if isinstance(obs, dict) else 0
    return day * TURNS_PER_DAY + hour


def _apply_sells(inv: dict[str, float], item: str, units: float,
                 params: Any) -> float:
    """A SELL of `units` lands in the market: +1 per unit unless price is 1.

    Engine `_commit_unit`: a sale at the floor price does NOT add supply (the
    floor stays responsive to later buys), one above it does. Returns the
    coins the sale fetched at the walk-down prices (the ladder moves with
    every unit) — that is the price impact F036 describes.
    """
    coins = 0.0
    whole = int(units)
    for _ in range(whole):
        price = K.market_price(item, inv[item], params)
        coins += price
        if price > PRICE_FLOOR:
            inv[item] += 1.0
    frac = float(units) - whole
    if frac > 0.0:                       # a fractional rate: no floor rule
        coins += frac * K.market_price(item, inv[item], params)
        inv[item] += frac
    return coins


def forecast(obs: Any, *, days: int = 30,
             our_sells: Mapping[int, Mapping[str, int]] | None = None,
             residual: Mapping[str, float] | None = None,
             unlock_policy: str = "mean",
             config: Any = None,
             market_params: Any = None,
             ) -> MarketForecast:
    """Forward-simulate market inventory, then price each day start.

    `our_sells` is `{absolute_step: {item: units}}`; `residual` is the
    opponent's sell pressure in units per day (#16 owns the estimate).
    Named assumptions (all of them are on the returned object):
    future sells are only modelled when passed in, the town's shop set
    only grows under `unlock_policy`, and nothing here models the
    `marketParams` override unless the caller passes one.
    """
    if unlock_policy not in UNLOCK_POLICIES:
        raise ValueError(f"unlock_policy must be one of {UNLOCK_POLICIES}, "
                         f"got {unlock_policy!r}")
    params = K.MARKET_PARAMS if market_params is None else market_params
    step = _step_of(obs)
    market = obs.get("market", {}) if isinstance(obs, dict) else {}
    raw_inv = dict(market.get("inventory", {}) or {})
    inv: dict[str, float] = {item: float(raw_inv.get(item, 0))
                             for item in PRODUCTS}
    shops = [str(s) for s in
             (obs.get("town", {}).get("unlocked_shops", ()) or ())]
    shop_interval = max(1, int(_get(config, "townShopSellInterval", 4)))
    center_interval = max(1, int(_get(config, "townCenterSellInterval", 24)))
    unlock_interval = max(1, int(_get(config, "townShopUnlockInterval", 3)))

    sells = our_sells or {}
    res = {item: float(n) for item, n in (residual or {}).items()}
    mean_demand = mean_shop_demand() if unlock_policy == "mean" else {}
    virtual_shops = 0.0                 # "mean" policy: fractional instances

    horizon = max(1, int(days))
    end = step + horizon * TURNS_PER_DAY
    first_day = step // TURNS_PER_DAY
    rows_inv: list[tuple[float, ...]] = []
    rows_price: list[tuple[int, ...]] = []

    def _snapshot() -> None:
        rows_inv.append(tuple(inv[item] for item in PRODUCTS))
        rows_price.append(tuple(int(K.market_price(item, inv[item], params))
                                for item in PRODUCTS))

    if step % TURNS_PER_DAY != 0:
        # mid-day: row 0 is TODAY as we see it, so the day index a caller
        # passes still means the day they are planning in
        _snapshot()
    for turn in range(step, end):
        if turn % TURNS_PER_DAY == 0:
            _snapshot()
            for item, units in res.items():        # the opponent's day
                _apply_sells(inv, item, units, params)
        for item, units in (sells.get(turn) or {}).items():
            _apply_sells(inv, item, units, params)
        for item, n in town_deltas(shops, turn, shop_interval,
                                   center_interval).items():
            inv[item] -= n
        if virtual_shops and turn % shop_interval == 0:
            for item, n in mean_demand.items():
                inv[item] -= n * virtual_shops
        if (turn + 1) % TURNS_PER_DAY == 0:        # end of day: one unlock?
            next_day = (turn + 1) // TURNS_PER_DAY
            if (unlock_policy == "mean" and next_day % unlock_interval == 0
                    and len(shops) + virtual_shops < MAX_SHOP_INSTANCES):
                virtual_shops += 1.0

    assumptions = [
        "town cadence: shops every %d turns, centre every %d turns "
        "(F037, engine `_town_consume`)" % (shop_interval, center_interval),
        "shop set: %d instance(s) observed; unlocks: %s"
        % (len(shops), unlock_policy),
        "our sells: %s" % ("included" if sells else "none passed in"),
        "opponent sells: %s" % ("residual %s units/day" % dict(res)
                                 if res else "none (named gap, #16)"),
    ]
    return MarketForecast(start_step=step, horizon_days=horizon,
                          first_day=first_day,
                          inventory=tuple(rows_inv), prices=tuple(rows_price),
                          unlock_policy=unlock_policy, residual=dict(res),
                          assumptions=tuple(assumptions))


def price_paths(fc: MarketForecast, days: int | None = None,
                items: Iterable[str] | None = None,
                from_day: int | None = None,
                ) -> dict[str, tuple[int, ...]]:
    """`{item: (price, ...)}` for the horizon — what feeds the master's `p_d`.

    F035 is the reason this exists: prices rise through the season, so the
    flat quote the master was started from under-prices every later day.
    The forecast replaces it with the town's own consumption walked forward
    through the engine's price function, so a tile's day-20 harvest is
    priced on the day-20 curve instead of today's.
    """
    horizon = fc.days if days is None else max(1, int(days))
    wanted = PRODUCTS if items is None else tuple(items)
    start = int(fc.first_day) if from_day is None else int(from_day)
    return {item: tuple(fc.price_of(item, start + d) for d in range(horizon))
            for item in wanted}
