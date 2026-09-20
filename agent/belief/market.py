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

## How an agent uses this module (the 60-second read)

You are a layer that needs to know what a sale is worth. Three calls:

1. `forecast(obs, days=N)` — the market walk: `fc.prices[d]` is the (9,)
   quote at the START of season day `first_day + d`; `fc.inventory_of(g, d)`
   is the inventory behind it; `fc.assumptions` names every assumption the
   walk made (read it before trusting the numbers).
2. `hourly_prices(fc)` — the SAME walk sampled per hour: row `(d, h)` is
   the quote a SELL at hour h of day `first_day + d` actually sees (after
   that turn's market, before its town consumption; past hours of the
   current day repeat the snapshot). This is what a slot-level decision —
   which hour to sell in, against the rival's queue — reads.
3. `hourly_prices` + `agent.belief.ladder.sell_coins` — the ladder over the
   hourly path: the coins of selling n units AT hour h (the walk already
   carries the price impact of your own earlier units, passed via
   `our_sells={step: {item: qty}}`).

The rival's future sells are NOT modelled by default (`residual=None`):
the price path is then biased toward HIGHER prices, on purpose and named.
Pass `residual={"WHEAT": 2.0}` (units/day) when you have an estimate —
`agent.belief.opponent.OpponentModel` (corpus-primed) produces one.

Worked example (engine-verified): day-1 observation, PASS policies, a rival
selling 10 WHEAT per day — `hourly_prices` day 0 reads
`26 26 26 26 26 25 25 24 24 ... 23 23 ...`: each 4-hour shop tick plus the
rival's drip pushes the quote one rung down, and day 1 holds 23. Sell at
hour 4, not hour 23 — the 3 coins a unit difference is the ladder.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from kaggle_environments.envs.kaggriculture import kaggriculture as K

from agent.belief.schemas import SHOP_BASKET

# The vocabulary and the tables all come from the engine (R002): a shop
# table or a product list edited there must move this module with it.
PRODUCTS: tuple[str, ...] = tuple(K.PRODUCTS)
SHOPS: dict[str, tuple[str, ...]] = {name: tuple(items)
                                     for name, items in K.SHOPS.items()}
TOWN_CENTER_PRODUCTS: tuple[str, ...] = tuple(K.TOWN_CENTER_PRODUCTS)
MAX_SHOP_INSTANCES: int = int(K.MAX_SHOP_INSTANCES)
PRICE_FLOOR: int = int(K.PRICE_FLOOR)

TURNS_PER_DAY = 24          # engine default (turnsPerDay); F029/F048 pin 30 days
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
    #: the per-turn walk behind the day rows — what `hourly_prices` samples.
    #: (T+1, 9) inventories; row 0 is the snapshot, row j after turn j-1.
    walk_inventory: np.ndarray = field(default_factory=lambda: np.zeros((1, 1)))
    walk_step: int = 0
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

    #: The tests' name for the same read (PR #61's test_market_layer calls
    #: `.price(item, day)`); one forecast, two spellings, no second table.
    price = price_of

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

    Vectorized form: the day's inventory walk is ONE (days+1, 9) array built
    from cumulative drains (the cadence is deterministic, F037), the sells
    enter as cumulative ladders (`belief/ladder.py`), and the prices are the
    quote table indexed at the walked rows — no per-turn Python loop.

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
    # The one snapshot (ARCHITECTURE §5 step 5): the forecast is a view of
    # `MarketState`, so the market has a single reader.
    from agent.belief.schemas import MarketState
    state = MarketState.from_obs(obs)
    inv0 = np.asarray(state.inventory, dtype=np.float64)
    shops = [str(s) for s in
             (obs.get("town", {}).get("unlocked_shops", ()) or ())]
    shop_interval = max(1, int(_get(config, "townShopSellInterval", 4)))
    center_interval = max(1, int(_get(config, "townCenterSellInterval", 24)))
    unlock_interval = max(1, int(_get(config, "townShopUnlockInterval", 3)))

    sells = {int(k): dict(v) for k, v in (our_sells or {}).items()}
    res = {item: float(n) for item, n in (residual or {}).items()}
    mean_demand = mean_shop_demand() if unlock_policy == "mean" else {}
    virtual_shops = 0.0                 # "mean" policy: fractional instances

    horizon = max(1, int(days))
    end = step + horizon * TURNS_PER_DAY
    first_day = step // TURNS_PER_DAY
    n = _PROD_INDEX

    # --- the (horizon, 9) drain matrix, one vectorized pass ----------------- #
    turns = np.arange(step, end)
    shop_ticks = (turns % shop_interval == 0).astype(np.float64)
    center_ticks = (turns % center_interval == 0).astype(np.float64)
    per_shop = np.zeros(len(PRODUCTS))
    for name in shops:
        items, mult = SHOP_BASKET[name]
        for it in items:
            per_shop[n[it]] += mult
    per_center = np.zeros(len(PRODUCTS))
    for it in TOWN_CENTER_PRODUCTS:
        per_center[n[it]] += 1.0
    mean_row = np.array([mean_demand.get(it, 0.0) for it in PRODUCTS])
    # virtual shops only join AFTER their unlock day's end-of-day refresh;
    # the count is per DAY but the demand applies PER TURN
    unlock_day = np.zeros(horizon, dtype=np.float64)     # virtual count by day
    if unlock_policy == "mean":
        v = 0.0
        for d in range(horizon):
            day_end = (first_day + d + 1)
            if (day_end % unlock_interval == 0
                    and len(shops) + v < MAX_SHOP_INSTANCES):
                v += 1.0
            unlock_day[d] = v
    day_of_turn = np.minimum((turns // TURNS_PER_DAY - first_day), horizon - 1
                             ).astype(np.int64)
    virtual_by_day = np.concatenate([[0.0], unlock_day[:-1]])   # demand starts the day after
    virtual_per_turn = virtual_by_day[day_of_turn]              # (T,)
    drains = (shop_ticks[:, None] * per_shop[None, :]
              + center_ticks[:, None] * per_center[None, :]
              + virtual_per_turn[:, None] * mean_row[None, :])   # (T, 9), per TURN

    # --- our sells + the residual, as per-turn unit counts per good --------- #
    our = np.zeros((len(turns), len(PRODUCTS)))
    for abs_step, basket in sells.items():
        t = int(abs_step) - step
        if 0 <= t < len(turns):
            for it, units in basket.items():
                if it in n:
                    our[t, n[it]] += float(units)
    res_vec = np.array([res.get(it, 0.0) for it in PRODUCTS])
    rival = np.repeat(res_vec[None, :] / TURNS_PER_DAY, len(turns), axis=0)

    # --- inventory walk: cumsum of (rival + our - drain), per TURN ----------
    # both seats' sales ADD supply (+1 per unit, engine `_commit_unit`); the
    # town's consumption removes.
    net = rival + our - drains
    walk = inv0[None, :] + np.concatenate(
        [np.zeros((1, len(PRODUCTS))), np.cumsum(net, axis=0)], axis=0)

    # the forecast's rows are DAY STARTS: row d = the walk after turn 24d - 1
    # (the engine's day-start observation), row 0 = the snapshot itself.
    # A mid-day forecast (step % 24 != 0) spends the first partial day first:
    # its row 0 is the snapshot, row 1 is the NEXT day start, so the walk
    # keeps horizon * 24 + (24 - hour) turns and the sampling skips the stub.
    stub = TURNS_PER_DAY - (step % TURNS_PER_DAY)
    if stub == TURNS_PER_DAY:
        stub = 0
    else:
        pass                                    # the stub turns run first
    day_rows = [0]
    if stub:
        day_rows.append(stub)
    day_rows.extend(range(stub + TURNS_PER_DAY, len(turns), TURNS_PER_DAY))
    day_rows = day_rows[:horizon]           # rows are the horizon's day STARTS
    rows_inv = []
    rows_price = []
    for d in range(horizon):
        row = walk[day_rows[d]]
        rows_inv.append(tuple(float(x) for x in row))
        # the quote table IS the engine function (`world/prices`, parity-tested);
        # the per-row call below keeps the engine in the loop so a patched
        # `K.market_price` moves the prices with it (the parity guard watches).
        rows_price.append(tuple(
            int(K.market_price(PRODUCTS[i], float(row[i]),
                               params if market_params is not None else None))
            for i in range(len(PRODUCTS))))

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
                          walk_inventory=walk, walk_step=step,
                          unlock_policy=unlock_policy, residual=dict(res),
                          assumptions=tuple(assumptions))


def hourly_prices(fc: MarketForecast, days: int | None = None,
                  items: Iterable[str] | None = None,
                  ) -> np.ndarray:
    """The market's value PER HOUR: (days*24, 9) quotes in PRODUCTS order.

    The day-start table is what a day-granular plan prices on; a slot-level
    decision (which hour of the day to sell in, against the rival's queue)
    needs the same walk sampled every turn. The walk itself is already
    per-turn — this is a sampling choice, not a second simulation.

    Row `(d, h)` = day `first_day + d`, hour `h`, quoted at the inventory
    the walk holds after that turn's market and before that turn's town
    consumption — the inventory a SELL in that turn is actually quoted at
    (engine turn order: units, market, town). Row 0 is the observation's
    own snapshot; a mid-day forecast starts with its stub and the remaining
    hours of that day follow, so the table's length stays days*24.
    """
    walk = np.asarray(fc.walk_inventory, dtype=np.float64)
    step = int(fc.walk_step)
    horizon = fc.horizon_days if days is None else max(1, int(days))
    wanted = (PRODUCTS if items is None else tuple(items))
    ix = [_PROD_INDEX[g] for g in wanted]

    stub = TURNS_PER_DAY - (step % TURNS_PER_DAY)
    if stub == TURNS_PER_DAY:
        stub = 0
    # Row indexing. The walk's row j is the inventory after turn (step+j-1);
    # row 0 is the snapshot ("now", mid-day). Hour h of the CURRENT day:
    #   - h < hour  : already played — the table shows the snapshot (the walk
    #                 has no earlier rows; history is not re-quoted);
    #   - h >= hour : the quote a SELL at that hour sees = the walk row after
    #                 the turns up to it = row (h - hour) — hour `hour`'s own
    #                 quote is the snapshot (its market has not run yet).
    # Later days start at walk row (stub + 24d) + (h) as before.
    hour_now = step % TURNS_PER_DAY
    rows = [0] * TURNS_PER_DAY                       # day 0, past hours
    for h in range(hour_now, TURNS_PER_DAY):
        rows[h] = h - hour_now                       # 0 = the snapshot itself
    # day d >= 1: its hour-0 quote is the walk row after the whole previous
    # day = row (stub + 24d - hour_now)... in walk terms the turn at absolute
    # step (first_day + d)*24 - 1 sits at row (first_day + d)*24 - step:
    for d in range(1, horizon):
        base = (fc.first_day + d) * TURNS_PER_DAY - step
        rows.extend([base + h for h in range(TURNS_PER_DAY)])
    rows = rows[:horizon * TURNS_PER_DAY]
    out = np.zeros((horizon * TURNS_PER_DAY, len(wanted)), dtype=np.int64)
    for i, r in enumerate(rows):
        inv = walk[min(r, walk.shape[0] - 1)]
        out[i] = [K.market_price(PRODUCTS[i2], float(inv[i2])) for i2 in ix]
    return out


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
