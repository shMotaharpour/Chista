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

The price itself is never modelled here: `agent.world.prices` owns the engine's
formula (one transcription, held to `market_price` cell by cell by
`test_price_parity`), and this module reads it through `price_grid` — the same
numbers, priced for a whole (rows x goods) surface in one pass.

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
from agent.world.prices import price_grid
from agent.world.rules import (CENTER_SELL_INTERVAL_TURNS,
                               SHOP_SELL_INTERVAL_TURNS,
                               SHOP_UNLOCK_INTERVAL_DAYS, TURNS_PER_DAY)

# The vocabulary and the tables all come from the engine (R002): a shop
# table or a product list edited there must move this module with it.
PRODUCTS: tuple[str, ...] = tuple(K.PRODUCTS)
SHOPS: dict[str, tuple[str, ...]] = {name: tuple(items)
                                     for name, items in K.SHOPS.items()}
TOWN_CENTER_PRODUCTS: tuple[str, ...] = tuple(K.TOWN_CENTER_PRODUCTS)
MAX_SHOP_INSTANCES: int = int(K.MAX_SHOP_INSTANCES)
PRICE_FLOOR: int = int(K.PRICE_FLOOR)

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
    #: The same walk priced from the OTHER end of the rival's estimate. When a
    #: caller hands in a ceiling (their shed-level stock plus the harvest they
    #: have not dropped), the rival term becomes that ceiling, so more supply
    #: presses the price down: this is the pessimistic band of the same model,
    #: not a second model. Left at zero means no ceiling was given and
    #: `walk_inventory` is the only path -- the bit-identical case.
    walk_inventory_high: np.ndarray = field(
        default_factory=lambda: np.zeros((1, 1)))
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
             rival_supply: np.ndarray | None = None,
             rival_ceiling: np.ndarray | None = None,
             rival_sells: Mapping[int, Mapping[str, int]] | None = None,
             unlock_policy: str = "mean",
             config: Any = None,
             market_params: Any = None,
             ) -> MarketForecast:
    """Forward-simulate market inventory, then price each day start.

    Vectorized form: the day's inventory walk is ONE (days+1, 9) array built
    from cumulative drains (the cadence is deterministic, F037), the sells
    enter as cumulative ladders (`belief/ladder.py`), and the prices are the
    quote table indexed at the walked rows — no per-turn Python loop.

    `our_sells` and `rival_sells` are both `{absolute_step: {item: units}}`.
    `rival_sells` is the DATED form of the rival's pressure — the hours their
    supply actually lands in — and it REPLACES `rival_supply`'s day-level total
    on every day it names, because a caller that knows the hours and also passes
    the day's total would count the same units twice. A day it does not name
    keeps the calendar's own number. `residual` is the
    opponent's sell pressure in units per day (#16 owns the estimate), and
    `rival_supply` is that same pressure DATED — `(days, 9)` units per day per
    good, as `belief/rival_calendar.supply_curve` builds it from their public
    board. The dated form is what makes the timing real; `residual` remains the
    input for a caller that has no calendar. Named assumptions (all of them are
    on the returned object):
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
    # The engine's own defaults (kaggriculture.py:733-734, :867), and the run may
    # override each one: the key names are the engine's, `terms` is what resolves
    # them, and the defaults here are the world's transcription — not a second
    # copy of the number.
    shop_interval = max(1, int(_get(config, "townShopSellInterval",
                                    SHOP_SELL_INTERVAL_TURNS)))
    center_interval = max(1, int(_get(config, "townCenterSellInterval",
                                      CENTER_SELL_INTERVAL_TURNS)))
    unlock_interval = max(1, int(_get(config, "townShopUnlockInterval",
                                      SHOP_UNLOCK_INTERVAL_DAYS)))

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
    if rival_supply is None:
        # The flat estimate: the opponent's pressure spread evenly over the
        # horizon. Kept as the caller's own input (the same path, a different
        # vector) because `residual` is what a caller without a calendar has.
        rival = np.repeat(res_vec[None, :] / TURNS_PER_DAY, len(turns), axis=0)
    else:
        # The dated curve (`belief/rival_calendar.supply_curve`): units per DAY
        # per good. A rival tile that pays out on day 7 pushes the price down on
        # day 7, not on every day — which is the whole difference between a
        # residual and a calendar, and the half of the board a path that assumes
        # no rival supply ignores.
        curve = np.asarray(rival_supply, dtype=np.float64)
        if curve.shape != (horizon, len(PRODUCTS)):
            raise ValueError(
                f"rival_supply: expected shape ({horizon}, {len(PRODUCTS)}), "
                f"got {curve.shape}")
        rival = curve[day_of_turn] / TURNS_PER_DAY
    if rival_sells:
        # The rival's supply at the HOUR it lands, not spread over the day. The
        # days it names lose the calendar's own total for that day: the dated
        # units are what the tracker and the opponent model know, and adding both
        # would push the same supply into the walk twice.
        touched = set()
        for abs_step, basket in rival_sells.items():
            t = int(abs_step) - step
            if 0 <= t < len(turns):
                touched.add(int(abs_step) // TURNS_PER_DAY - first_day)
                for it, units in basket.items():
                    if it in n:
                        rival[t, n[it]] += float(units)
        for rel in touched:
            if 0 <= rel < horizon:
                rival[day_of_turn == rel, :] = 0.0
                for abs_step, basket in rival_sells.items():
                    t = int(abs_step) - step
                    if (0 <= t < len(turns)
                            and int(abs_step) // TURNS_PER_DAY - first_day == rel):
                        for it, units in basket.items():
                            if it in n:
                                rival[t, n[it]] += float(units)

    # --- inventory walk: cumsum of (rival + our - drain), per TURN ----------
    # both seats' sales ADD supply (+1 per unit, engine `_commit_unit`); the
    # town's consumption removes.
    net = rival + our - drains
    walk = inv0[None, :] + np.concatenate(
        [np.zeros((1, len(PRODUCTS))), np.cumsum(net, axis=0)], axis=0)

    # The pessimistic band is the SAME arithmetic with the rival at the other
    # end of their own estimate: their ceil replaces their floor and nothing
    # else moves. One cumsum, no loop, and no work at all when the caller has
    # no ceiling to give.
    walk_high = walk
    if rival_ceiling is not None:
        high = np.asarray(rival_ceiling, dtype=np.float64)
        if high.shape != (horizon, len(PRODUCTS)):
            raise ValueError(
                f"rival_ceiling: expected shape ({horizon}, {len(PRODUCTS)}), "
                f"got {high.shape}")
        rival_high = high[day_of_turn] / TURNS_PER_DAY
        net_high = rival_high + our - drains
        walk_high = inv0[None, :] + np.concatenate(
            [np.zeros((1, len(PRODUCTS))), np.cumsum(net_high, axis=0)], axis=0)

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
    rows_inv = [tuple(float(x) for x in walk[day_rows[d]]) for d in range(horizon)]
    if market_params is None:
        # the quote table IS the engine function (`world/prices`, held to it by
        # `test_price_parity`), read for the whole surface at once — the rows the
        # hourly layer samples come from the same call.
        quotes = price_grid(walk[np.asarray(day_rows[:horizon], dtype=np.int64)])
        rows_price = [tuple(int(x) for x in quotes[d]) for d in range(horizon)]
    else:
        # a caller's own params: the engine prices those, one row at a time
        rows_price = [
            tuple(int(K.market_price(PRODUCTS[i], float(walk[day_rows[d]][i]),
                                     market_params))
                  for i in range(len(PRODUCTS)))
            for d in range(horizon)]

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
                          walk_inventory=walk, walk_inventory_high=walk_high, walk_step=step,
                          unlock_policy=unlock_policy, residual=dict(res),
                          assumptions=tuple(assumptions))


def _walk_rows(fc: MarketForecast) -> np.ndarray:
    """The forecast's per-turn inventory walk, `(T+1, 9)`, row 0 = the snapshot."""
    return np.asarray(fc.walk_inventory, dtype=np.float64)


def _hourly_rows(fc: MarketForecast, horizon: int) -> np.ndarray:
    """The walk row each (day, hour) of the hourly tables reads.

    ONE definition, because `hourly_prices` and `hourly_inventory` must sample
    the same row: a plan that prices a sale at one inventory and is filled at
    another is pricing a market that does not exist.

    Row indexing. The walk's row j is the inventory after turn (step+j-1);
    row 0 is the snapshot ("now", mid-day). Hour h of the CURRENT day:
      - h < hour  : already played — the table shows the snapshot (the walk
                    has no earlier rows; history is not re-quoted);
      - h >= hour : the quote a SELL at that hour sees = the walk row after
                    the turns up to it = row (h - hour) — hour `hour`'s own
                    quote is the snapshot (its market has not run yet).
    Later days start at walk row (24d) + h.

    The table is built as ONE arithmetic expression rather than a per-row walk
    (the depth surface calls this per (good, day, hour) cell and the loops cost
    far more than the values): day 0 is `max(0, h - hour_now)`, a later day is
    `(first_day + d) * 24 - step + h`, both clamped to the walk's last row.
    Both are the same number the per-row form writes; `_walk_row_of` (depth)
    resolves the identical expression for a single cell.
    """
    hours = np.arange(TURNS_PER_DAY)
    hour_now = int(fc.walk_step) % TURNS_PER_DAY
    parts = [np.maximum(hours - hour_now, 0)]
    if horizon > 1:
        base = ((int(fc.first_day) + np.arange(1, horizon)) * TURNS_PER_DAY
                - int(fc.walk_step))
        parts.append((base[:, None] + hours[None, :]).ravel())
    rows = np.concatenate(parts)[:horizon * TURNS_PER_DAY]
    return np.minimum(rows, len(fc.walk_inventory) - 1)


def hourly_inventory(fc: MarketForecast, days: int | None = None,
                     items: Iterable[str] | None = None,
                     ) -> np.ndarray:
    """The market's INVENTORY per hour: (days*24, 9), the rows the prices read.

    `hourly_prices` samples the walk for the QUOTE; a depth read (what a lot
    fetches) needs the inventory behind that quote, at the same turn — the
    ladder is a function of it. One walk, two readings, the same rows.
    """
    horizon = fc.horizon_days if days is None else max(1, int(days))
    wanted = (PRODUCTS if items is None else tuple(items))
    ix = [_PROD_INDEX[g] for g in wanted]
    walk = _walk_rows(fc)
    rows = _hourly_rows(fc, horizon)
    # one fancy index over the walk: the (T, 9) rows the table reads, the
    # requested columns taken from them. `int()` on a float truncates toward
    # zero, which is what the cast below does too.
    return np.asarray(walk[np.asarray(rows)][:, ix], dtype=np.int64)


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

    One pass: the rows are the walk's own indices (`_hourly_rows`), priced for
    every good at once by `price_grid` — the engine's formula, held to
    `market_price` cell by cell by `tests/test_market_analyzer.py::test_price_parity`
    on whole AND fractional inventories.
    """
    horizon = fc.horizon_days if days is None else max(1, int(days))
    wanted = (PRODUCTS if items is None else tuple(items))
    ix = [_PROD_INDEX[g] for g in wanted]
    walk = _walk_rows(fc)
    rows = _hourly_rows(fc, horizon)
    quotes = price_grid(walk[np.asarray(rows)])
    return np.asarray(quotes[:, ix], dtype=np.int64)


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
