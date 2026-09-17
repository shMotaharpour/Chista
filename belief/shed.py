"""Secretary B: the shed — what lands, what fits, what has to be sold.

Issue #15. F043 is the hard wall: capacity is **100 items across all items
together**, `_end_of_day` pours every unit's bag into the shed and
**destroys** whatever does not fit, and `_commit_unit` refuses
`BUY_PRODUCT` / `BUY_ANIMAL` outright while `sum(shed) >= 100` — so a full
shed does not only destroy tonight's harvest, it blocks tomorrow's
fertiliser and feed. The acceptance is **zero destruction events**, so the
guard below is a constraint that outranks any price argument.

What this module owns, and what it deliberately does not:

- **owns** the shed projection (what the night drop will destroy), the
  sell decision (how much of which item to release on which turn of the
  day, and why), and the per-turn order queue the dispatcher consumes.
- **does not own** movement, PICKUP/DROP trips and hiring (#14), the
  opponent's sell pressure (#16), or the price curve itself — the latter
  is `market_price` in the engine, called through `day/market.py`.

Two engine facts the schedule is built around, both measured here rather
than assumed (`bench/bench_market_forecast.py --lag`):

1. **`SELL` reads the shed, never a unit's bag** (`_commit_unit`). So the
   stock this module may sell is `obs["private"]["shed"]`, not the bags;
   tonight's harvest is sellable tomorrow unless a unit spends a turn
   `DROP`ping it onto a shed-adjacent tile.
2. **Within a turn the engine runs unit actions, then the market.** A
   `DROP` issued at hour `h` therefore puts goods in the shed *before*
   hour `h`'s market queue runs, and a `SELL` in the same queue lands the
   same turn. Taking that option is #14's routing decision; this module
   records the choice in `Sale.reason` when it ever relies on it, and
   never assumes it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from belief.market import PRODUCTS, TURNS_PER_DAY

SHED_CAPACITY = 100          # engine default; the run's config can override it
MAX_ORDERS_PER_TURN = 10     # F031 - the engine drops the 11th silently
SEASON_DAYS = 30             # F029


def _get(config: Any, key: str, default: Any) -> Any:
    if config is None:
        return default
    try:
        return config.get(key, default)
    except AttributeError:
        return default


@dataclass(frozen=True)
class ShedState:
    """The shed, the bags, and what tonight's drop will do to them."""

    shed: Mapping[str, int]
    bags: tuple[Mapping[str, int], ...]
    capacity: int = SHED_CAPACITY

    @property
    def held(self) -> int:
        """Units in the shed — the only stock a SELL can reach (F043)."""
        return int(sum(self.shed.values()))

    @property
    def carried(self) -> int:
        """Units in unit bags; the nightly drop empties these into the shed."""
        return int(sum(sum(bag.values()) for bag in self.bags))

    @property
    def room(self) -> int:
        return max(0, self.capacity - self.held)

    @property
    def buys_blocked(self) -> bool:
        """F043's second half: at the cap, BUY_PRODUCT/BUY_ANIMAL are refused."""
        return self.held >= self.capacity

    def night_overflow(self, extra_carried: int = 0) -> int:
        """Units the night drop will DESTROY tonight.

        `extra_carried` is what the day's harvests, herds and pickups will
        still add to the bags before the drop (#14 publishes the real
        number; a caller without a plan passes its own estimate).
        """
        incoming = self.carried + max(0, int(extra_carried))
        return max(0, incoming - self.room)

    def sellable(self) -> dict[str, int]:
        """Shed contents, positive quantities only."""
        return {item: int(n) for item, n in self.shed.items() if int(n) > 0}


def shed_state(obs: Any, capacity: int | None = None) -> ShedState:
    """Read the shed and the bags out of our own observation."""
    private = obs.get("private", {}) if isinstance(obs, dict) else {}
    shed = {str(k): int(v) for k, v in (private.get("shed", {}) or {}).items()}
    bags = tuple({str(k): int(v) for k, v in (bag or {}).items()}
                 for bag in (private.get("inventories", []) or []))
    return ShedState(shed=shed, bags=bags,
                     capacity=int(capacity) if capacity else SHED_CAPACITY)


@dataclass(frozen=True)
class Sale:
    """One SELL, on one turn of the day, with the rule that produced it."""

    hour: int
    item: str
    units: int
    reason: str              # "season-end" | "shed-guard" | "cash" | "peak"

    @property
    def order(self) -> list:
        return ["SELL", self.item, int(self.units)]


def _guard_margin(capacity: int) -> int:
    """Slack left below the cap by the guard.

    The day's harvest is an ESTIMATE (`harvest_expected`; #14 publishes the
    real number) and the drop destroys whatever does not fit, so the guard
    keeps a small margin rather than filling the shed to the last slot.
    """
    return max(1, capacity // 20)          # 5 of 100


def plan_sales(stock: Mapping[str, int], forecast, *, day: int, hour: int = 0,
               harvest_expected: int = 0,
               money: float = 0.0, cash_needed: float = 0.0,
               end_day: int = SEASON_DAYS - 1,
               capacity: int = SHED_CAPACITY) -> tuple[Sale, ...]:
    """How much of which item to sell on which turn, and why.

    Order of the rules is the priority order, and the shed guard is first
    because F043 destroys product while the price forecast only prices it:

    1. `season-end` — on the last day everything is sold: the season ends
       with unsold stock (F029) and the bags are never dropped (the last
       turn is not an end-of-day turn), so holding on day 29 is a loss.
    2. `shed-guard` — sell enough that tonight's drop cannot overflow
       (`held + harvest_expected` against the room, with a margin).
    3. `cash` — sell enough to cover the day's planned outflow (F038:
       money binds in the first week; the hire ladder is F039). No caller
       supplies `cash_needed` yet — the plan that owns it is #14's — so the
       rule is INERT on today's path and the schedule is the guard, the
       peak rule and the season-end liquidation.
    4. `peak` — sell stock whose price is at its forecast maximum over the
       rest of the horizon, i.e. the price path stops rising from here.
       With prices rising through the season (F035) this is normally empty,
       and it is here so the rule exists rather than being assumed away.
       The scan covers the FORECAST's own horizon
       (`first_day .. first_day + days - 1`), which is not season day 0.

    Stock is the SHED's contents (F043: a SELL cannot reach a bag). The
    returned sales are spread across the day's remaining turns, and the two
    measurements behind that choice (probe, 2026-09-16) say what it is
    worth:

    - grouping orders WITHIN a turn is worth nothing — the engine quotes
      unit by unit, so 50 wheat as one order and as five orders of 10 both
      fetched exactly 1,131 coins (reproduced 2026-09-16: a 50-wheat shed,
      one `SELL 50` against five `SELL 10` on the same season state);
    - spreading over TURNS pays when the basket is large next to the town's
      drain: the day-29 liquidation spread across the day beat holding it
      all to the last turn by ~92 coins a season (mean 3,927 vs 3,835 over
      12 seeds; reproduced 2026-09-16: the spread arm against an arm whose
      day-29 sales all land at hour 23, seeds 0..11, `weedSpawnChance`
      0.005, means 3,926.5 vs 3,834.2).

    So the spread is not a price trick on small baskets; it is what keeps a
    large forced sale from landing as one basket at the seasonal peak.
    """
    held = {item: int(n) for item, n in stock.items() if int(n) > 0}
    if not held:
        return ()
    total = sum(held.values())
    hours = [h for h in range(max(0, int(hour)), TURNS_PER_DAY)]
    if not hours:
        return ()

    take: dict[str, int] = {}
    reasons: dict[str, str] = {}

    def _release(item: str, units: int, reason: str) -> None:
        units = min(int(units), held[item] - take.get(item, 0))
        if units <= 0:
            return
        take[item] = take.get(item, 0) + units
        # keep the FIRST reason that claimed an item: the priority order
        reasons.setdefault(item, reason)

    def _order_of_preference() -> list[str]:
        """Largest holding first: it is what fills the shed, and its
        quantity is the one the ladder punishes."""
        return sorted(held, key=lambda i: (-held[i], i))

    if int(day) >= int(end_day):
        for item in _order_of_preference():
            _release(item, held[item], "season-end")
    else:
        forcing = (total + max(0, int(harvest_expected))
                   - max(1, int(capacity)) + _guard_margin(int(capacity)))
        if forcing > 0:
            need = forcing
            for item in _order_of_preference():
                if need <= 0:
                    break
                units = min(need, held[item] - take.get(item, 0))
                _release(item, units, "shed-guard")
                need -= units
        need_cash = max(0.0, float(cash_needed) - float(money))
        if need_cash > 0:
            for item in _order_of_preference():
                if need_cash <= 0:
                    break
                units = held[item] - take.get(item, 0)
                if units <= 0:
                    continue
                price = max(1, forecast.price_of(item, int(day)))
                affordable = int(min(units, -(-need_cash // price)))
                _release(item, affordable, "cash")
                need_cash -= affordable * price
        for item in _order_of_preference():
            if reasons.get(item):           # already claimed by a rule above
                continue
            price_today = forecast.price_of(item, int(day))
            # The forecast's rows are indexed from ITS OWN start
            # (`first_day`), not from season day 0: scanning
            # `range(day, forecast.days)` mixes the two spaces, and the
            # range collapses to empty once `day >= days` (day 15 of a
            # 30-day season with `days = SEASON_DAYS - day`), which made the
            # comparison read today's price against itself and fire on a
            # rising path. Scan the forecast's own horizon instead.
            first = int(getattr(forecast, "first_day", int(day)))
            horizon = max((forecast.price_of(item, first + offset)
                           for offset in range(max(1, int(forecast.days)))),
                          default=price_today)
            if price_today >= horizon:
                _release(item, held[item], "peak")

    sales: list[Sale] = []
    for item in sorted(take):
        units = take[item]
        if units <= 0:
            continue
        chunks = _spread(units, len(hours))
        for offset, chunk in enumerate(chunks):
            sales.append(Sale(hour=hours[offset], item=item, units=chunk,
                              reason=reasons[item]))
    return tuple(sorted(sales, key=lambda s: (s.hour, s.item)))


def _spread(units: int, turns: int) -> list[int]:
    """`units` split over `turns` orders, as evenly as integers allow."""
    if turns <= 0:
        return [units]
    base, extra = divmod(int(units), turns)
    out = [base + 1] * extra + [base] * (turns - extra)
    return [n for n in out if n > 0]


def orders_by_hour(sales: Iterable[Sale],
                   hours: int = TURNS_PER_DAY) -> list[list[list]]:
    """Sales -> the day's per-turn market queue, the dispatcher's shape.

    The queue is `queue[hour] -> [order, ...]`, so the ≤ 10-order cap is a
    PER-TURN cap (F031): a 24-turn day has 240 slots and no order needs to
    be lost.
    """
    queue: list[list[list]] = [[] for _ in range(hours)]
    for sale in sales:
        if 0 <= sale.hour < hours:
            queue[sale.hour].append(sale.order)
    return queue


def _assert_within_cap(queue: list[list[list]]) -> None:
    """F031 guard: one order per item per turn can never reach the cap.

    At most `len(PRODUCTS)` SELLs exist per turn (one per item), so the
    engine's 10-order limit is unreachable by construction — and if a
    caller merges hires and purchases into a row, this raises instead of
    letting the engine drop the 11th in silence.
    """
    for hour, row in enumerate(queue):
        if len(row) > MAX_ORDERS_PER_TURN:
            raise ValueError(
                f"hour {hour} queues {len(row)} market orders; the engine "
                f"executes {MAX_ORDERS_PER_TURN} and drops the rest in "
                "silence (F031)")


def market_queue(obs: Any, forecast_obj=None, *, harvest_expected: int = 0,
                 cash_needed: float = 0.0, config: Any = None,
                 sort_market=None) -> list[list[list]]:
    """The market half of one day's plan, from the observation alone.

    Supply side stand-in: the shed and the bags as the observation shows
    them. #14 publishes a plan of what the units will harvest today
    (`harvest_expected`); until that is wired in, the caller passes its own
    estimate and this module says so on the result's assumptions.
    """
    from belief.market import forecast as _forecast
    capacity = int(_get(config, "shedCapacity", SHED_CAPACITY))
    state = shed_state(obs, capacity=capacity)
    day = int(obs.get("day", 0)) if isinstance(obs, dict) else 0
    hour = int(obs.get("hour", 0)) if isinstance(obs, dict) else 0
    fc = forecast_obj or _forecast(obs, days=SEASON_DAYS - day, config=config)
    # The night drop empties EVERY unit's bag into the shed wherever that
    # unit stands (`_drop_inventories_to_shed`), so the guard's incoming is
    # what the bags already hold plus whatever the day still harvests.
    incoming = state.carried + max(0, int(harvest_expected))
    sales = plan_sales(state.sellable(), fc, day=day, hour=hour,
                       harvest_expected=incoming,
                       money=_money(obs), cash_needed=cash_needed,
                       capacity=capacity)
    queue = orders_by_hour(sales)
    if sort_market is not None:            # F032: land, sells, hires, buys
        queue = [sort_market(row) if row else row for row in queue]
    _assert_within_cap(queue)
    return queue


def _money(obs: Any) -> float:
    farms = obs.get("farms", []) if isinstance(obs, dict) else []
    player = int(obs.get("player", 0)) if isinstance(obs, dict) else 0
    farm = farms[player] if len(farms) > player else {}
    return float(farm.get("money", 0.0))
