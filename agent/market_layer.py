"""The market half of the day plan, riding on top of whatever rung answered (#15).

The fallback ladder (`agent/runtime.py`) decides the UNITS, and units are the
wrong owner for sells: `SELL` reads the shed (`_commit_unit`), and the shed is
filled by the nightly drop no matter which rung ran. So the market layer plans
once per day at hour 0 and attaches that turn's orders to whatever action dict
the ladder produced — greedy, the replanner's plan, or the all-PASS rung.

Modes, and why there are two (`CHISTA_MARKET`):

- `spread` — `secretary/inventory.py::market_queue`: the shed guard (F043,
  destruction is a bug and never a tuning choice), the cash rule (F038), the
  forecast peak rule, the season-end liquidation (F029), spread across the
  day's remaining turns because one big basket walks the price ladder down
  (F036) and because the cap is per TURN, not per day (F031).
- `dump` — the baseline `spread` has to beat: every sellable item, one order,
  at hour 0, the moment it is in the shed.

The layer never raises: a failure returns the rung's own action unchanged, so
the ladder's contract ("never raise, always a legal shape") holds with the
market layer on exactly as it does with it off. The rung's SELLs are dropped
while a mode is active — those orders and this schedule would double-count the
same stock, and the engine would refuse the surplus in silence (F047).
"""

from __future__ import annotations

import os
from typing import Any, Sequence

MODES = ("spread", "dump")
MAX_MARKET_ORDERS = 10                    # F031; the engine drops the 11th


def _sort_market(orders: list[list]) -> list[list]:
    """F032's queue order, reused from the repair layer (never reimplemented).

    `planner/repair.py::_sort_market` is the one implementation of
    land → sells → hires → purchases; importing it here keeps the two
    from drifting. It is private there, which is why this is a thin
    wrapper rather than a copy.
    """
    from planner.repair import _sort_market as sorter
    return sorter(orders)


class MarketLayer:
    """Per-day market plan; `attach()` is the only entry the ladder calls."""

    def __init__(self, mode: str = "spread", config: Any = None) -> None:
        if mode not in MODES:
            raise ValueError(f"unknown market mode {mode!r}: {MODES}")
        self.mode = mode
        self.config = config
        self.queue: list[list[list]] | None = None
        self.day = -1
        self.signature: tuple | None = None
        self.last: dict[str, Any] = {}

    # ---- planning (once per day, hour 0) ----
    def plan_day(self, obs: Any) -> list[list[list]]:
        """This day's per-turn order queue (`queue[hour] -> [order, ...]`)."""
        if self.mode == "dump":
            return self._dump_queue(obs)
        from secretary.inventory import market_queue
        queue = market_queue(obs, config=self.config, sort_market=_sort_market)
        self.last = {"mode": self.mode, "hours_with_orders":
                     sum(1 for row in queue if row)}
        return queue

    def _dump_queue(self, obs: Any) -> list[list[list]]:
        """Every sellable item, one order, hour 0 — the baseline."""
        private = obs.get("private", {}) if isinstance(obs, dict) else {}
        shed = {str(k): int(v) for k, v in (private.get("shed", {}) or {}).items()}
        orders = [["SELL", item, units] for item, units in sorted(shed.items())
                  if int(units) > 0]
        self.last = {"mode": "dump", "orders": len(orders)}
        return _rows_of_orders(orders, at_hour=0)

    # ---- per turn ----
    def attach(self, action: dict, obs: Any) -> dict:
        """Merge this turn's orders into `action`; never raises."""
        try:
            orders = _market_row(self, action, obs)
        except Exception as exc:                 # noqa: BLE001 - ladder first
            self.last = {"mode": self.mode,
                         "error": f"{type(exc).__name__}: {exc}"}
            return action
        # A rung that returned something that is not an action dict is the
        # layer's problem too: this method's contract is a shape-VALID dict,
        # so a missing rung action degrades to PASS plus this turn's orders.
        merged = dict(action) if isinstance(action, dict) \
            else {"farmer": ["PASS"], "hands": []}
        merged.setdefault("farmer", ["PASS"])
        merged.setdefault("hands", [])
        merged["market"] = orders
        return merged


def _rows_of_orders(orders: Sequence[list], at_hour: int = 0,
                    hours: int = 24) -> list[list[list]]:
    rows: list[list[list]] = [[] for _ in range(hours)]
    if orders:
        rows[min(max(0, int(at_hour)), hours - 1)] = [list(o) for o in orders]
    return rows


def from_env(config: Any = None) -> "MarketLayer | None":
    """`CHISTA_MARKET=spread|dump` -> a layer; anything else -> disabled."""
    mode = os.environ.get("CHISTA_MARKET", "")
    return MarketLayer(mode, config) if mode in MODES else None


def attach(layer: "MarketLayer | None", action: dict, obs: Any) -> dict:
    """Attach the market half to a rung's action (never raises).

    Called for every rung, so the sells ride on the greedy brain exactly as
    they ride on a committed replanner plan. On any failure the rung's own
    action is returned untouched — the ladder's contract outranks this
    layer.
    """
    if layer is None:
        return action
    return layer.attach(action, obs)


def _market_row(layer: "MarketLayer", action: dict, obs: Any) -> list[list]:
    """This turn's merged orders; raises if the row cannot fit F031's cap.

    The day is re-planned whenever the stock moved, not once per day: the
    sells issued at hour 1 change what the shed holds at hour 2, and the
    guard must see that. The plan itself reads the CURRENT observation, so
    a re-plan never double-counts an executed sale — it sees the smaller
    shed.
    """
    hour = int(obs.get("hour", 0)) if isinstance(obs, dict) else 0
    day = int(obs.get("day", 0)) if isinstance(obs, dict) else 0
    signature = (day, hour, _stock_signature(obs))
    if layer.queue is None or signature != layer.signature:
        layer.queue = layer.plan_day(obs)
        layer.day = day
        layer.signature = signature
    row = list(layer.queue[hour]) if 0 <= hour < len(layer.queue) else []
    given = (action.get("market") if isinstance(action, dict) else []) or []
    others = [list(o) for o in given if not (o and o[0] == "SELL")]
    orders = _sort_market(others + row)
    if len(orders) > MAX_MARKET_ORDERS:
        raise ValueError(
            f"hour {hour} queues {len(orders)} market orders: the engine "
            f"executes {MAX_MARKET_ORDERS} and drops the rest in silence "
            "(F031)")
    return orders


def _stock_signature(obs: Any) -> tuple:
    """(shed total, bag total) — what a plan's stock decision depends on."""
    private = obs.get("private", {}) if isinstance(obs, dict) else {}
    shed = sum(int(v) for v in (private.get("shed", {}) or {}).values())
    bags = sum(int(v) for bag in (private.get("inventories", []) or [])
               for v in (bag or {}).values())
    return (shed, bags)
