"""Repair: a plan that is legal in the LP's arithmetic and illegal in the engine.

Issue #13 §2. The rounding hands back one plan per tile; those plans were priced
as arithmetic, not as orders, and **the engine fails silently** (F047). So before
a plan reaches the dispatcher it is walked order by order and every op that would
be refused is dropped — and **counted**, because a plan that quietly loses a
third of its ops looks exactly like one that works.

The checklist is F047's catalogue, each item with the rule that names it:

| check | rule |
|---|---|
| more than 10 market orders in a turn → truncate | F031 |
| market queue order: land, sells, hires, purchases | F032 |
| an order the purse cannot pay → drop | F031 (silent refusal) |
| FERTILIZE / FEED without the item in the acting unit's bag → drop | F004, F047 |
| any op on a tile the farm does not own → drop | F042 |
| SELL of an item the shed does not hold → drop | F043, F047 |
| one op per unit per turn (the dispatcher's own shape) | F030 |

Two things are deliberately *not* here:

- The **shed row** (F043's 100) is a cumulative coupling row, not an op check —
  it is `planner/columns.py::violations` over the `stored` row, checked against
  the assignment while it is still numbers.
- Routing and pickups. A unit works the tile it stands on and the plans carry no
  movement; filling that in is the day's (#14), so this module can only
  refuse what it can see, never invent a route.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from kaggle_environments.envs.kaggriculture import kaggriculture as K

from agent.dispatch import MAX_MARKET_ORDERS

# The market ops that need money, in the order the engine reads the queue.
_QUEUE_RANK = {"BUY_LAND": 0, "SELL": 1, "HIRE": 2}

# Items a worker op must already be carrying (F004: the FERTILIZE is refused
# when the acting unit holds none; FEED is the same shape).
_UNIT_INPUT = {"FERTILIZE": "FERTILIZER", "FEED": "WHEAT"}


@dataclass(frozen=True)
class Drop:
    """One op or order the repair refused, and the rule it would have broken."""

    where: str        # "market" or "unit <i>"
    op: tuple[str, ...]
    reason: str       # the finding/rule the drop enforces

    def describe(self) -> str:
        return f"{self.where} {list(self.op)} dropped: {self.reason}"


@dataclass(frozen=True)
class RepairResult:
    """The repaired plan and every drop that produced it."""

    plan: dict
    drops: tuple[Drop, ...] = field(default=())

    @property
    def dropped(self) -> int:
        return len(self.drops)

    def by_reason(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for drop in self.drops:
            counts[drop.reason] = counts.get(drop.reason, 0) + 1
        return counts

    def summary(self) -> str:
        if not self.drops:
            return "no drops"
        return ", ".join(f"{n}x {reason}" for reason, n in
                         sorted(self.by_reason().items()))


def _money(obs: Any) -> float:
    farms = obs.get("farms", []) if isinstance(obs, dict) else []
    player = int(obs.get("player", 0)) if isinstance(obs, dict) else 0
    if len(farms) > player:
        return float(farms[player].get("money", 0))
    return 0.0


def _shed(obs: Any) -> dict:
    private = obs.get("private", {}) if isinstance(obs, dict) else {}
    return dict(private.get("shed", {}))


def _inventories(obs: Any) -> list[dict]:
    private = obs.get("private", {}) if isinstance(obs, dict) else {}
    return [dict(inv) for inv in private.get("inventories", [])]


def _prices(obs: Any) -> dict:
    market = obs.get("market", {}) if isinstance(obs, dict) else {}
    return dict(market.get("prices", {}))


def _own_tiles(obs: Any) -> list[list[bool]]:
    """Which tiles the farm owns, from the LOCKED sentinel (F042)."""
    farms = obs.get("farms", []) if isinstance(obs, dict) else []
    player = int(obs.get("player", 0)) if isinstance(obs, dict) else 0
    if len(farms) <= player:
        return []
    tiles = farms[player].get("tiles", [])
    return [[tile != "LOCKED" for tile in row] for row in tiles]


def land_step_price(obs: Any, land_bought: int = 0) -> int | None:
    """The price of the next land purchase, or None once the prefix is complete.

    F042: quadrants open in a fixed prefix at escalating prices (`LAND_PRICES`),
    never a gap, and NW is owned from the start — so the first purchase costs
    `LAND_PRICES[0]` and each later one the next entry.

    `land_bought` is how many purchases the *current repair walk* has already
    simulated. Land has no equivalent of `hires_today` in the observation, so
    the caller carries it, and **without it every purchase inside one day is
    priced as the first one** — 3 × 1000 = 3000 against a real 1000+2000+4000 =
    7000, which walked a 7000-coin plan through a 3000-coin purse. That is B1 of
    the PR #35 review.
    """
    farms = obs.get("farms", []) if isinstance(obs, dict) else []
    player = int(obs.get("player", 0)) if isinstance(obs, dict) else 0
    owned = 0
    if len(farms) > player:
        owned = len(farms[player].get("unlocked_quadrants", []))
    index = max(0, owned) - 1 + int(land_bought)   # NW is owned from the start
    if 0 <= index < len(K.LAND_PRICES):
        return int(K.LAND_PRICES[index])
    return None


def order_cost(order: Iterable[str], obs: Any, hires_today: int,
               land_bought: int = 0) -> float | None:
    """What an order costs, from the engine's own tables (R002).

    `None` means "not a purchase this module prices" (a SELL, or a `BUY_LAND`
    once every quadrant is owned — which the repair drops on its own account,
    F042). Every number is imported: seed and animal prices from the engine
    tables, the market quote from the observation, the hire from the engine's
    own fib rule (F039) and land from the engine's prefix table (F042).
    """
    op = list(order)
    if not op:
        return None
    kind = op[0]
    if kind == "BUY_SEED" and len(op) >= 3:
        crop = op[1]
        if crop in K.CROPS:
            return float(K.CROPS[crop]["seed"]) * int(op[2])
    elif kind == "BUY_ANIMAL" and len(op) >= 3:
        species = op[1]
        if species in K.ANIMALS:
            return float(K.ANIMALS[species]["cost"]) * int(op[2])
    elif kind == "BUY_PRODUCT" and len(op) >= 3:
        price = _prices(obs).get(op[1])
        if price is not None:
            return float(price) * int(op[2])
    elif kind == "BUY_LAND":
        price = land_step_price(obs, land_bought)
        if price is not None:
            return float(price)
    elif kind == "HIRE":
        return float(K._hire_cost(int(hires_today)))
    return None


def _sort_market(market: list[list[str]]) -> list[list[str]]:
    """F032's queue order: land, then sells, then hires, then purchases."""
    def rank(order: list[str]) -> int:
        return _QUEUE_RANK.get(order[0] if order else "", 3)

    return sorted(market, key=rank)          # stable: same-rank keeps its order


def repair_day(plan: dict, obs: Any) -> RepairResult:
    """Walk one day's plan and drop every op the engine would refuse.

    Returns the repaired plan (`{"units": [...], "market": [...]}`), shape-valid
    by construction, plus the drops with the rule each one enforces.
    """
    if not isinstance(plan, dict) or "units" not in plan:
        raise ValueError("repair needs a dispatcher plan")
    drops: list[Drop] = []
    farms = obs.get("farms", []) if isinstance(obs, dict) else []
    player = int(obs.get("player", 0)) if isinstance(obs, dict) else 0
    farm = farms[player] if len(farms) > player else {}
    owned = _own_tiles(obs)
    shed = _shed(obs)
    inventories = _inventories(obs)
    positions = ([tuple(farm.get("farmer", (0, 0)))]
                 + [tuple(pos) for pos in farm.get("hands", [])])

    # ---- units: one op per turn, and never on a tile we do not own ----
    units: list[list[list[str]]] = []
    for index, unit in enumerate(plan["units"]):
        kept: list[list[str]] = []
        for hour, op in enumerate(unit):
            op_list = list(op)
            where = "farmer" if index == 0 else f"hand {index}"
            if op_list and op_list[0] in _UNIT_INPUT:
                need = _UNIT_INPUT[op_list[0]]
                carried = inventories[index].get(need, 0) if \
                    index < len(inventories) else 0
                if carried <= 0:
                    drops.append(Drop(where, tuple(op_list),
                                      f"F004/F047: the unit carries no {need}"))
                    kept.append(["PASS"])
                    continue
            if index < len(positions) and owned:
                x, y = positions[index]
                if y < len(owned) and x < len(owned[y]) and not owned[y][x]:
                    drops.append(Drop(where, tuple(op_list),
                                      "F042: the tile is LOCKED"))
                    kept.append(["PASS"])
                    continue
            kept.append(op_list)
        units.append(kept)

    # ---- market: F032's order, then F031's cap, then the purse ----
    market = [list(order) for order in plan.get("market", [])]
    market = _sort_market(market)
    purse = _money(obs)
    hires_today = int(farm.get("hires_today", 0))
    land_bought = 0                 # carried through the walk: land escalates (B1)
    kept_market: list[list[str]] = []
    for order in market:
        if len(kept_market) >= MAX_MARKET_ORDERS:
            drops.append(Drop("market", tuple(order),
                              "F031: an 11th order is dropped silently"))
            continue
        if order and order[0] == "BUY_LAND" and \
                land_step_price(obs, land_bought) is None:
            # F042: with every quadrant owned the buy refuses in silence, so
            # this is an op the engine would ignore — the repair's own job.
            drops.append(Drop("market", tuple(order),
                              "F042: every quadrant is already owned"))
            continue
        if order and order[0] == "SELL" and len(order) >= 3:
            held = shed.get(order[1], 0)
            if int(order[2]) > held:
                drops.append(Drop("market", tuple(order),
                                  f"F043/F047: the shed holds {held} "
                                  f"{order[1]}"))
                continue
            purse += float(_prices(obs).get(order[1], 0)) * int(order[2])
            kept_market.append(order)
            continue
        cost = order_cost(order, obs, hires_today, land_bought)
        if cost is not None:
            if cost > purse:
                drops.append(Drop("market", tuple(order),
                                  f"F031/F047: {cost:.0f} coins against a "
                                  f"{purse:.0f}-coin purse"))
                continue
            purse -= cost
            if order[0] == "HIRE":
                hires_today += 1
            elif order[0] == "BUY_LAND":
                land_bought += 1
        kept_market.append(order)

    return RepairResult(plan={"units": units, "market": kept_market},
                        drops=tuple(drops))
