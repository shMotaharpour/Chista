"""The day's market queue: what it buys, who it hires, and what it sells.

Three sources, one queue, and the engine's own settle order between them
(F032: land, sells, hires, purchases). The cap is per TURN and not per day
(F031: ten), and an order past it is not refused out loud — the engine simply
does nothing with it (F047), which is the failure that looks like success.

The buys are derived from the columns the master committed, not from a
separate reading of the plan: a seed counted twice is a plan that cannot pay
for itself, and a seed counted nowhere is a PLANT the engine silently drops.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from agent.world.rules import (ANIMAL_RULES, CROP_RULES, TURNS_PER_DAY,
                               hire_cost)

#: The engine settles a turn's market in this order (F032). A day that must
#: sell before it can afford its seeds has to be queued in it, or the purchase
#: is refused for want of coins it is about to have.
QUEUE_RANK = {"BUY_LAND": 0, "SELL": 1, "HIRE": 2}
DEFAULT_RANK = 3


@dataclass(frozen=True)
class DayMarket:
    """The queue, and the bill it runs up."""

    rows: list                     # per turn: [order, ...]
    bill: int                      # coins the buys and hires cost
    buys: tuple = ()               # the purchase orders, for the record
    hires: int = 0                 # hands hired
    dropped: tuple = ()            # orders no turn had room for
    sells: int = 0                 # SELL orders the queue carries


def needs(chains) -> dict[str, int]:
    """What the day's chains consume that the farm has to own first.

    Keyed by `(good, kind)` and not by the good alone, because WHEAT is both a
    crop and a product and they are not the same purchase: a PLANT of wheat
    needs `BUY_SEED WHEAT` at 10, a FEED needs `BUY_PRODUCT WHEAT` at the
    market's quote, and buying one where the other was meant is a pickup of
    something that is not in the shed — refused without a word (F047).

    A PLANT needs its seed and a PLACE needs its animal (F001, F016), and both
    are the chain's ENTITY — 'PLANT' alone plants nothing in particular, so
    the ops are read together with it.

    A FEED and a FERTILIZE consume too, and neither is an entity: they eat the
    shared wheat and fertilizer stocks. Counting only the entities left the
    day picking up a wheat it had never bought, which the engine refuses
    without a word (F047) while the plan reports success.
    """
    out: dict[tuple[str, str], int] = {}
    for _cell, ops, entity in chains:
        if entity is not None:
            if "PLANT" in ops and entity in CROP_RULES:
                out[(entity, "SEED")] = out.get((entity, "SEED"), 0) + 1
            elif "PLACE" in ops and entity in ANIMAL_RULES:
                out[(entity, "ANIMAL")] = out.get((entity, "ANIMAL"), 0) + 1
        for op in ops:
            good = OP_CONSUMES.get(op)
            if good is not None:
                out[(good, "PRODUCT")] = out.get((good, "PRODUCT"), 0) + 1
    return out


#: What an op takes out of the shed, beyond the entity it constructs. FEED
#: eats the shared wheat stock (F017/F018) and FERTILIZE the shared fertilizer
#: (F006/F023); neither is the chain's entity, so neither is visible from the
#: entity alone.
OP_CONSUMES: dict[str, str] = {"FEED": "WHEAT", "FERTILIZE": "FERTILIZER"}


def availability(chains, seeds: dict, shed: dict) -> dict[str, int]:
    """The hour each good the day needs is in the shed — wsr's `available`.

    What is on the shelf already is there at hour 0. What the day buys is
    ordered in turn 0 and lands at hour 1, because the engine settles units
    BEFORE market inside a turn (F030), so a task that consumes it cannot run
    in turn 0 however early the order is queued.

    A good left OUT of this dict is available at hour 0 as far as wsr is
    concerned — it schedules the fetch whenever it likes. That is how a day
    came back with `PICKUP COW` in turn 0 beside the `BUY_ANIMAL COW` that
    pays for it: the cow is not in the shed yet, the engine refuses the pickup
    without a word (F047), and the plan reports success. So every good the day
    touches is named here, the entities AND what the ops eat.
    """
    out: dict[str, int] = {}
    for _cell, ops, entity in chains:
        wanted = [entity] if entity else []
        wanted += [OP_CONSUMES[op] for op in ops if op in OP_CONSUMES]
        for good in wanted:
            if good is None:
                continue
            held = int(seeds.get(good, 0)) or int(shed.get(good, 0))
            hour = 0 if held > 0 else 1
            out[good] = max(out.get(good, hour), hour)
    return out


def buy_orders(chains, seeds: dict, shed: dict,
               quotes: dict | None = None) -> tuple[list, int]:
    """The purchases the day needs, netted against what the farm already holds.

    Seeds live in `private["seeds"]` and bypass the shed (F001); a bought
    animal lands in the shed (`_commit_unit`), so the two are counted from
    different places and neither can stand in for the other.

    `quotes` prices the two DUAL goods, which the market quotes rather than a
    rule table fixing. Absent, they are priced at zero and the bill understates
    — so the caller passes the observation's own prices.
    """
    quotes = quotes or {}
    orders, bill = [], 0
    for (good, kind), units in sorted(needs(chains).items()):
        # Seeds live in `private["seeds"]`; animals and products in the shed.
        held = int(seeds.get(good, 0)) if kind == "SEED" \
            else int(shed.get(good, 0))
        short = max(0, units - held)
        if not short:
            continue
        if kind == "SEED":
            orders.append(["BUY_SEED", good, short])
            bill += short * int(CROP_RULES[good]["seed"])
        elif kind == "ANIMAL":
            orders.append(["BUY_ANIMAL", good, short])
            bill += short * int(ANIMAL_RULES[good]["cost"])
        else:
            # WHEAT and FERTILIZER: the two the market sells back (F033's
            # DUAL goods, engine :598 — BUY_PRODUCT accepts these and nothing
            # else), so what a FEED or a FERTILIZE eats is bought at the quote.
            orders.append(["BUY_PRODUCT", good, short])
            bill += short * int(quotes.get(good, 0))
    return orders, bill


def hire_orders(hands: int, hires_today: int, multiplier: int = 1
                ) -> tuple[list, int]:
    """`hands` HIRE orders and what they cost.

    The n-th hire of a day costs `farmHandCostMult · fib(n)` and the count
    resets at nightfall (F039), so the price depends on how many were already
    taken TODAY — not on how many hands are on the field.
    """
    orders = [["HIRE"] for _ in range(max(0, int(hands)))]
    bill = sum(hire_cost(int(hires_today) + i) * int(multiplier)
               for i in range(max(0, int(hands))))
    return orders, bill


def sell_rows(obs, harvest_expected: int, cash_needed: float, config=None,
              *, model=None, activity: int | None = None,
              forecast_obj=None) -> list:
    """Belief's per-hour SELL queue, or no rows if it cannot build one.

    Called through `market_queue`, which is belief's documented entry point and
    the only one that assembles the shed state, the forecast, the capacity
    guard and the season-end liquidation in one place. It is a UNIFORM spread
    unless a `model` is handed in: then the quantities stay the guard's decision
    and only their HOUR placement is re-timed by the slot circuit — the circuit
    that is built and benched at +1513 coins a season, and `market_queue`'s own
    `model=`/`activity=` is the wire that reaches it (#78). `activity` is the
    rival's own sell bucket, `tracker.MarketTracker.activity_bucket(step)`.
    Without a model the queue is the uniform spread and needs neither.
    """
    from agent.belief.shed import market_queue
    return market_queue(_sellable_obs(obs), forecast_obj=forecast_obj,
                        harvest_expected=int(harvest_expected),
                        cash_needed=float(cash_needed), config=config,
                        model=model, activity=activity)


def _sellable_obs(obs):
    """The observation with only PRODUCTS in the shed — a guard for #77.

    `private["shed"]` is keyed by `PRODUCTS + ANIMALS` (world/model.py:194,
    engine :171), `ShedState.sellable()` returns everything in it with a
    positive count, and `ladder._IX` is built over PRODUCTS alone — so a goose
    in the shed reaches `good_index` and raises `KeyError`. Measured here: ten
    raises in one season, and the seat stopped at day 20.

    This is #77 and belief owns it — the real fix is in `sellable()`, because
    SELL takes products and the three animals are shed items that are not
    products. Until then the day layer hands belief an input it can price,
    rather than catching the raise and quietly selling nothing, which would
    look exactly like a day with nothing to sell.
    """
    from agent.world.model import PRODUCTS
    private = obs.get("private", {}) if isinstance(obs, dict) else {}
    shed = private.get("shed") or {}
    if all(good in PRODUCTS for good in shed):
        return obs
    trimmed = dict(obs)
    trimmed["private"] = dict(private)
    trimmed["private"]["shed"] = {g: n for g, n in shed.items()
                                  if g in PRODUCTS}
    return trimmed


def merge(sells: list, hires: list, buys: list, *, cap: int = 10,
          turns: int = TURNS_PER_DAY) -> tuple[list, tuple]:
    """One queue, in the engine's settle order, capped per turn.

    Sells keep the head of each row because the engine settles them before
    hires and purchases (F032). Opening orders spill into later turns rather
    than over the cap, and whatever still has no room is REPORTED — an order
    the queue quietly forgot is a plan that silently does less than it says.
    """
    rows = [[] for _ in range(turns)]
    opening = [list(o) for o in hires] + [list(o) for o in buys]
    for turn in range(turns):
        row = [list(o) for o in (sells[turn] if turn < len(sells) else [])]
        row.sort(key=lambda o: QUEUE_RANK.get(o[0] if o else "", DEFAULT_RANK))
        while opening and len(row) < cap:
            row.append(opening.pop(0))
        rows[turn] = row[:cap]
    return rows, tuple(tuple(o) for o in opening)


def build(obs, chains, *, hands: int, harvest_expected: int = 0,
          config=None, cap: int = 10, model=None, activity: int | None = None,
          forecast_obj=None) -> DayMarket:
    """The whole day's market side, from the committed chains."""
    private = obs.get("private", {}) if isinstance(obs, dict) else {}
    farms = obs.get("farms", []) if isinstance(obs, dict) else []
    player = int(obs.get("player", 0)) if isinstance(obs, dict) else 0
    farm = farms[player] if len(farms) > player else {}

    quotes = (obs.get("market", {}) or {}).get("prices", {}) or {}
    buys, bill = buy_orders(chains, dict(private.get("seeds", {}) or {}),
                            dict(private.get("shed", {}) or {}), quotes)
    multiplier = int((config or {}).get("farmHandCostMult", 1) or 1) \
        if config is not None else 1
    hires, hire_bill = hire_orders(hands, int(farm.get("hires_today", 0)),
                                   multiplier)
    bill += hire_bill
    sells = sell_rows(obs, harvest_expected, float(bill), config,
                      model=model, activity=activity,
                      forecast_obj=forecast_obj)
    rows, dropped = merge(sells, hires, buys, cap=cap)
    return DayMarket(rows=rows, bill=int(bill), buys=tuple(map(tuple, buys)),
                     hires=len(hires), dropped=dropped,
                     sells=sum(1 for r in rows for o in r if o and o[0] == "SELL"))
