"""Secretary A: the day compiler — one chain, one unit, one day the engine accepts.

Issue #14's routing half, and the reason the replan rung was inert (F047): the
tile graph prices a chain **as if the worker already stands on the tile and its
inputs are in its bag**. `tile_dp/graph.py::_exec_chain` realises the buys and
the pickups on a scratch sim where the worker happens to start on a shed-access
tile, so every chain in the shipped graph is realisable — but only from there.
At runtime the units are wherever they are, and an op whose precondition is not
met is refused **in silence**: the day passes and nothing happens.

So this module answers one question per unit: *what is the exact sequence of
ops that gets this unit from where it stands, through the shed if it needs to
carry something, onto its tile, through its chain, and back to the shed with the
harvest?* It emits only ops whose preconditions it has itself satisfied, and it
reports what it had to drop.

## The engine facts this is built on (all read out of `kaggriculture.py`)

- **Movement is one tile per op, and the op name is the direction**: the action
  string is `"NORTH" | "SOUTH" | "EAST" | "WEST"` (`_apply_unit_action:323`), no
  `MOVE_` prefix. Off-board is a silent no-op; LOCKED tiles are walkable
  (`:326-331`) — a hand can spawn on one.
- **PICKUP and DROP require a shed-access tile** (`_is_shed_adjacent:344,359`),
  which is exactly `_shed_access_tiles`: the four tiles of the shed block,
  `(4,4) (5,4) (4,5) (5,5)` at the default board size. Anywhere else both are
  silent no-ops.
- **Seeds never travel.** `BUY_SEED` credits `private["seeds"]` and `PLANT`
  consumes that directly (`:367-368, 425-427`), so a crop needs a market order
  and a worker standing on the tile — never a pickup.
- **`BUY_PRODUCT` and `BUY_ANIMAL` land in the shed** (`:670, 685`) and are
  refused outright while `sum(shed) >= shedCapacity` (`:667, 682`). So a buy is
  only real if the shed has room on that turn — the seller's business (#15).
- **A purchase is usable the turn after it lands** (F030): the unit acts before
  the market, so a `BUY_*` at hour `h` can only be picked up at hour `h+1` or
  later.
- **`DROP` empties the whole bag** and destroys whatever does not fit
  (`:343-356`), so it is scheduled as late as the day allows: the day's sales
  have already made room, and the harvest is in the shed for the next market.
- **`HARVEST` and `COLLECT_FERTILIZER` put goods in the unit's bag**, and a
  `SELL` can only reach the shed (`_commit_unit`), so the walk back to the shed
  is what turns a harvest into money.
- **Hiring costs the engine's Fibonacci ladder** and the hand spawns on a
  shed-access tile (`_spawn_hand:533`), which is why a new hand's first pickup
  is free and its trip to a tile is not.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from kaggle_environments.envs.kaggriculture import kaggriculture as K

TURNS_PER_DAY = 24          # the run configuration's turnsPerDay (F058)
DEFAULT_BOARD = 10          # boardSize
MAX_ORDERS_PER_TURN = 10    # F031 — the engine executes 10 and drops the rest in silence

#: op name -> (dx, dy); the engine's own table, never transcribed (R002).
MOVE_OPS: dict[str, tuple[int, int]] = dict(K.FARMER_MOVES)

#: the goods a worker must be carrying, and the op that eats them.
CARRIED_BY_OP: dict[str, str] = {"FERTILIZE": "FERTILIZER", "FEED": "WHEAT"}

#: ops that need the entity itself in the bag (an animal), not a product.
PLACE_OPS = ("PLACE", "PLACE_ANIMAL")

#: ops that put goods into the unit's bag, and therefore want a shed trip after.
BAGGING_OPS = ("HARVEST", "COLLECT_FERTILIZER")

#: ops that name an entity and cannot run without one.
ENTITY_OPS = ("PLANT", "BUILD", "PLACE", "PLACE_ANIMAL")


def shed_access(board: int = DEFAULT_BOARD) -> tuple[tuple[int, int], ...]:
    """The tiles from which PICKUP and DROP work — the engine's own list."""
    return tuple((int(x), int(y)) for x, y in K._shed_access_tiles(board))


def walk(start: tuple[int, int], goal: tuple[int, int],
         board: int = DEFAULT_BOARD) -> list[tuple[str, ...]]:
    """The move ops that carry a unit from `start` to `goal`, x first then y.

    Shortest by construction (Manhattan) and always legal: the engine only
    refuses a move that leaves the board, and this never emits one. Ties are
    broken x-then-y so a plan is a function of its inputs, not of a hash order.
    """
    x, y = int(start[0]), int(start[1])
    gx, gy = int(goal[0]), int(goal[1])
    out: list[tuple[str, ...]] = []
    while x != gx:
        step = "EAST" if gx > x else "WEST"
        x += MOVE_OPS[step][0]
        out.append((step,))
    while y != gy:
        step = "SOUTH" if gy > y else "NORTH"
        y += MOVE_OPS[step][1]
        out.append((step,))
    if not (0 <= x < board and 0 <= y < board):
        raise ValueError(f"walk({start}, {goal}) leaves the board")
    return out


def nearest_shed(pos: tuple[int, int], board: int = DEFAULT_BOARD) -> tuple[int, int]:
    """The closest shed-access tile, ties broken by the engine's NWSE order."""
    tiles = shed_access(board)
    return min(tiles, key=lambda t: (abs(t[0] - pos[0]) + abs(t[1] - pos[1]), tiles.index(t)))


def op_turns(ops: Sequence[str], entity: str | None) -> list[tuple[str, ...]]:
    """A chain's ops -> the worker's op per turn, in canonical order.

    The same expansion `agent/replan.py::chain_turns` has always done, moved
    here so the compiler owns the whole vocabulary: a chain names an abstract
    `PLANT`, the engine wants `["PLANT", "WHEAT"]`, and `BUILD` becomes the
    structure the entity needs.
    """
    out: list[tuple[str, ...]] = []
    for op in ops:
        if op == "NO_ACT" or op == "PASS":
            out.append(("PASS",))
        elif op == "BUILD":
            if entity not in K.ANIMALS:
                raise ValueError(f"BUILD {entity!r} is not an animal")
            out.append((f"BUILD_{K.ANIMALS[entity]['structure']}",))
        elif op in PLACE_OPS:
            if entity not in K.ANIMALS:
                raise ValueError(f"PLACE {entity!r} is not an animal")
            out.append(("PLACE", str(entity)))
        elif op == "PLANT":
            if entity not in K.CROPS:
                raise ValueError(f"PLANT {entity!r} is not a crop")
            out.append(("PLANT", str(entity)))
        else:                                   # WATER, HARVEST, DIG, FERTILIZE,
            out.append((str(op),))              # FEED, CARE, COLLECT_FERTILIZER
    return out


@dataclass(frozen=True)
class Need:
    """One thing the market must deliver, and the last turn it may land on."""

    hour: int                # latest landing hour (a pickup needs h_buy < h_pick)
    order: tuple             # the engine order, e.g. ("BUY_SEED", "WHEAT", 1)
    reason: str              # which op wanted it


@dataclass(frozen=True)
class UnitRoute:
    """One unit's whole day, plus what it cost the plan to build it."""

    unit: int
    ops: tuple[tuple[str, ...], ...]     # TURNS_PER_DAY entries, PASS-padded
    needs: tuple[Need, ...]
    dropped: tuple[str, ...]             # ops the day could not carry
    hours_used: int
    drop_hour: int | None
    walked: int                          # move ops spent
    #: (hour, item, units) the day's DROP puts in the shed. A SELL can only
    #: reach the shed, so this is what the day's market may actually sell.
    arrivals: tuple[tuple[int, str, int], ...] = ()

    @property
    def carry_trips(self) -> int:
        """Shed trips this route spends — the travel the tile graph does not see."""
        return sum(1 for op in self.ops if op and op[0] == "PICKUP")


def carried_item(op: tuple[str, ...], entity: str | None) -> str | None:
    """What this op needs in the unit's bag, or None if it needs nothing."""
    if op[0] in PLACE_OPS:
        return entity
    return CARRIED_BY_OP.get(op[0])


def _buy_order(item: str) -> tuple:
    if item in K.ANIMALS:
        return ("BUY_ANIMAL", item, 1)
    if item in K.CROPS:
        return ("BUY_SEED", item, 1)
    return ("BUY_PRODUCT", item, 1)


def route_unit(ops: Sequence[str], entity: str | None, pos: tuple[int, int], *,
               unit: int = 0, hour: int = 0, hours: int = TURNS_PER_DAY,
               target: tuple[int, int] | None = None,
               carried: Mapping[str, int] | None = None,
               harvest_yields: Mapping[str, int] | None = None,
               drop: bool = True, board: int = DEFAULT_BOARD) -> UnitRoute:
    """Compile one chain into one unit's day.

    `pos` is where the unit stands at `hour`; `target` is the tile the chain
    works (defaults to `pos` — the tile it is standing on). `carried` is what
    the unit already holds, so a route does not buy what it already has.

    Ops whose preconditions cannot be met inside `hours` are **dropped and
    named**, never emitted: an op the engine refuses is worse than an op that
    was never sent, because it looks like work in the log (F047).
    """
    target = (int((target or pos)[0]), int((target or pos)[1]))
    bag = {k: int(v) for k, v in (carried or {}).items()}
    yields = dict(harvest_yields or {})
    bagged: dict[str, int] = {}
    bagged_ops = 0
    turns = op_turns(ops, entity)
    seq: list[tuple[str, ...]] = []
    needs: list[Need] = []
    dropped: list[str] = []
    at = (int(pos[0]), int(pos[1]))
    walked = 0
    drop_hour: int | None = None

    def free() -> int:
        """Turns left, counting from `hour`."""
        return hours - (hour + len(seq))

    def push(steps: Iterable[tuple[str, ...]]) -> None:
        nonlocal walked
        for step in steps:
            if step and step[0] in MOVE_OPS:
                walked += 1
            seq.append(step)

    for op in turns:
        item = carried_item(op, entity)
        if item is not None and bag.get(item, 0) <= 0:
            shed = nearest_shed(at, board)
            back = walk(shed, target, board)
            trip = walk(at, shed, board) + [("PICKUP", item, 1)] + back
            if free() < len(trip) + 1:
                dropped.append(" ".join(op))
                continue
            push(trip)
            # the pickup happens before that turn's market, so the buy must land
            # on the turn BEFORE the pickup (F030)
            pickup_turn = hour + len(seq) - len(back) - 1
            needs.append(Need(hour=pickup_turn - 1, order=_buy_order(item),
                              reason=" ".join(op)))
            bag[item] = 1
            at = target
        elif at != target:
            trip = walk(at, target, board)
            if free() < len(trip) + 1:
                dropped.append(" ".join(op))
                continue
            push(trip)
            at = target
        if free() < 1:
            dropped.append(" ".join(op))
            continue
        if op[0] == "PLANT":
            # seeds ride in private["seeds"] and PLANT consumes them directly:
            # one market order, and a worker standing on the tile. No carrying.
            # The buy still lands after that turn's units, so it must be on an
            # EARLIER turn than the plant (F030) - the same rule as a pickup.
            needs.append(Need(hour=hour + len(seq) - 1,
                              order=("BUY_SEED", op[1], 1), reason="PLANT"))
        push([op])
        if op[0] == "HARVEST":
            bagged_ops += 1
            # the tile's own yield_units is what the engine hands over (the
            # caller reads it off the observation, so this is a measurement);
            # without it the drop still happens and the sell waits for tomorrow
            item = str(op[1]) if len(op) > 1 else (entity or "")
            bagged[item] = bagged.get(item, 0) + int(yields.get(item, 0))
        elif op[0] == "COLLECT_FERTILIZER":
            bagged_ops += 1
            bagged["FERTILIZER"] = bagged.get("FERTILIZER", 0) + 1

    arrivals: list[tuple[int, str, int]] = []
    if drop and bagged_ops > 0:
        shed = nearest_shed(at, board)
        trip = walk(at, shed, board)
        if free() >= len(trip) + 1:
            push(trip)
            push([("DROP",)])
            at = shed
            drop_hour = hour + len(seq) - 1
            # the DROP resolves before that turn's market (units act, then the
            # market), so goods dropped at hour h can be sold at hour h
            arrivals = [(drop_hour, item, units)
                        for item, units in sorted(bagged.items()) if units > 0]

    head = (("PASS",),) * max(0, int(hour))
    padded = head + tuple(seq)
    padded = padded + (("PASS",),) * max(0, hours - len(padded))
    route = UnitRoute(unit=unit, ops=padded[:hours], needs=tuple(needs),
                      dropped=tuple(dropped), hours_used=len(seq),
                      drop_hour=drop_hour, walked=walked,
                      arrivals=tuple(arrivals))
    _assert_shed_ops_are_reachable(route, pos, 0, board)
    return route


def _assert_shed_ops_are_reachable(route: UnitRoute, start: tuple[int, int],
                                   hour: int, board: int) -> None:
    """Every PICKUP/DROP the route emits must be on a shed-access tile.

    The compiler's own precondition check: the engine refuses both ops anywhere
    else **in silence** (`:344, 359`), so a route that emits one is a plan that
    lies about its day. Simulating the walk is cheap and it is the only way to
    know where the unit actually is at that turn.
    """
    tiles = set(shed_access(board))
    pos = (int(start[0]), int(start[1]))
    for index, op in enumerate(route.ops):
        if not op:
            continue
        if op[0] in MOVE_OPS:
            dx, dy = MOVE_OPS[op[0]]
            pos = (pos[0] + dx, pos[1] + dy)
        elif op[0] in ("PICKUP", "DROP") and pos not in tiles:
            raise AssertionError(
                f"unit {route.unit} would {op[0]} at {pos} on turn {hour + index}: "
                "not a shed-access tile, so the engine refuses it in silence")


# --------------------------------------------------------------- the whole day

def manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(int(a[0]) - int(b[0])) + abs(int(a[1]) - int(b[1]))


def spawn_position(occupied: Sequence[tuple[int, int]],
                   board: int = DEFAULT_BOARD) -> tuple[int, int]:
    """Where the engine puts the next hired hand (`_spawn_hand:533`).

    Least-occupied shed-access tile, ties broken by the engine's NWSE order — so
    a new hand always starts within reach of the shed, which is what makes its
    first pickup free and its walk to a tile the only travel it pays.
    """
    tiles = shed_access(board)
    counts = {tile: 0 for tile in tiles}
    for pos in occupied:
        pos = (int(pos[0]), int(pos[1]))
        if pos in counts:
            counts[pos] += 1
    return min(tiles, key=lambda t: (counts[t], tiles.index(t)))


@dataclass(frozen=True)
class DayPlan:
    """A whole farm-day: what each unit does, and what the market does."""

    units: tuple[tuple[tuple[str, ...], ...], ...]
    market: tuple[tuple[tuple, ...], ...]        # per hour, in F032 order
    needs: tuple[Need, ...]
    hires: int
    dropped: tuple[str, ...]                     # ops the day could not carry
    unplaced: tuple[Need, ...]                   # needs the market could not take
    idle_units: int                              # units with no tile to work
    assignments: tuple[tuple[int, int] | None, ...]   # unit -> (tile x, y)

    def as_plan(self) -> dict:
        """The dispatcher's shape (`agent/dispatch.py`)."""
        return {"units": [list(unit) for unit in self.units],
                "market": [list(row) for row in self.market]}


def needs_cost(needs: Sequence[Need], prices: Mapping[str, float] | None = None) -> float:
    """What the market orders cost at the engine's own tables and today's quotes."""
    total = 0.0
    for need in needs:
        op = need.order
        if op[0] == "BUY_SEED":
            total += float(K.CROPS[op[1]]["seed"])
        elif op[0] == "BUY_ANIMAL":
            total += float(K.ANIMALS[op[1]]["cost"])
        else:
            total += float((prices or {}).get(op[1], 0.0))
    return total


def hire_cost(hires_today: int, count: int) -> float:
    """The engine's Fibonacci ladder, from the hires already made today (F039)."""
    return float(sum(K._hire_cost(int(hires_today) + i) for i in range(int(count))))


def merge_market(sells: Sequence[Sequence[Sequence]], needs: Sequence[Need], *,
                 shed_total: float, capacity: int, hires: int = 0,
                 hours: int = TURNS_PER_DAY,
                 harvest_sells: Sequence[tuple[int, str, int]] = ()
                 ) -> tuple[list[list[tuple]], list[Need]]:
    """Sells, hires and the routes' buys -> one queue, in F032 order per turn.

    F032 fixes the order inside a turn: land, sells, hires, purchases. It matters
    because the sells of a turn free shed room *before* that turn's buys, and
    `BUY_PRODUCT`/`BUY_ANIMAL` are refused while the shed is full (`:667, 682`).
    A buy whose latest hour passes unplaced is returned, never dropped quietly.
    """
    queue: list[list[tuple]] = [[] for _ in range(hours)]
    for hour in range(hours):
        row = sells[hour] if hour < len(sells) else []
        queue[hour].extend(tuple(order) for order in (row or []))
    # The day's own harvest, sold the same day: a unit's DROP resolves before
    # that turn's market, so the goods are in the shed in time for a SELL in the
    # same row. Without this the day's produce waits for tomorrow's queue (the
    # season measured 2,069 coins against the greedy stub's 2,840 with every
    # harvest unsold) and the whole milestone earns nothing.
    for hour, item, units in sorted(harvest_sells):
        for target in range(max(0, int(hour)), hours):
            if len(queue[target]) < MAX_ORDERS_PER_TURN:
                queue[target].append(("SELL", str(item), int(units)))
                break
    room = float(capacity) - float(shed_total)
    pending = sorted(needs, key=lambda n: (n.hour, n.order))
    unplaced: list[Need] = []
    for hour in range(hours):
        room += sum(float(order[2]) for order in queue[hour]
                    if order and order[0] == "SELL")
        if hour == 0 and hires:
            queue[hour].extend(tuple(["HIRE"]) for _ in range(int(hires)))
        for need in list(pending):
            if need.hour < hour:
                pending.remove(need)
                unplaced.append(need)          # its deadline passed: never sent
                continue
            lands_in_shed = need.order[0] in ("BUY_PRODUCT", "BUY_ANIMAL")
            if lands_in_shed and room < 1.0:
                continue                        # no room this turn: try the next
            if len(queue[hour]) >= MAX_ORDERS_PER_TURN:
                break
            queue[hour].append(need.order)
            pending.remove(need)
            if lands_in_shed:
                room -= 1.0
        if len(queue[hour]) > MAX_ORDERS_PER_TURN:
            raise ValueError(
                f"hour {hour} queues {len(queue[hour])} market orders; the engine "
                f"executes {MAX_ORDERS_PER_TURN} and drops the rest in silence (F031)")
    unplaced.extend(pending)
    return queue, unplaced


def plan_day(tiles: Sequence[tuple[tuple[int, int], Sequence[str], str | None]],
             units: Sequence[tuple[int, int]], *, new_hands: int = 0,
             bags: Sequence[Mapping[str, int]] = (), shed: Mapping[str, int] | None = None,
             money: float = 0.0, hires_today: int = 0,
             sells: Sequence[Sequence[Sequence]] | None = None,
             yields: Sequence[Mapping[str, int]] | None = None,
             values: Sequence[float] | None = None,
             prices: Mapping[str, float] | None = None,
             capacity: int = 100, board: int = DEFAULT_BOARD,
             hours: int = TURNS_PER_DAY) -> DayPlan:
    """The whole farm's day: assign the priced tiles to the units and compile.

    `tiles` is what the contractor priced — `(tile, chain ops, entity)` per column,
    in the caller's order. `units` are the positions that exist now (farmer first,
    then hands in order); `new_hands` are the ones hired at hour 0, placed by the
    engine's own spawn rule so their routes start where they will really stand.

    Assignment is nearest-first, farmer first, ties by tile position — a function
    of its inputs, not of a hash order. A unit with no tile to work gets a PASS
    day, and every unit is *counted*: `idle_units` says how much of the farm the
    day's chains did not reach.
    """
    positions = [(int(p[0]), int(p[1])) for p in units]

    def assign(order: list[int], free: list[int]) -> list[int | None]:
        """Nearest-first assignment of the free tile columns to these units."""
        out: list[int | None] = []
        for index in order:
            if not free:
                out.append(None)
                continue
            best = min(free, key=lambda i: (manhattan(positions[index], tiles[i][0]),
                                            tiles[i][0]))
            free.remove(best)
            out.append(best)
        return out

    existing = list(range(len(positions)))
    free = list(range(len(tiles)))
    assigned: list[int | None] = [None] * len(positions)
    for index, column in zip(existing, assign(existing, free)):
        assigned[index] = column

    def compile_one(index: int, column: int, start: int) -> UnitRoute:
        tile, ops, entity = tiles[int(column)]
        return route_unit(ops, entity, positions[index], unit=index, hour=start,
                          target=tile,
                          carried=bags[index] if index < len(bags) else None,
                          harvest_yields=(yields[column] if yields
                                          and column < len(yields) else None),
                          board=board, hours=hours)

    routes = [compile_one(i, int(assigned[i]), 0) for i in existing
              if assigned[i] is not None]

    # The engine settles a HIRE inside turn 0's market, i.e. AFTER that turn's unit
    # actions (`_process_market`), so the tile the farmer just walked off is free
    # for the hand. Predicting the spawn from the day-start positions puts every
    # new hand one tile out and every one of its ops on the wrong tile - measured:
    # a hand planned for (5,4) spawned on (4,4) and its PLANT was refused in
    # silence (F047). Simulate turn 0, then spawn.
    settled = [positions[i] for i in existing]
    for route in routes:
        first = route.ops[0] if route.ops else ("PASS",)
        if first and first[0] in MOVE_OPS:
            dx, dy = MOVE_OPS[first[0]]
            settled[route.unit] = (settled[route.unit][0] + dx, settled[route.unit][1] + dy)
    occupied = list(settled)
    for _ in range(int(new_hands)):
        positions.append(spawn_position(occupied, board))
        occupied.append(positions[-1])

    hired = list(range(len(existing), len(positions)))
    assigned.extend([None] * (len(positions) - len(assigned)))
    for index, column in zip(hired, assign(hired, free)):
        assigned[index] = column

    def compile_all() -> list[UnitRoute]:
        out: list[UnitRoute] = []
        for index in range(len(positions)):
            if assigned[index] is None:
                continue
            out.append(compile_one(index, int(assigned[index]),
                                   0 if index < len(existing) else 1))
        return out

    routes = compile_all()

    def day_bill(chosen: list[UnitRoute]) -> tuple[float, float]:
        """(the market bill, what tomorrow's seeds for the same work would cost).

        The reserve is the plan's own seed bill, not a tuned fraction: a farm that
        spends its last coin on animals cannot re-plant tomorrow. Measured leak
        without it: six COWs bought in one day cost 2,472 of 2,988 coins, and the
        season ended at 0-734 coins against the greedy stub's 2,840.
        """
        wants = [need for route in chosen for need in route.needs]
        bill = needs_cost(wants, prices) + hire_cost(hires_today, new_hands)
        seed_reserve = needs_cost([n for n in wants if n.order[0] == "BUY_SEED"], prices)
        return bill, seed_reserve

    # A refused buy is a silent no-op (F031), so the day is trimmed until the
    # purse covers it — least valuable plan first, where "value" is the tile DP's
    # own V_0 (`PricedBoard.tile_values`), so the trim is a number the contractor
    # already computed rather than a distance heuristic.
    for _ in range(len(positions) + 1):
        bill, reserve = day_bill(routes)
        if bill + reserve <= float(money):
            break
        if not routes:
            break
        worst = min(routes, key=lambda route: (values[assigned[route.unit]]
                                               if values and assigned[route.unit] is not None
                                               else 0.0, -route.unit))
        routes.remove(worst)

    needs = [need for route in routes for need in route.needs]
    arrivals = [arrival for route in routes for arrival in route.arrivals]
    queue, unplaced = merge_market(sells or [], needs, shed_total=sum(
        int(v) for v in (shed or {}).values()), capacity=capacity,
        hires=new_hands, hours=hours, harvest_sells=arrivals)

    by_unit: list[tuple[tuple[str, ...], ...]] = [((("PASS",),) * hours)
                                                  for _ in positions]
    for route in routes:
        by_unit[route.unit] = route.ops
    # a route whose buy could not be placed is a route that would emit no-ops:
    # its day becomes PASS rather than a lie
    bad = {need for need in unplaced}
    if bad:
        for route in routes:
            if bad.intersection(route.needs):
                by_unit[route.unit] = (("PASS",),) * hours

    return DayPlan(
        units=tuple(by_unit), market=tuple(tuple(row) for row in queue),
        needs=tuple(needs), hires=int(new_hands),
        dropped=tuple(op for route in routes for op in route.dropped),
        unplaced=tuple(unplaced),
        idle_units=sum(1 for column in assigned if column is None),
        assignments=tuple((tiles[c][0] if c is not None else None)
                          for c in assigned))

