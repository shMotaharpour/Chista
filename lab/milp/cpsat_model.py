"""CP-SAT model builder for the Kaggriculture season planner."""
from __future__ import annotations

from ortools.sat.python import cp_model

from lab.milp.constants import (DAYS, TURNS_PER_DAY, CROPS, SELLABLE, ANIMALS,
                                LAND_PRICES, LAND_DAYS, STARTING_MONEY, SHED_CAP,
                                QUADRANT_TILES, FORCE_SELL_START)


def buy_wheat_cost(d):
    return 0  # v1: wheat planted, not bought; feed from harvest

def build(model: cp_model.CpModel, price_fn=None):
    """Build the season MILP. price_fn(item, day) → expected sell price (int)."""
    m = model
    days = range(DAYS)

    # ---------------- prices (expected, from market model)
    if price_fn is None:
        base_prices = {**{c: CROPS[c]["base"] for c in CROPS},
                       "EGG": 50, "MILK": 160, "WOOL": 200, "FERTILIZER": 100}
        price_fn = lambda item, d: base_prices.get(item, 1)

    # ---------------- variables
    plant = {}
    for c in CROPS:
        for d in days:
            plant[c, d] = m.NewIntVar(0, 30, f"plant_{c}_{d}")

    land = [m.NewBoolVar(f"land_{q}") for q in range(3)]

    crew = [m.NewIntVar(0, 12, f"crew_{d}") for d in days]

    sell = {}
    stock = {}
    for p in SELLABLE:
        for d in days:
            sell[p, d] = m.NewIntVar(0, 200, f"sell_{p}_{d}")
            stock[p, d] = m.NewIntVar(0, 100, f"stock_{p}_{d}")

    animals = {}
    for a in ANIMALS:
        for d in days:
            animals[a, d] = m.NewIntVar(0, 4, f"animals_{a}_{d}")

    mode = {}
    for a in ANIMALS:
        mode[a] = m.NewIntVar(0, 1, f"mode_{a}")  # 0=KEEP_ALIVE 1=FULL_CARE

    money = [m.NewIntVar(0, 500000, f"money_{d}") for d in days]
    m.Add(money[0] == STARTING_MONEY)

    # ---------------- derived: harvest schedule
    # one-time crops: plant at day d → harvest at day d+maxyd, yield = yield_base
    # ongoing: production every interval from first_yield_day, 1/day (no fert modeling)
    # Simplification (v1): one-time crops only (WHEAT, CARROT, MELON) + animals.
    # TOMATO/STRAWBERRY modeled as fixed schedule too (max_yield over lifespan).

    harvest = {}
    for p in SELLABLE:
        for d in days:
            harvest[p, d] = m.NewIntVar(0, 60, f"harvest_{p}_{d}")

    # one-time crops: EXACT harvest = yb * plant[c, d - maxyd]; zero on other days
    yb_map = {"WHEAT": 4, "CARROT": 3, "MELON": 6}
    for c in ("WHEAT", "CARROT", "MELON"):
        maxyd = CROPS[c]["maxyd"]
        for d in days:
            contribs = []
            if d >= maxyd:
                contribs.append(yb_map[c] * plant[c, d - maxyd])
            if contribs:
                m.Add(harvest[c, d] == sum(contribs))
            else:
                m.Add(harvest[c, d] == 0)

    # ongoing crops: strawberry — plant day d → productions d+10, d+12, ..., 4×1
    # tomato — plant day d → productions d+8, d+9, d+10, d+11 (4×1)
    # build EXACT per-day harvest = sum of all plants contributing on that day
    for crop, base in (("STRAWBERRY", 10), ("TOMATO", 8)):
        interval = CROPS[crop]["interval"] if CROPS[crop]["interval"] > 0 else 1
        n_prod = CROPS[crop]["maxyd"]  # 4 productions each
        # for each harvest day: exact equality (sum of contributing plants, or zero)
        for hd in days:
            contribs = []
            for d in days:
                for k in range(4):
                    plant_day = hd - base - k * interval
                    if plant_day == d:
                        contribs.append(plant[crop, d])
            if contribs:
                m.Add(harvest[crop, hd] == sum(contribs))
            else:
                m.Add(harvest[crop, hd] == 0)
    # animals: production per animal = 1 per interval day (KEEP_ALIVE) or 2 (FULL_CARE)
    for a, info in ANIMALS.items():
        for d in days:
            if d < info["first"]:
                m.Add(harvest[info["product"], d] == 0)
                continue
            is_prod_day = (d - info["first"]) % info["interval"] == 0
            if is_prod_day:
                # production day: base 1/animal regardless; +1/animal if FULL_CARE
                prod = m.NewIntVar(0, 8, f"prod_{a}_{d}")
                m.Add(prod == animals[a, d] * 2).OnlyEnforceIf(mode[a])
                m.Add(prod == animals[a, d]).OnlyEnforceIf(mode[a].Not())
                m.Add(harvest[info["product"], d] == prod)
            else:
                m.Add(harvest[info["product"], d] == 0)

    # animal population over time
    for a, info in ANIMALS.items():
        prev = m.NewIntVar(0, 4, f"init_{a}")
        m.Add(prev == 0)
        for d in days:
            bought = m.NewIntVar(0, 2, f"buy_{a}_{d}")
            if d == 0:
                m.Add(animals[a, d] == bought)
            else:
                m.Add(animals[a, d] == animals[a, d - 1] + bought)

    # ---------------- constraints
    # tiles: live plants ≤ 25 × unlocked quadrants
    for d in days:
        live = sum(plant[c, dd] for c in ("WHEAT", "CARROT", "MELON")
                   for dd in range(max(0, d - CROPS[c]["maxyd"] - 1), d + 1))
        unlocked = 1 + sum(land[q] * (d >= LAND_DAYS[q]) for q in range(3))
        m.Add(live <= QUADRANT_TILES * unlocked)

    # work capacity (VRP proxy): water/feed/care/harvest actions ≤ crew × 24
    for d in days:
        actions = sum(plant[c, d] for c in CROPS)
        actions += sum(1 for _ in [1])  # placeholder; real capacity modeled in executor
        # simplified: plant+harvest actions ≤ crew × TURNS
        m.Add(actions <= crew[d] * TURNS_PER_DAY)

    # shed
    for p in SELLABLE:
        prev = 0
        for d in days:
            inflow = harvest[p, d]
            outflow = sell[p, d]
            if d == 0:
                m.Add(stock[p, d] == inflow - outflow)
            else:
                m.Add(stock[p, d] == stock[p, d - 1] + inflow - outflow)

    # forced sell at end
    for p in SELLABLE:
        m.Add(sell[p, DAYS - 2] + sell[p, DAYS - 1] >= stock[p, FORCE_SELL_START])

    # crew hiring cost (fib cumulative — simplified: hire[d] hands, cost = fib sum)
    fib = [1, 1]
    while len(fib) < 16:
        fib.append(fib[-1] + fib[-2])

    for d in days:
        # crew can only grow within a day (hires spawn same day, act next turn)
        if d == 0:
            m.Add(crew[d] == 1)
        else:
            m.Add(crew[d] >= crew[d - 1])

    # ---------------- CASH PATH (was missing!): money[d+1] = money[d] + rev - costs
    for d in range(DAYS - 1):
        revenue = sum(sell[p, d] * price_fn(p, d) for p in SELLABLE)
        costs = sum(plant[c, d] * CROPS[c]["seed"] for c in CROPS)
        costs += sum(land[q] * LAND_PRICES[q] * (d == LAND_DAYS[q]) for q in range(3))
        costs += (crew[d + 1] - crew[d]) * 10 if d > 0 else 0
        costs += sum((animals[a, d + 1] - animals[a, d]) * ANIMALS[a]["cost"] for a in ANIMALS) if d > 0 else 0
        costs += buy_wheat_cost(d)
        m.Add(money[d + 1] == money[d] + revenue - costs)
        m.Add(money[d + 1] >= 0)

    # ---------------- objective: maximize final money
    revenue_terms = []
    for p in SELLABLE:
        for d in days:
            # revenue = sell × price (linear approx — piecewise in v2)
            revenue_terms.append(sell[p, d] * price_fn(p, d))
    cost_terms = []
    for c in CROPS:
        for d in days:
            cost_terms.append(plant[c, d] * CROPS[c]["seed"])
    for q in range(3):
        cost_terms.append(land[q] * LAND_PRICES[q])
    # animal costs
    for a, info in ANIMALS.items():
        for d in days:
            bought_var = None
            # handled via animals[a,d] delta
            if d > 0:
                delta = m.NewIntVar(0, 2, f"delta_{a}_{d}")
                m.Add(animals[a, d] - animals[a, d - 1] == delta)
                cost_terms.append(delta * info["cost"])
        
    m.Maximize(sum(revenue_terms) - sum(cost_terms))

    return m, {"plant": plant, "land": land, "sell": sell, "stock": stock,
               "crew": crew, "harvest": harvest, "animals": animals, "money": money}


def solve(time_limit: int = 60):
    model = cp_model.CpModel()
    m, vars_ = build(model)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_workers = 8
    status = solver.Solve(m)
    status_name = solver.StatusName(status)
    return status_name, solver, vars_
