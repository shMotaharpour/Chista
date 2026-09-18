"""The numbers and the turn order, as the engine has them.

Every value here is copied from the environment (its `kaggriculture.py` and the
`kaggriculture.json` specification it ships) and cited by line. This file is a
reference: it holds data, not decisions.

Where a rule is a *formula*, the formula is written out once (`hire_cost`,
`water_window_start`, `price`) rather than tabulated, because the engine computes it
too.
"""

from __future__ import annotations

from agent.world.model import PRODUCTS

# --- the season -------------------------------------------------------------- #

#: kaggriculture.json → configuration
EPISODE_STEPS: int = 720
ACT_TIMEOUT_S: int = 1
TURNS_PER_DAY: int = 24
DAYS: int = EPISODE_STEPS // TURNS_PER_DAY          # 30
BOARD_SIZE: int = 10
STARTING_MONEY: int = 3000
MAX_MARKET_ORDERS_PER_TURN: int = 10                # extras are dropped (:551,560)
SHED_CAPACITY: int = 100                            # seeds excluded (:553,867)
WEED_SPAWN_CHANCE: float = 0.005                    # per empty unlocked tile/day (:865)

#: `step = day * 24 + hour`; `day = step // turns_per_day` (kaggriculture.py:911)
#: and the end-of-day refresh runs when `(step + 1) % turns_per_day == 0` (:945).
#: The season ends when `step >= episodeSteps - 2` (:960), so the last playable
#: turn is step 719.

#: The four shed-access tiles, in the engine's NWSE order (:132-135): the inner
#: corners, one per quadrant. The shed is never locked; a unit standing on any of
#: these four may use PICKUP, DROP and PLACE-into-shed.
HALF: int = BOARD_SIZE // 2
SHED_ACCESS: tuple[tuple[int, int], ...] = ((HALF - 1, HALF - 1), (HALF, HALF - 1),
                                            (HALF - 1, HALF), (HALF, HALF))

#: The main farmer spawns on the first shed-access tile in NW order (:161-166) —
#: (4, 4) at board 10 — and is sent back there at the end of every day (:879).
#: A hired hand takes the least-occupied access tile, ties by NWSE (:533-541); with
#: the farmer on (4, 4) the first hire of the day lands on (5, 4), which starts
#: locked until the NE quadrant is bought.

# --- crops ------------------------------------------------------------------- #

#: CROPS (kaggriculture.py:11-17). `seed` is the BUY_SEED price (:603), `max_yield`
#: the cap on `yield_units`, `interval` the gap between scheduled productions for
#: ongoing crops, `ongoing` whether the plant keeps producing after a harvest.
CROP_RULES: dict[str, dict] = {
    "WHEAT":      {"seed": 10, "first_yield_day": 2,  "max_yield_day": 4,  "interval": 0, "max_yield": 6, "ongoing": False},
    "CARROT":     {"seed": 20, "first_yield_day": 2,  "max_yield_day": 3,  "interval": 0, "max_yield": 4, "ongoing": False},
    "TOMATO":     {"seed": 50, "first_yield_day": 8,  "max_yield_day": 8,  "interval": 1, "max_yield": 4, "ongoing": True},
    "STRAWBERRY": {"seed": 100, "first_yield_day": 10, "max_yield_day": 10, "interval": 2, "max_yield": 4, "ongoing": True},
    "MELON":      {"seed": 80, "first_yield_day": 10, "max_yield_day": 12, "interval": 0, "max_yield": 6, "ongoing": False},
}

#: A plant is watered once per day (:434-436). For a one-time crop the water is
#: worth yield only inside this window, inclusive, and worth 2 instead of 1 while
#: fertilised (:439-443); the cap is `max_yield`.
def water_window_start(crop: str) -> int:
    """`(max_yield_day + 1) // 2`, kaggriculture.py:440."""
    return (CROP_RULES[crop]["max_yield_day"] + 1) // 2


#: A plant is born `watered_today=False` with `consecutive_unwatered=1` (:222): the
#: planting day counts as the first missed day, so a seed planted and never watered
#: becomes a WEED at that night's refresh (:783-784). Two consecutive missed days is
#: the limit, for plants and for animals alike (:783, :817).

#: FERTILIZE spends 1 FERTILIZER from the unit's inventory — not from the shed — and
#: sets `fertilized_until_day = day + 2` (:478-481): active on `day`, `day+1`,
#: `day+2`. It needs a PLANT tile to stand on.

#: A one-time crop is harvested once: HARVEST needs `yield_units > 0` and, for crops,
#: `day - planted_day >= first_yield_day` (:446-468); the tile then becomes empty.
#: An ongoing crop keeps its tile and keeps producing: every `interval` days from
#: `first_yield_day` it adds 1 (2 if watered and fertilised that day), capped at
#: `max_yield` (:789-802). Once the cumulative production count reaches `max_yield`
#: its `max_lifespan_step` is set to `(next_day + 1) * turns_per_day` (:801-802);
#: from that step on, every other step costs 1 `yield_units`, and at 0 the tile
#: becomes a WEED (:752-766). One-time crops reach their lifespan at
#: `(planted_day + max_yield_day + 1) * turns_per_day` (:224).

# --- animals ----------------------------------------------------------------- #

#: ANIMALS (kaggriculture.py:19-23). `cost` is the BUY_ANIMAL price (:605),
#: `max_held` caps `yield_units` on the tile, `interval` the gap between scheduled
#: productions after `first_yield_day`.
ANIMAL_RULES: dict[str, dict] = {
    "GOOSE": {"cost": 300, "structure": "COOP",    "first_yield_day": 4, "interval": 1, "max_held": 4, "product": "EGG"},
    "COW":   {"cost": 400, "structure": "PASTURE", "first_yield_day": 8, "interval": 2, "max_held": 6, "product": "MILK"},
    "SHEEP": {"cost": 500, "structure": "PASTURE", "first_yield_day": 6, "interval": 3, "max_held": 6, "product": "WOOL"},
}

#: PLACE needs the animal in the unit's inventory and a matching structure that
#: holds none (:384-392). A newly placed animal has `consecutive_unfed = 0` (:236),
#: so it survives its first day unfed; from then on, two consecutive unfed days and
#: it escapes — the structure stays, the animal is gone (:813-820).
#: FEED spends 1 WHEAT from the unit's inventory, once per day (:505-513). CARE is
#: once per day (:524-530) and banks a bonus that is paid on the next scheduled
#: production, but only if that day is also fed (:826-830). COLLECT_FERTILIZER
#: yields 1 FERTILIZER when `fertilizer_available` (:515-522), which every surviving
#: animal sets at the end of each day, fed or not (:831).

# --- labour and land --------------------------------------------------------- #

#: LAND_ORDER / LAND_PRICES (kaggriculture.py:96-97). NW is free and always
#: unlocked; BUY_LAND takes the next quadrant in this order and unlocks its 25 tiles
#: (:712-725). Three purchases, then BUY_LAND is a no-op.
LAND_ORDER: tuple[str, ...] = ("NE", "SW", "SE")
LAND_PRICES: tuple[int, ...] = (1000, 2000, 4000)

#: HIRE costs `farmHandCostMult * fib(n)` for the n-th hire of the day, and
#: `hires_today` resets at the end of the day (:698-709, :881).
HIRE_SEQUENCE: tuple[int, ...] = (1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233,
                                  377, 610, 987, 1597, 2584, 4181, 6765)


def hire_cost(n_already_today: int) -> int:
    """The price of the next hire, kaggriculture.py:698-699 (`_fib` indexed at 1)."""
    return HIRE_SEQUENCE[min(n_already_today, len(HIRE_SEQUENCE) - 1)]


# --- the town ---------------------------------------------------------------- #

#: SHOPS (kaggriculture.py:103-112). A shop instance consumes one of each of its
#: products every `townShopSellInterval` turns; a single-product shop consumes 2
#: (:741). Instances are drawn with replacement at most `MAX_SHOP_INSTANCES` times
#: (:890-891), so duplicates are normal and variety is not guaranteed.
SHOPS: dict[str, tuple[str, ...]] = {
    "BAKERY":         ("EGG", "WHEAT"),
    "PIZZA_SHOP":     ("MILK", "TOMATO", "WHEAT"),
    "BRUNCH_SPOT":    ("EGG", "WHEAT", "STRAWBERRY"),
    "YARN_STORE":     ("WOOL",),
    "ICE_CREAM_SHOP": ("STRAWBERRY", "MILK", "WHEAT"),
    "PET_CAFE":       ("CARROT",),
    "SMOOTHIE_SHOP":  ("STRAWBERRY", "MILK"),
    "FARMERS_MARKET": ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY"),
}

#: Every product except fertiliser, one unit each, every `townCenterSellInterval`
#: turns — once a day at the defaults, flat all season (:114, :745-747).
TOWN_CENTER_PRODUCTS: tuple[str, ...] = tuple(p for p in PRODUCTS if p != "FERTILIZER")

MAX_SHOP_INSTANCES: int = 8
SHOP_UNLOCK_INTERVAL_DAYS: int = 3      # the first unlock is on day 3 (:886)
SHOP_SELL_INTERVAL_TURNS: int = 4
CENTER_SELL_INTERVAL_TURNS: int = 24

# --- the turn, in the engine's order ----------------------------------------- #

#: kaggriculture.py:913-946, in order:
#: 1. every unit acts — the farmer, then the hands in index order; a unit's action
#:    is applied to the tile it stands on, and an illegal one is a silent no-op.
#:    PLANT is atomic per crop: if the turn's PLANT requests for a crop exceed the
#:    seeds held, ALL of them become PASS (:920-933).
#: 2. the market runs — at most `MAX_MARKET_ORDERS_PER_TURN` orders per player, in
#:    order; HIRE and BUY_LAND resolve atomically first, in player order, then the
#:    SELL/BUY_* orders are quoted and committed one unit at a time, both players
#:    quoted from the same pre-commit inventory (:562-628).
#: 3. the town consumes (:728-749).
#: 4. plants decay (:752-766).
#: 5. at the day's last turn, the end-of-day refresh (:860-891).
UNIT_ACTIONS_FIRST: str = "units act before the market and the town"
TURN_ORDER: tuple[str, ...] = ("units", "market", "town", "decay", "end_of_day")

#: End-of-day, in order (:875-891): plants refresh (a missed watering turns the
#: plant into a weed; ongoing crops may produce), animals refresh (a missed feeding
#: counts toward escape; a surviving animal makes fertiliser available and may
#: produce), weeds may spawn on empty unlocked tiles, every unit's inventory is
#: dropped into the shed up to `SHED_CAPACITY` with the overflow discarded, the
#: farmer returns to the spawn tile, all hands are removed, `hires_today` resets,
#: and every inventory is emptied. A shop may unlock (up to 8 instances).
END_OF_DAY_ORDER: tuple[str, ...] = (
    "plants_refresh", "animals_refresh", "weeds_spawn", "inventories_dropped",
    "farmer_respawned", "hands_removed", "hires_reset", "inventories_cleared",
    "shop_may_unlock",
)


#: Which structure each species lives in. A definition, so it lives beside the rules it is
#: read from: the builder and the domain filter both need it.
ANIMAL_STRUCTURE: dict[str, str] = {a: spec["structure"] for a, spec in ANIMAL_RULES.items()}


#: The engine's default board and its per-turn market limit: it executes this many orders a
#: turn and drops the rest in silence.
DEFAULT_BOARD: int = 10
MAX_ORDERS_PER_TURN: int = 10
