"""One vocabulary for goods, shops and the shed ring — derived from the engine.

Four dialects grew for the same facts (the engine's own names, `tile_dp`'s
`BUILD`/`PLACE_ANIMAL`, `secretary/models.py`, `world/replay_agent.py`), and an
op the engine does not know is refused in silence (F047). This module is the
single place the derived names live, so the next layer imports instead of
re-declaring. Nothing here is typed by hand: every table comes from
`kaggriculture` (R002), and `tests/test_market_analyzer.py` pins the two
properties the sell side depends on (the 7/2 split and the shop baskets).

Not yet here: the op sets (`WORKER_OPS`, `MARKET_OPS`). They are the executor's
vocabulary and belong to the compiler work, where a guard can be written
against the engine's own literals; inventing them now would be a fifth dialect
with no caller.
"""

from __future__ import annotations

from kaggle_environments.envs.kaggriculture import kaggriculture as K

# --- goods ------------------------------------------------------------------ #

GOODS: tuple[str, ...] = tuple(K.PRODUCTS)                        # 9
DUAL: tuple[str, ...] = ("WHEAT", "FERTILIZER")                   # buyable AND sellable
SELL_ONLY: tuple[str, ...] = tuple(g for g in GOODS if g not in DUAL)   # 7
G_IX: dict[str, int] = {g: i for i, g in enumerate(GOODS)}

# --- the town --------------------------------------------------------------- #

SHOP_TYPES: tuple[str, ...] = tuple(sorted(K.SHOPS))
SHOP_BASKET: dict[str, tuple[tuple[str, ...], int]] = {
    s: (tuple(K.SHOPS[s]), 2 if len(K.SHOPS[s]) == 1 else 1) for s in SHOP_TYPES
}
CENTER_PRODUCTS: tuple[str, ...] = tuple(K.TOWN_CENTER_PRODUCTS)
UNLOCK_INTERVAL = 3        # shops unlock every 3 days
MAX_SHOP_INSTANCES = K.MAX_SHOP_INSTANCES

# --- the farm's own limits --------------------------------------------------- #

SHED_CAP = 100             # engine default shedCapacity, per player, all goods
MAX_ORDERS = 10            # engine default maxMarketOrdersPerTurn, per turn
SHOP_INTERVAL = 4          # townShopSellInterval
CENTER_INTERVAL = 24       # townCenterSellInterval

# The four shed-access tiles, from the engine's own helper. Three start LOCKED;
# DROP/PICKUP resolve before the LOCKED guard, which is what keeps the shed
# reachable from them.
SHED_ACCESS: frozenset[tuple[int, int]] = frozenset(
    tuple(t) for t in K._shed_access_tiles(10))
