"""The town and the shed ring: shops, unlock cadence, limits. Engine-derived.

Goods, resources, actions and the chain ops live in `world/model.py`; this module
keeps what the sell side needs about the town. See `docs/ARCHITECTURE.md` §1.
"""

from __future__ import annotations

from kaggle_environments.envs.kaggriculture import kaggriculture as K

from world.model import DUAL, GOODS, SHED_ACCESS

#: The 7 goods only the rival can sell against us; the 2 dual goods both trade.
SELL_ONLY: tuple[str, ...] = tuple(g for g in GOODS if g not in DUAL)
G_IX: dict[str, int] = {g: i for i, g in enumerate(GOODS)}

SHOP_TYPES: tuple[str, ...] = tuple(sorted(K.SHOPS))
SHOP_BASKET: dict[str, tuple[tuple[str, ...], int]] = {
    s: (tuple(K.SHOPS[s]), 2 if len(K.SHOPS[s]) == 1 else 1) for s in SHOP_TYPES
}
CENTER_PRODUCTS: tuple[str, ...] = tuple(K.TOWN_CENTER_PRODUCTS)
UNLOCK_INTERVAL = 3        # shops unlock every 3 days
MAX_SHOP_INSTANCES = K.MAX_SHOP_INSTANCES

SHED_CAP = 100             # engine default shedCapacity, per player, all goods
MAX_ORDERS = 10            # engine default maxMarketOrdersPerTurn, per turn
SHOP_INTERVAL = 4          # townShopSellInterval
CENTER_INTERVAL = 24       # townCenterSellInterval

__all__ = [
    "GOODS", "DUAL", "SELL_ONLY", "G_IX", "SHOP_TYPES", "SHOP_BASKET",
    "CENTER_PRODUCTS", "UNLOCK_INTERVAL", "MAX_SHOP_INSTANCES", "SHED_CAP",
    "MAX_ORDERS", "SHOP_INTERVAL", "CENTER_INTERVAL", "SHED_ACCESS",
]
