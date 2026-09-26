"""The engine's own numbers, resolved for one turn.

`world/rules.py` transcribes the engine's constants and cites its lines. This
module decides which SOURCE answers when a caller asks for one of the numbers
the ENGINE ITSELF reads out of its run configuration
(`get(env.configuration, "shedCapacity", 100)` — kaggriculture.py:552-553,
:733-734, :866-867, :908):

  1. the observation's own `configuration`, when the harness put one there. It
     carries one only if the entry point asked for it, and ours takes `obs`
     alone (the one-argument entry the harness itself calls), so in a real
     episode this is normally absent; and
  2. otherwise the world's transcription of the engine's default.

ONE place does that resolution, so no caller needs to know which source won and
no caller carries a second copy of a default:

    terms = EngineTerms.from_obs(obs)
    terms.shed_capacity                 # int
    terms.hand_cost_mult                # int
    terms.get("farmHandCostMult", 1)    # the engine's own vocabulary

`get` speaks the ENGINE's key names (see `_KEYS`) because the callers on this
path arrived speaking them (`belief.market._get`, `belief.shed._get`) and the
alternative is a second name for every number.

Deliberately NOT here: the planner's own price of a hand
(`rules.HAND_COST_MULT = 0`, the owner's order — the planner is not given a
labour-cost model yet). `hand_cost_mult` is what a hire COSTS the farm; the
price the LP reasons with is that separate number. Wiring the two together is a
policy change, not a wiring one.

Also NOT here: the per-turn order cap (`rules.MAX_MARKET_ORDERS_PER_TURN`). The
engine never reads it from a configuration, so there is nothing to resolve — a
constant with one definition in `world/rules.py` is the whole answer.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import Any

from agent.world.rules import (CENTER_SELL_INTERVAL_TURNS, FARM_HAND_COST_MULT,
                               SHED_CAPACITY, SHOP_SELL_INTERVAL_TURNS,
                               SHOP_UNLOCK_INTERVAL_DAYS)

#: The engine's key name (and, for a caller that prefers ours, the constant's
#: name) → the field it answers with. Both spellings are read from the same
#: source dict: a run's configuration speaks the engine's camelCase.
_KEYS: dict[str, str] = {
    "farmHandCostMult": "hand_cost_mult",
    "FARM_HAND_COST_MULT": "hand_cost_mult",
    "shedCapacity": "shed_capacity",
    "SHED_CAPACITY": "shed_capacity",
    "townShopSellInterval": "shop_interval",
    "SHOP_SELL_INTERVAL_TURNS": "shop_interval",
    "townCenterSellInterval": "center_interval",
    "CENTER_SELL_INTERVAL_TURNS": "center_interval",
    "townShopUnlockInterval": "shop_unlock_interval",
    "SHOP_UNLOCK_INTERVAL_DAYS": "shop_unlock_interval",
}


@dataclass(frozen=True)
class EngineTerms:
    """The engine's configurable numbers, one resolved value per number.

    The defaults are the engine's own defaults as transcribed in
    `world/rules.py`; `from_obs` is the door the manager uses.
    """

    #: What the n-th hire of a day costs, per `fib(n)` (kaggriculture.py:101,
    #: overridable as `farmHandCostMult` at :552).
    hand_cost_mult: int = FARM_HAND_COST_MULT
    #: The shed's capacity in items, seeds excluded (:553, :867).
    shed_capacity: int = SHED_CAPACITY
    #: Turns between a shop instance's purchases (:733).
    shop_interval: int = SHOP_SELL_INTERVAL_TURNS
    #: Turns between the town centre's purchases (:734).
    center_interval: int = CENTER_SELL_INTERVAL_TURNS
    #: Days between shop unlocks (:867).
    shop_unlock_interval: int = SHOP_UNLOCK_INTERVAL_DAYS

    @classmethod
    def from_obs(cls, obs: Any, override: Mapping | None = None) -> "EngineTerms":
        """The terms this turn runs under.

        `override` is a caller's explicit say-so (a test, a probe, a harness
        that hands the configuration in beside the observation) and wins over
        the observation: the caller is stating what THIS run is, and the
        observation can only carry what the entry point was given.
        """
        source: dict = {}
        if isinstance(obs, Mapping):
            carried = obs.get("configuration")
            if isinstance(carried, Mapping):
                source.update(carried)
        if isinstance(override, Mapping):
            source.update(override)
        values: dict[str, int] = {}
        for key, attr in _KEYS.items():
            if key in source and source[key] is not None:
                values[attr] = int(source[key])
        return cls(**values)

    def get(self, key: str, default: Any = None) -> Any:
        """The engine's own lookup shape: `get(config, key, default)`.

        An unknown key returns the caller's default, exactly as the engine's
        `get` does — a caller asking for a number this object does not model is
        not an error, it is a caller reading a key we never transcribed.
        """
        attr = _KEYS.get(key)
        return default if attr is None else getattr(self, attr)

    def as_dict(self) -> dict[str, int]:
        """The engine's key names, for a caller that wants the whole set."""
        return {attr: getattr(self, attr) for attr in
                (f.name for f in fields(self))}
