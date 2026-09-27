"""The engine's own numbers, resolved for one turn.

`world/rules.py` transcribes the engine's constants and cites its lines; this
module is the ONE place that decides which SOURCE answers when a caller asks for
one of the numbers the engine itself reads out of a run's configuration:

  1. the run's own configuration, which `kaggle_environments` hands to the agent
     as the SECOND positional argument (`agent.py:151-153`, `:171-172`: the
     framework builds `[observation, configuration]` and truncates the call to
     the callable's `co_argcount`, so a two-argument entry receives it and a
     one-argument entry never does). Measured on this machine: a 1-arg callable
     got the observation alone; a 2-arg callable got all 15 keys, with the run's
     overrides applied (`/tmp/chista_probe/agent_signature.py`);
  2. otherwise the world's transcription of the engine's own default, which the
     guard `tests/test_engine_terms.py` checks against the INSTALLED
     `kaggriculture.json` key by key.

So no caller has to know which source won, and no caller carries a second copy
of a default:

    terms = EngineTerms.from_obs(obs, config)
    terms.shed_capacity          # int
    terms.hand_cost_mult         # int, the engine's `farmHandCostMult`
    terms.get("farmHandCostMult", 1)   # the engine's own vocabulary

`get` speaks the ENGINE's key names (see `_KEYS`) because the callers on this
path arrived speaking them (`belief.market._get`, `belief.shed._get`) and the
alternative is a second name for every number.

`market_params` and `seed` are carried the same way but are not numbers: the
first is the run's sparse price-curve override (a mapping, or None for the
engine's own defaults) and the second is the resolved episode seed the engine
publishes on `env.info` (None when the run did not set one).

What a hand COSTS has one reference too: `hand_cost_mult` here, read by every
pricing site through `rules.hire_cost` — so a run that sets `farmHandCostMult`
moves the hire bill, the LP's hour price and the farmer's hour floor at once.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from typing import Any

from agent.world.rules import (ACT_TIMEOUT_S, BOARD_SIZE, CENTER_SELL_INTERVAL_TURNS,
                               EPISODE_STEPS, FARM_HAND_COST_MULT,
                               MAX_MARKET_ORDERS_PER_TURN, SHED_CAPACITY,
                               SHOP_SELL_INTERVAL_TURNS, SHOP_UNLOCK_INTERVAL_DAYS,
                               STARTING_MONEY, TURNS_PER_DAY, WEED_SPAWN_CHANCE)

#: The engine's key name (and, for a caller that prefers ours, the constant's
#: name) → the field it answers with. Both spellings are read from the same
#: source dict: a run's configuration speaks the engine's camelCase, and the
#: framework adds one key of its own (`__raw_path__`, `agent.py:150`), so a
#: reader must ignore what it does not model.
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
    "boardSize": "board_size",
    "BOARD_SIZE": "board_size",
    "maxMarketOrdersPerTurn": "max_orders_per_turn",
    "MAX_MARKET_ORDERS_PER_TURN": "max_orders_per_turn",
    "turnsPerDay": "turns_per_day",
    "TURNS_PER_DAY": "turns_per_day",
    "episodeSteps": "episode_steps",
    "EPISODE_STEPS": "episode_steps",
    "actTimeout": "act_timeout",
    "ACT_TIMEOUT_S": "act_timeout",
    "startingMoney": "starting_money",
    "STARTING_MONEY": "starting_money",
    "weedSpawnChance": "weed_spawn_chance",
    "WEED_SPAWN_CHANCE": "weed_spawn_chance",
    "seed": "seed",
    "marketParams": "market_params",
}

#: How a value out of a run's configuration becomes the field's type. `seed` and
#: `market_params` keep their own shapes (None included).
_CASTERS: dict[str, Any] = {"weed_spawn_chance": float, "seed": int,
                            "market_params": dict}


@dataclass(frozen=True)
class EngineTerms:
    """The engine's configurable numbers, one resolved value per number.

    Every default is the engine's own default as transcribed in
    `world/rules.py`; `from_obs` is the door the manager uses, and the guard in
    `tests/test_engine_terms.py` pins each one against the installed
    `kaggriculture.json`.
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
    #: Tiles a side of each player's square farm (:550).
    board_size: int = BOARD_SIZE
    #: Market orders the engine settles per player per turn (:551).
    max_orders_per_turn: int = MAX_MARKET_ORDERS_PER_TURN
    #: Turns in one day (:864, :906).
    turns_per_day: int = TURNS_PER_DAY
    #: The season's steps (:960 reads `cfg.episodeSteps`).
    episode_steps: int = EPISODE_STEPS
    #: Seconds a turn is allowed before the overage bank is charged.
    act_timeout: int = ACT_TIMEOUT_S
    #: Money each player starts with (:252).
    starting_money: int = STARTING_MONEY
    #: Per-tile weed probability at the nightly refresh (:865).
    weed_spawn_chance: float = WEED_SPAWN_CHANCE
    #: The resolved episode seed, or None when the run set none. The engine keeps
    #: it off the observation and publishes it on `env.info` (config description
    #: of `seed`), so this is what a caller can see of it.
    seed: int | None = None
    #: The run's sparse price-curve override, or None for the engine's defaults
    #: (`marketParams`, resolved in the engine by `_resolve_market_params`).
    market_params: Mapping | None = field(default=None)

    @classmethod
    def from_obs(cls, obs: Any, config: Mapping | None = None) -> "EngineTerms":
        """The terms this turn runs under.

        `config` is the run's configuration — the second argument the harness
        hands a two-argument entry. The observation is read too, for a harness
        that puts the configuration there instead; `config` wins when both are
        present, because it is the framework's own channel.
        """
        source: dict = {}
        if isinstance(obs, Mapping):
            carried = obs.get("configuration")
            if isinstance(carried, Mapping):
                source.update(carried)
        if isinstance(config, Mapping):
            source.update(config)
        values: dict[str, Any] = {}
        for key, attr in _KEYS.items():
            if key in source and source[key] is not None:
                caster = _CASTERS.get(attr)
                values[attr] = source[key] if caster is None else caster(source[key])
        return cls(**values)

    def get(self, key: str, default: Any = None) -> Any:
        """The engine's own lookup shape: `get(config, key, default)`.

        An unknown key returns the caller's default, exactly as the engine's
        `get` does — a caller asking for a number this object does not model is
        not an error, it is a caller reading a key we never transcribed.
        """
        attr = _KEYS.get(key)
        return default if attr is None else getattr(self, attr)

    def as_dict(self) -> dict[str, Any]:
        """The engine's key names, for a caller that wants the whole set."""
        return {attr: getattr(self, attr) for attr in
                (f.name for f in fields(self))}
