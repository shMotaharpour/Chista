"""RL Manager: the neural macro-manager, wired into the day layer's own tools.

The division of labour this module is built around — each part answers ONE
question, and no question has two answers:

- the POLICY (`agent/artifact/rl_manager_weights.npz`, a pure NumPy forward
  pass, built offline by `offline_lab/build/train_jax_bc.py`) answers the
  macro questions: how many hands today, whether to buy the next quadrant,
  and which crops the farm wants. It never sees a tile.
- the TILE DP (`TileContractor`) answers the per-tile question: at today's
  prices, what does one day on THIS tile look like. One pricing sweep.
- the DAY LAYER (`agent/planner/day.py`, `agent/wsr/`) answers the routing
  question: which hand walks where, in what order, and does the day fit.
- the QUEUE REGULATOR (`agent/planner/market_regulator.py`) answers the
  market question: the ten slots, in the engine's own settlement order.

Two repairs sit between the policy and the DP, and both are FEASIBILITY
guards rather than policy: an animal the farm cannot house or feed, and a
plant with no days left to pay for itself. Each is named at its site.

The LP this replaces took 100.2 s for one season (seed 42, measured) and
scored 14,424 against `v3-agent`'s 162,770 in the same episode. This path's
own numbers are printed by `offline_lab/bench/bench_rl_manager.py`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from agent.belief.market import forecast
from agent.config import Config
from agent.planner import columns as C
from agent.planner import day as D
from agent.planner import market as K
from agent.planner import master as M
from agent.planner.inputs import GRAPH_PATH, dual_stand_in
from agent.planner.market_regulator import MarketQueueRegulator
from agent.tile_dp.chains import ENTITY_OF_CODE, chain_ops
from agent.tile_dp.contractor import HORIZON_DAYS, TileContractor
from agent.tile_dp.graph import TileGraph
from agent.world.model import ANIMALS, CROPS, PRODUCTS
from agent.world.rules import (BOARD_SIZE, LAND_PRICES, TURNS_PER_DAY,
                               hire_cost)
from agent.world.terms import EngineTerms

ENTITIES = list(CROPS) + list(ANIMALS)
ENTITY_INDEX = {name: i for i, name in enumerate(ENTITIES)}

WEIGHTS_PATH = Path(__file__).resolve().parents[1] / "artifact" / "rl_manager_weights.npz"

#: Tiles one worker can carry in a day — the capacity the tile selection
#: hands the day layer. A growing tile costs one care op a day (WATER, FEED),
#: so this is a load, not a count of tiles on the board.
TILES_PER_WORKER = 2.5

#: How many of each species the farm keeps. Measured from the vendored
#: `v3-agent`'s own day-0 orders (2 sheep, 2 cows); the uncapped version
#: bought 18 sheep in one day and had no coins left for their feed.
ANIMAL_CAP = 2

#: Melon's first yield is ten days out (F026) and carrot's is two, so a melon
#: planted after this day cannot pay for itself before the season ends.
MELON_LAST_PLANT_DAY = 18
#: The last day anything is planted at all.
PLANT_LAST_DAY = 26


def _affordable_hands(money: float, mult: int, cap: int = 8) -> int:
    """The largest pool today's purse can really hire (F039, F041).

    The wage is never the constraint — the ladder is fib and cheap (a fifth
    hand costs 5 coins) — but a HIRE behind a short purse is refused in
    silence, so the cap has to be the ladder the engine will actually charge:
    the largest n whose cumulative bill fits the purse.
    """
    bill, n = 0, 0
    while n < cap:
        nxt = bill + hire_cost(n, mult)
        if nxt > money:
            break
        bill, n = nxt, n + 1
    return n


def _next_land_price(farm: dict) -> int | None:
    """The engine's price for the next quadrant (F042), or None when all
    three are bought. The table is `rules.LAND_PRICES` — never a constant
    written here, because a second copy of an engine table is a second rule.
    """
    bought = max(0, len(farm.get("unlocked_quadrants") or ()) - 1)
    if bought >= len(LAND_PRICES):
        return None
    return int(LAND_PRICES[bought])


def _ranked_tiles(tiles: list, steps: np.ndarray) -> list[tuple[int, int, tuple, int]]:
    """Every owned tile as `(rank, shed distance, cell, plan index)`.

    `plan index` is the tile's position in `plans`, which is `decode_farm`'s
    own order (row-major, LOCKED skipped) — the same walk `_owned_states`
    flattens, so this index is what PAIRS a tile with its priced plan.
    Selection sorts for priority and must not break that pairing, which is
    why the index travels with the tile instead of being re-derived.

    The rank is the day's priority: a tile holding a crop or an animal comes
    first (its value dies without today's care — F001 unwatered plants become
    weeds, F017 unfed animals escape), then a weed (it occupies a tile until
    it is dug), then empty ground — the only rank that is new investment.
    """
    out: list[tuple[int, int, tuple, int]] = []
    index = 0
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            tile = tiles[r][c]
            if tile == "LOCKED":
                continue
            if isinstance(tile, dict) and (tile.get("crop") or tile.get("animal")):
                rank = 0
            elif isinstance(tile, dict) and tile.get("kind") == "WEED":
                rank = 1
            else:
                rank = 2
            out.append((rank, int(steps[r * BOARD_SIZE + c]), (r, c), index))
            index += 1
    out.sort(key=lambda e: (e[0], e[1]))
    return out


def _cash_crop(day: int) -> str | None:
    """The crop a tile plants when the plan's own choice is not feasible.

    Melon pays ten days after planting (F026) and carrot two, so the choice
    is the season's own calendar, not a preference.
    """
    if day <= MELON_LAST_PLANT_DAY:
        return "MELON"
    if day <= PLANT_LAST_DAY:
        return "CARROT"
    return None


def _repair(ops: tuple, entity: str | None, day: int,
            counts: dict) -> tuple[tuple, str | None]:
    """Feasibility guards on one tile's chain — the repair layer, named.

    Each guard replaces the chain with what the SAME tile may legally do
    instead; none of them is a second policy:

    - a structure the farm does not need becomes the cash-crop chain: a
      pasture with no animal holds no crop, and every extra one is a tile
      that can never pay (the measured farm turned 14 of its 25 tiles into
      empty pasture and never harvested again);
    - an animal of a species already at `ANIMAL_CAP` becomes the cash-crop
      chain;
    - a PLANT with no days left to pay for itself (F026) is dropped, and a
      tile with nothing left to do is skipped by the caller.
    """
    builds = "BUILD_PASTURE" in ops or "BUILD_COOP" in ops
    wants_animal = entity in counts["animals"]
    over_structure = builds and counts["structures"] >= ANIMAL_CAP
    over_animal = wants_animal and counts["animals"][entity] >= ANIMAL_CAP
    if over_structure or over_animal:
        crop = _cash_crop(day)
        if crop is None:
            return (), None
        return ("PLANT", "WATER"), crop
    if builds:
        counts["structures"] += 1
    if wants_animal:
        counts["animals"][entity] += 1
    if "PLANT" in ops and day > PLANT_LAST_DAY:
        ops = tuple(op for op in ops if op != "PLANT")
    return ops, entity


def extract_macro_state(obs: dict[str, Any], player_idx: int) -> np.ndarray:
    """The policy's own input: a normalized 40-dim vector from the observation.

    Every field is either the observation's or a log-scaled count; the crop
    counts read the observation's own keys (`crop`, `animal`), which the
    engine writes on a planted tile (probed, not guessed).
    """
    step = int(obs.get("step", 0))
    day = step // TURNS_PER_DAY
    farm = obs["farms"][player_idx]
    cash = float(farm.get("money", 0))
    hands = len(farm.get("hands", []))
    tiles = farm.get("tiles", [])

    owned_count = 0
    growing_counts = np.zeros(len(ENTITIES) + 1, dtype=np.float32)

    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            tile = tiles[r][c]
            if isinstance(tile, dict) or (isinstance(tile, str) and tile != "LOCKED"):
                owned_count += 1
                if isinstance(tile, dict):
                    plant = tile.get("crop") or tile.get("animal")
                    if plant in ENTITY_INDEX:
                        growing_counts[ENTITY_INDEX[plant]] += 1.0
                    else:
                        growing_counts[-1] += 1.0
                else:
                    growing_counts[-1] += 1.0

    market = obs.get("market", {})
    prices = np.zeros(len(PRODUCTS), dtype=np.float32)
    market_inv = np.zeros(len(PRODUCTS), dtype=np.float32)
    for i, prod in enumerate(PRODUCTS):
        item = market.get(prod, {})
        prices[i] = float(item.get("price", 10.0)) / 100.0
        market_inv[i] = float(item.get("inventory", 0.0)) / 50.0

    private = obs.get("private", {}) or {}
    shed = private.get("shed", {}) or {}
    shed_inv = np.zeros(len(PRODUCTS), dtype=np.float32)
    for i, prod in enumerate(PRODUCTS):
        shed_inv[i] = float(shed.get(prod, 0)) / 50.0

    head = [
        float(day) / 30.0,
        np.log1p(max(0.0, cash)) / 12.0,
        float(owned_count) / 25.0,
        float(hands) / 8.0,
    ]
    return np.concatenate([
        np.array(head, dtype=np.float32),
        growing_counts / max(1.0, float(owned_count)),
        prices,
        market_inv,
        shed_inv,
    ])


class NumpyPolicy:
    """The trained MLP as a pure NumPy forward pass.

    Weights are the exported `npz` (`train_jax_bc.py` verifies the NumPy pass
    against JAX to < 1e-5 before writing it), so a turn needs no JAX and no
    framework import.
    """

    def __init__(self, path: Path | str):
        data = np.load(path)
        self.w1, self.b1 = data["w1"], data["b1"]
        self.w2, self.b2 = data["w2"], data["b2"]
        self.w_hands, self.b_hands = data["w_hands"], data["b_hands"]
        self.w_land, self.b_land = data["w_land"], data["b_land"]
        self.w_crops, self.b_crops = data["w_crops"], data["b_crops"]

    def predict(self, x: np.ndarray) -> tuple[int, bool, np.ndarray]:
        """`(hands, buy_land, crop distribution)` for one state vector."""
        h1 = np.maximum(0.0, x @ self.w1 + self.b1)
        h2 = np.maximum(0.0, h1 @ self.w2 + self.b2)

        hands_logits = h2 @ self.w_hands + self.b_hands
        land_logit = float((h2 @ self.w_land + self.b_land).squeeze())
        crops_logits = h2 @ self.w_crops + self.b_crops

        exps = np.exp(crops_logits - np.max(crops_logits))
        crop_probs = exps / np.sum(exps)
        return int(np.argmax(hands_logits)), bool(land_logit > 0.0), crop_probs


class RLManager:
    """`observe` at hour 0, `step` every later turn, `best` whenever asked."""

    def __init__(self, config: Config | None = None,
                 weights_path: Path | str | None = None) -> None:
        self.cfg = config or Config.load()
        self.graph = TileGraph.load(GRAPH_PATH)
        self.keys = frozenset(self.graph.key_index)
        self.contractor = TileContractor(self.graph, days=HORIZON_DAYS)
        self.steps = C.shed_distance()
        self.terms = EngineTerms()
        self.regulator = MarketQueueRegulator()
        self.plan: dict = {"units": [[["PASS"]] * TURNS_PER_DAY], "market": []}

        self.policy = NumpyPolicy(weights_path or WEIGHTS_PATH)
        self.certified = True
        self.pool: list = []
        #: Per-observe numbers for the benches (`offline_lab/bench/diag_rl_manager.py`):
        #: what the policy asked for and what the day layer made of it.
        self.debug: dict = {}

    # -- the day ----------------------------------------------------------
    def _price_day(self, obs, days: int, day: int, pred_hands: int,
                   crop_probs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """The `(p, w)` the tile DP is priced at for this day.

        `p` is belief's own forecast path (F035's rising season), tilted by
        the policy's crop basket — the one place the imitation signal reaches
        the DP. `w` prices the farm's own labour: cheap early, and priced UP
        over the last two days so the DP prefers harvesting what is ripe over
        starting what cannot finish.
        """
        p_mkt, w_stand = dual_stand_in(obs, cfg=self.cfg)
        p = np.array(p_mkt[:days], dtype=np.float64)
        w = np.array(w_stand[:days], dtype=np.float64)
        try:
            fc = forecast(obs, days=days, config=self.terms)
            prices = np.array(fc.prices, dtype=np.float64)
            if (prices.ndim == 2 and prices.shape[0] >= days
                    and prices.shape[1] >= len(PRODUCTS)):
                p[:days, :len(PRODUCTS)] = prices[:days, :len(PRODUCTS)]
        except Exception:                     # noqa: BLE001 - belief is optional
            pass                              # the stand-in path is the fallback
        for gi, prod in enumerate(PRODUCTS):
            if prod in ENTITY_INDEX and gi < p.shape[1]:
                p[:, gi] *= 1.0 + float(crop_probs[ENTITY_INDEX[prod]])
        if day >= 29:
            w[:, 0] = 50.0
        elif day >= 28:
            w[:, 0] = 10.0
        else:
            w[:, 0] = max(1.0, float(pred_hands) * 1.5)
        return p, w

    def observe(self, obs: dict[str, Any], config=None) -> None:
        """Plan the day: the policy's answer, the DP's chains, wsr's route."""
        self.terms = EngineTerms.from_obs(obs, config)
        player_idx = int(obs.get("player", 0))
        farm = obs["farms"][player_idx]
        tiles = farm.get("tiles", [])
        money = float(farm.get("money", 0.0))
        day = int(obs.get("step", 0)) // TURNS_PER_DAY

        # 1. the policy's macro answer, capped by the purse's own hire ladder
        pred_hands, buy_land, crop_probs = self.policy.predict(
            extract_macro_state(obs, player_idx))
        pred_hands = min(int(pred_hands),
                         _affordable_hands(money, self.terms.hand_cost_mult))

        # 2. the horizon and the prices the DP reads
        days = M.season_horizon(obs)
        if int(self.contractor.days) != days:
            self.contractor = TileContractor(self.graph, days=days)
        p, w = self._price_day(obs, days, day, pred_hands, crop_probs)
        owned_states = M._owned_states(object(), obs)
        priced = self.contractor.price_many(p, w, {0: owned_states})[0]

        # 3. the day's work: survival first, then weeds, then new planting.
        #    The animal counts start from the BOARD's own animals, so the cap
        #    is about the farm, not about today's orders.
        animals = {"COW": 0, "SHEEP": 0, "GOOSE": 0}
        structures = 0
        for row in tiles:
            for tile in row:
                if not isinstance(tile, dict):
                    continue
                if tile.get("kind") in ("PASTURE", "COOP"):
                    structures += 1
                species = tile.get("animal")
                if species in animals:
                    animals[species] += 1
        counts = {"structures": structures, "animals": animals}

        capacity = max(3, int((1 + pred_hands) * TILES_PER_WORKER))
        chains: list = []
        for rank, _dist, cell, index in _ranked_tiles(tiles, self.steps):
            if len(chains) >= capacity:
                break
            if index >= len(priced.plans):
                continue                      # no priced plan for this tile
            if rank == 1:                     # a weed: dig it, buys nothing
                chains.append((cell, ("DIG",), None))
                continue
            ops = chain_ops(int(priced.plans[index][0][2]))
            entity = ENTITY_OF_CODE.get(int(priced.per_day_entity[index, 0]))
            ops, entity = _repair(ops, entity, day, counts)
            if ops:
                chains.append((cell, ops, entity))

        # 4. does the day fit? wsr answers; the tail (least urgent) is dropped
        fitted = D.fit(chains, hands=pred_hands,
                       available=D.availability(obs, chains),
                       hours_committed=0.0)
        while not fitted.complete and len(chains) > 1:
            chains.pop()
            fitted = D.fit(chains, hands=pred_hands,
                           available=D.availability(obs, chains),
                           hours_committed=0.0)

        # 5. the units, compiled from the same chains the search priced
        day_plan = D.DayPlan(master=object(), choices=[], mixes={},
                             day=fitted, rounds=1, overhead=1.0,
                             hands=fitted.pool, net=0.0, solves=1)
        compiled = D.compile(day_plan, obs, hands=fitted.pool,
                             terms=self.terms, model=None, activity=None,
                             forecast_obj=None)

        # 6. the market: the shed's own goods are sold, the chains' needs are
        #    bought, and the regulator owns the ten slots
        private = obs.get("private", {}) or {}
        shed = dict(private.get("shed", {}) or {})
        sells_by_turn = [[list(o) for o in (row or [])
                          if o and str(o[0]) == "SELL"]
                         for row in (compiled.get("market") or [[]])]
        shed_sells = [["SELL", good, int(n)] for good, n in shed.items()
                      if good in PRODUCTS and n > 0]
        if shed_sells:
            if not sells_by_turn:
                sells_by_turn = [[]]
            sells_by_turn[0] = shed_sells + sells_by_turn[0]

        buys, _bill = K.buy_orders(
            fitted.chains, dict(private.get("seeds", {}) or {}), shed,
            (obs.get("market", {}) or {}).get("prices", {}) or {})
        if day >= 29:
            buys = []                         # the last day buys nothing

        land_price = _next_land_price(farm)
        can_buy_land = bool(buy_land and land_price is not None
                            and money >= land_price and day <= 22)

        market_rows = self.regulator.schedule_day_orders(
            sells_by_turn=sells_by_turn, hires=fitted.pool,
            buy_land=can_buy_land, buys=buys)

        self.debug = {
            "day": day, "pred_hands": pred_hands, "buy_land": bool(buy_land),
            "can_buy_land": can_buy_land, "n_chains": len(chains),
            "complete": bool(fitted.complete), "pool": int(fitted.pool),
            "tasks": int(fitted.tasks), "placed": int(fitted.placed),
            "chains": list(chains),
            "turn0_orders": [list(o) for o in market_rows[0]],
        }
        self.plan = {
            "units": compiled.get("units", [[["PASS"]] * TURNS_PER_DAY]),
            "market": market_rows,
        }

    def step(self, obs: dict[str, Any]) -> None:
        """Nothing to do between days.

        The plan the day was compiled with is sliced per hour by
        `dispatch_plan`; the LP manager's pool work has no counterpart here
        because the policy answers once, at hour 0.
        """

    def best(self) -> dict[str, Any]:
        """The day's plan; idle before the first `observe`."""
        return self.plan