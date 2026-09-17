"""Replanner: the day's plan, priced by the contractor (issue #11 §6).

The first heavy rung of the fallback ladder, and the socket `agent/runtime.py`
already had (`_rung_plan` dispatches `self.plan` when it is not None).

It runs **once per day**, at hour 0, and the spine dispatches the stored plan for
the remaining 23 hours. The graph is day-invariant, so within a day nothing the
DP sees changes except our own execution — replanning every turn would burn the
budget for nothing.

It **polls the published deadline** between steps and raises `TimeoutError`
mid-work. The ladder's between-rung gate can only see a rung that already spent
the turn; `self._deadline` exists so the rung itself can bail (the drill in
`tests/test_agent_runtime.py::test_deadline_gates_the_ladder` pins the contract,
this module is the real thing it was pinning).

## Two stand-ins, both named, both somebody else's issue

**The duals (#12).** The Walrasian master now sits on top of the stand-in
(`planner/master.py`, on by default inside this rung; `CHISTA_MASTER=0`
falls back to the flat quotes): the stand-in quotes are the FLOOR for the
purchasable inputs and the exogenous product prices, and the master raises
the internal prices only where the tiles' shared stocks actually bind. The
measured table below predates the master; the F047 execution gap it reports
is unchanged (the chains' purchases still ride on #14).

**The secretary (#14).** The contractor prices ONE TILE at a time; turning tile
chains into per-unit op lists — movement, pickups, market batching — is the
secretary's job and is not written. So each unit works the tile it already
stands on, and only the chain's own worker ops are dispatched: the purchases and
carries the chain assumes are missing, and the engine refuses an op whose input
the unit does not carry, **in silence** (F047).

That is why the rung is **off by default** (`CHISTA_REPLAN=1` turns it on), and
the reason is F047 rather than a hunch: shipping ops the engine ignores is the
mistake class the project named, and every op in a chain whose input the unit
does not carry is exactly that.

Measured, so nobody has to guess what "off" is worth (official path, against
`random`, seeds 0-2, 720 steps, rung toggled by the env switch):

| | final money |
|---|---|
| rung off (greedy) | 2,840 |
| rung on | 3,000 |

Both numbers are bad, and neither says the rung works. The rung's side is the
**starting money untouched** (F038: 3,000) — it dispatches ops like
`BUILD_PASTURE, PLACE SHEEP` whose animal nobody bought, the engine refuses them
in silence, and it spends nothing, which scores above a greedy policy that buys
wheat seed and never sells the harvest. So: greedy is a weak baseline, the rung
currently does nothing useful, and the honest place for it is behind a switch
until #14 can carry the inputs. The arena (#18/#20) is the instrument that will
settle it, not this table.

**Arena measurement, 2026-09-16 (after the master landed, #12):** the rung
played full seasons against pool agents (paired seeds, official
evaluate harness) and finished at **exactly 3,000 every time — starting
money untouched** (`BUILD_PASTURE` day 0 then `DIG` days 1-29; every
purchase the chains assume is refused in silence because nothing buys,
F047). Against a pool agent that also banks 3,000, every game is a tie
(+160/-160 readings in the arena are the *opponent* beating a third
seat, not us). Conclusion unchanged and now measured over full seasons:
the rung is inert until #14 carries the purchases — the master (#12)
prices correctly, but pricing cannot fix ops the engine silently
refuses.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from kaggle_environments.envs.kaggriculture import kaggriculture as K

from agent.obs import LOCKED_KEY, WorldView, _nearest_modelled, decode_world
from tile_dp.chains import N_RESOURCE, RESOURCE_ID, chain_ops, entity_of_code
from secretary.routing import plan_day
from tile_dp.contractor import HORIZON_DAYS, TileContractor
from tile_dp.graph import TileGraph

from pathlib import Path

GRAPH_PATH = (Path(__file__).resolve().parents[1] / "tile_dp" / "models"
              / "graph_tile_lifecycle.npz")

# The engine's own hire cost is imported, never transcribed (R002): the n-th
# hire of a day costs `_hire_cost(n)`. A hand hired at hour 0 first acts at
# hour 1 (F040), so a hand buys 23 hours and that is what an hour costs.
HOURS_PER_HAND = 23

# Hands are cleared every night (F039), so this is a daily decision. Five cost
# 1+1+2+3+5 = 12 coins and a hand has to walk to its tile, so beyond five the
# day's travel eats the work it buys.
MAX_HANDS = 5

# Ops that name the entity a chain constructs, and therefore cannot be
# dispatched without one.
_ENTITY_OPS = frozenset(("PLANT", "BUILD", "PLACE", "PLACE_ANIMAL"))


def load_contractor() -> TileContractor:
    """The shipped graph, cast once per process — never rebuilt at runtime."""
    return TileContractor(TileGraph.load(GRAPH_PATH))


def dual_stand_in(obs: Any) -> tuple[np.ndarray, np.ndarray]:
    """The master's duals (#12), stood in for by the engine's own quotes.

    Every component has a source (R005): products are the observation's own
    market prices, seeds and animals are the engine tables, and the wage is the
    engine's marginal hire over the hours a hand actually works.
    """
    market = obs.get("market", {}) if isinstance(obs, dict) else {}
    prices = market.get("prices", {}) or {}
    farms = obs.get("farms", []) if isinstance(obs, dict) else []
    player = int(obs.get("player", 0)) if isinstance(obs, dict) else 0
    farm = farms[player] if len(farms) > player else {}

    p = np.zeros((HORIZON_DAYS, N_RESOURCE))
    for product in K.PRODUCTS:
        if product in RESOURCE_ID:
            p[:, RESOURCE_ID[product]] = float(prices.get(product, 0.0))

    w = np.zeros((HORIZON_DAYS, N_RESOURCE))
    for crop, spec in K.CROPS.items():
        w[:, RESOURCE_ID[f"SEED_{crop}"]] = float(spec["seed"])
    for species, spec in K.ANIMALS.items():
        w[:, RESOURCE_ID[f"ANIMAL_{species}"]] = float(spec["cost"])
    # FEED eats wheat and FERTILIZE eats fertilizer: what an animal's feed and a
    # fertilizer dose cost is the price they are bought back at.
    w[:, RESOURCE_ID["WHEAT"]] = float(prices.get("WHEAT", 0.0))
    w[:, RESOURCE_ID["FERTILIZER"]] = float(prices.get("FERTILIZER", 0.0))
    w[:, RESOURCE_ID["LABOR_HOURS"]] = (
        float(K._hire_cost(int(farm.get("hires_today", 0)))) / HOURS_PER_HAND)
    return p, w


def unit_positions(view: WorldView) -> list[tuple[int, int]]:
    """The farmer first, then the hands in `hands` order (F030)."""
    positions = [(int(view.me.farmer[0]), int(view.me.farmer[1]))]
    positions += [(int(x), int(y)) for x, y in view.me.hands]
    return positions


def unit_state_ids(view: WorldView, graph: TileGraph) -> list[int | None]:
    """The graph state under each unit, or None where there is nothing to price.

    A LOCKED tile (F042) has no state, and a key the graph does not model falls
    back to its nearest neighbour exactly as `agent/obs.py` does for the
    classes — the same rule, so the pricing and the counts cannot disagree.
    """
    keys = view.me.keys
    known = frozenset(graph.key_index)
    ids: list[int | None] = []
    for x, y in unit_positions(view):
        key = int(keys[y][x])
        if key == LOCKED_KEY:
            ids.append(None)
            continue
        state_id = graph.key_index.get(key)
        if state_id is None:
            state_id = _nearest_modelled(key, known)
        ids.append(int(state_id))
    return ids


def owned_tiles(view: WorldView, graph: TileGraph) -> list[tuple[tuple[int, int], int]]:
    """Every non-LOCKED tile we own, with the graph state standing on it.

    The old rung priced only the tiles the units happened to stand on, so the
    rest of the farm was never planned at all (one column per unit). The board is
    position-invariant, so pricing every owned tile costs one more column.
    """
    keys = view.me.keys
    known = frozenset(graph.key_index)
    out: list[tuple[tuple[int, int], int]] = []
    for y in range(keys.shape[0]):
        for x in range(keys.shape[1]):
            key = int(keys[y][x])
            if key == LOCKED_KEY:
                continue
            state_id = graph.key_index.get(key)
            if state_id is None:
                state_id = _nearest_modelled(key, known)
            out.append(((x, y), int(state_id)))
    return out


def hire_count(view: WorldView, tiles: list, money: float) -> int:
    """How many hands to hire today (F039: they are cleared every night).

    The ladder is cheap (1, 1, 2, 3, 5 for five hands) but not free, and the
    seeds are the first claim on the purse, so the count is what fits after them
    and is bounded by the tile-work actually waiting.
    """
    spare = max(0, len(tiles) - (1 + len(view.me.hands)))
    want = min(MAX_HANDS, spare)
    seed_bill = sum(float(K.CROPS[entity]["seed"]) for _xy, ops, entity in tiles
                    if "PLANT" in ops and entity in K.CROPS)
    purse = float(money) - seed_bill
    hired = 0
    while hired < want:
        cost = float(K._hire_cost(int(view.me.hires_today) + hired))
        if cost > purse:
            break
        purse -= cost
        hired += 1
    return hired


def replan_day(runtime, obs, graph: TileGraph | None = None,
               contractor: TileContractor | None = None) -> dict:
    """One day's plan: decode, price the whole board, compile the day. Polls at each step.

    `graph` / `contractor` are injectable for tests; in the runtime they are
    loaded once and cached on the runtime object.
    """
    deadline = getattr(runtime, "_deadline", None)
    _poll(deadline)
    if graph is None or contractor is None:
        cached = getattr(runtime, "_replan_resources", None)
        if cached is None:
            loaded = load_contractor()
            cached = (loaded.graph, loaded)
            runtime._replan_resources = cached
        graph = graph or cached[0]
        contractor = contractor or cached[1]
    _poll(deadline)
    # `at_day_start` asserts hour 0: a mid-day decode fed the DP the wrong state
    # once already (the pre-v15 rebuild bug, #10).
    view = decode_world(obs, at_day_start=True,
                        graph_keys=frozenset(graph.key_index))
    _poll(deadline)
    priced = owned_tiles(view, graph)
    owned = [state_id for _xy, state_id in priced]
    # #12: the Walrasian master sets the internal prices. The stand-in
    # quotes stay as the FLOOR (the farm can buy any input at the quote)
    # and the exogenous product prices; the master's tâtonnement raises
    # the internal prices only where the tiles' shared stocks bind. The
    # warm start rides on the runtime (yesterday's published w);
    # `CHISTA_MASTER=0` falls back to the flat stand-in quotes.
    import os
    if os.environ.get("CHISTA_MASTER", "1") == "1":
        from planner.master import equilibrate, supply_from_obs
        master = equilibrate(runtime, obs, contractor,
                             supply_from_obs(obs),
                             w_warm=getattr(runtime, "_master_w", None),
                             owned=owned, poll=lambda: _poll(deadline))
        runtime._master_w = master.w            # tomorrow's warm start
        runtime._master_last = master           # the plan record's evidence
        p, w = master.p, master.w
    else:
        p, w = dual_stand_in(obs)
    _poll(deadline)
    # the day's plan: each tile's best response at the PUBLISHED prices
    # (the master's λ mix is fractional; per-tile rounding is #13)
    board = contractor.price(p, w, owned)
    _poll(deadline)
    tiles: list[tuple[tuple[int, int], tuple[str, ...], str | None]] = []
    yields: list[dict[str, int]] = []
    for column, (xy, _state_id) in enumerate(priced):
        _day, _state, chain_id = board.plans[column][0]
        entity = entity_of_code(int(board.per_day_entity[column, 0]))
        tiles.append((xy, chain_ops(chain_id), entity))
        # what a HARVEST on this tile hands over, read off the observation: the
        # engine's own yield_units (the tile is the one the chain works).
        raw = obs["farms"][int(obs["player"])]["tiles"][xy[1]][xy[0]]
        n = int(raw.get("yield_units", 0)) if isinstance(raw, dict) else 0
        crop = raw.get("crop") if isinstance(raw, dict) else None
        yields.append({str(crop): n} if n > 0 and crop else {})
    _poll(deadline)
    hires = hire_count(view, tiles, float(view.me.money))
    # The sell side (#15): the shed guard needs to know what today will bring in.
    # The compiler knows exactly (one unit per harvest chain is a lower bound for
    # the guard; the routes themselves publish the real arrivals to the queue).
    from secretary.inventory import market_queue
    harvest_estimate = sum(sum(y.values()) for y in yields)
    sells = market_queue(obs, harvest_expected=harvest_estimate)
    _poll(deadline)
    plan = plan_day(tiles, unit_positions(view), new_hands=hires,
                    bags=view.private.inventories, shed=view.private.shed,
                    money=float(view.me.money), hires_today=view.me.hires_today,
                    sells=sells, yields=yields,
                    values=[float(v) for v in board.tile_values],
                    prices=view.market_prices)
    runtime._day_plan = plan                    # the record's evidence
    return plan.as_plan()


def _poll(deadline) -> None:
    """The rung's own bail: the ladder only gates BETWEEN rungs."""
    if deadline is not None and deadline.expired():
        raise TimeoutError("replanner over budget: the plan rung bailed")


def enable(runtime) -> None:
    """Turn the rung on for a runtime object (the tests' handle on it)."""
    runtime.replanner = replan_day
