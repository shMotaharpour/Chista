"""The measurements behind `belief/`'s docstrings, reproducible from a checkout.

Run from the repository root (dev mode, seed 11, `world.fast_sim`):

    .venv/bin/python -m bench.bench_market_analyzer [days]

Six readings, each against ground truth rather than against the analyzer's own
beliefs:

  1. price parity — the vectorised price function against `K.market_price`;
  2. phase 1, isolation — our seat trades nothing, so the inventory residual is
     the rival's channel alone and can be compared with their *committed orders*;
  3. phase 1, both seats trading — the same comparison with our own flow present,
     which measures how well the tracker replicates the unit phase before the
     market;
  4. the demand forecast, and the fact that already-open shops are facts;
  5. the hidden intra-turn order, inferred from the rival's realised average
     price (public through their money balance);
  6. both solvers, and the season LP with and without its absorption row.

Timings are printed against the F046 budget at the end.
"""

from __future__ import annotations

import sys
import time

import numpy as np

from kaggle_environments.envs.kaggriculture import kaggriculture as K
from world.fast_sim import FastSim
from world.prices import price_of, price_vec
from world.vocabulary import G_IX, GOODS, MAX_ORDERS, SHED_ACCESS, SHED_CAP

from belief.opponent import OpponentModel, drain_forecast, infer_rival_slot
from belief.solvers import (default_schedules, maximin_mixed_lp,
                            maximin_mixed_slsqp, season_plan_maximin,
                            slot_game_matrix, round_tiles)
from belief.tracker import MarketTracker, committed_flow

NW_TARGETS = [(1, 1), (2, 2), (3, 1), (1, 3), (3, 3), (2, 4), (4, 2)]


def _move_toward(pos, goal):
    x, y = pos
    gx, gy = goal
    if x < gx:
        return "EAST"
    if x > gx:
        return "WEST"
    if y < gy:
        return "SOUTH"
    if y > gy:
        return "NORTH"
    return "PASS"


def scripted(shift: int, sell_every: int, buy_product: bool, market_silent: bool = False,
             sell_on_center: bool = False):
    """A small economic policy: plant, water, harvest, carry to the ring, drop, sell.

    Two engine rules shape it and both refuse in silence (F047): PLANT requests are
    dropped *en masse* when the turn's demand exceeds the seed stock, and HARVEST
    before `first_yield_day` does nothing.

    `sell_on_center` makes the rival sell on the turns the town centre consumes
    (every 24 steps). Without it a guard against a missing drain term cannot see
    the case: a drain-only turn pushes the residual negative, the `max(net, 0)`
    clamp absorbs it, and the error stays hidden.
    """
    def policy(obs):
        step = int(obs["step"])
        farm = obs["farms"][int(obs["player"])]
        tiles = farm["tiles"]
        shed = dict(obs["private"]["shed"])
        seeds = dict(obs["private"]["seeds"])
        bags = obs["private"]["inventories"]

        market = []
        if not market_silent:
            sell_turn = ((step + shift) % sell_every == 0
                         or (sell_on_center and step % 24 == 0))
            if sell_turn:
                for good, n in sorted(shed.items()):
                    if n > 0 and good != "FERTILIZER":
                        market.append(["SELL", good, min(n, 3)])
            if seeds.get("WHEAT", 0) < 2 and farm["money"] > 40:
                market.append(["BUY_SEED", "WHEAT", 2])
            if buy_product and (step + shift) % 17 == 0 and sum(shed.values()) < SHED_CAP - 10:
                market.append(["BUY_PRODUCT", "WHEAT", 2])
            if step % 24 == 0 and len(farm["hands"]) < 2 and farm["money"] > 200:
                market.append(["HIRE"])

        units = [list(farm["farmer"])] + [list(p) for p in farm["hands"]]
        acts = []
        plant_budget = int(seeds.get("WHEAT", 0))
        for idx, pos in enumerate(units):
            bag = bags[idx] if idx < len(bags) else {}
            x, y = int(pos[0]), int(pos[1])
            target = NW_TARGETS[idx % len(NW_TARGETS)]
            carried = sum(v for k, v in bag.items() if k != "FERTILIZER")
            if carried >= 1 and (x, y) in SHED_ACCESS:
                acts.append(["DROP"])
                continue
            if carried >= 1:
                acts.append([_move_toward((x, y), (4, 4))])
                continue
            tile = tiles[y][x]
            if tile == "LOCKED" or (x, y) != target:
                acts.append([_move_toward((x, y), target)])
                continue
            if tile is None:
                if plant_budget > 0:
                    acts.append(["PLANT", "WHEAT"])
                    plant_budget -= 1
                else:
                    acts.append(["PASS"])
            elif isinstance(tile, dict) and tile.get("kind") == "WEED":
                acts.append(["DIG"])
            elif isinstance(tile, dict) and tile.get("kind") == "PLANT":
                age = int(obs["day"]) - int(tile.get("planted_day", 0))
                ripe = (int(tile.get("yield_units", 0)) > 0
                        and age >= int(K.CROPS[tile["crop"]]["first_yield_day"]))
                acts.append(["HARVEST"] if ripe
                            else (["PASS"] if tile.get("watered_today") else ["WATER"]))
            else:
                acts.append(["PASS"])
        return {"farmer": acts[0], "hands": acts[1:], "market": market}
    return policy


def center_drain_probe(seed: int = 5, lot: int = 5) -> dict:
    """Build the one turn a missing drain term hides in, and read the residual.

    The town centre consumes at every step divisible by 24, but a scripted rival
    does not reliably hold goods in the shed on exactly those turns, so the
    situation is constructed: the rival's shed is stocked before a centre turn and
    they sell into it. Returns the drain of that turn, the residual, and the truth.

    This exists because the first residual guard passed with the drain term deleted:
    on a drain-only turn the residual goes negative and `max(net, 0)` clamps the
    error away, so the guard never saw the term it claimed to cover.
    """
    sim = FastSim(configuration={"seed": seed, "episodeSteps": 720}, validate="dev")
    tracker = MarketTracker(player=0)
    passive = {"farmer": ["PASS"], "hands": [], "market": []}
    while int(sim.observations()[0]["step"]) < 24:          # stop ON the centre turn
        obs = sim.observations()[0]
        tracker.observe(obs)
        tracker.note_our_action(passive, obs)
        sim.step([passive, passive])

    obs = sim.observations()[0]
    sim.state[1].observation.private["shed"]["MILK"] = lot
    sim.state[1].observation.private["shed"]["WHEAT"] = lot
    rival_action = {"farmer": ["PASS"], "hands": [],
                    "market": [["SELL", "MILK", lot], ["SELL", "WHEAT", lot]]}
    tracker.observe(obs)
    tracker.note_our_action(passive, obs)
    sim.step([passive, rival_action])

    rec = tracker.observe(sim.observations()[0])
    # the record's drain and residual both belong to the turn just played, which is
    # the observation's step minus one: that is the centre turn the probe aims at.
    drained_step = int(rec.step) - 1
    return dict(step=rec.step, drained_step=drained_step, drain=rec.drain,
                residual=rec.rival_sales, truth_units=lot,
                at_center=bool(drained_step % 24 == 0))


def run_episode(days: int, seat0_silent: bool, seed: int = 11) -> dict:
    """One episode, one tracker, and the error against the rival's real orders."""
    sim = FastSim(configuration={"seed": seed, "episodeSteps": 720}, validate="dev")
    pol0 = scripted(0, sell_every=2, buy_product=False, market_silent=seat0_silent)
    pol1 = scripted(1, sell_every=6, buy_product=True, sell_on_center=True)
    tracker = MarketTracker(player=0)
    model = OpponentModel()

    residual, residual7, residual_dual, shed_err, pred_err = [], [], [], [], []
    floor_turns, rival_moved, our_moved = 0, 0.0, 0.0
    seven = [G_IX[g] for g in GOODS if g not in ("WHEAT", "FERTILIZER")]
    duals = [G_IX[g] for g in ("WHEAT", "FERTILIZER")]

    for t in range(days * 24):
        obs0, obs1 = sim.observations()[:2]
        a0, a1 = pol0(obs0), pol1(obs1)
        true_rival, _ = committed_flow(a1, dict(sim.state[1].observation.private["shed"]),
                                       float(sim.state[1].observation.farms[1]["money"]),
                                       tracker.inventory)
        our_sold, _ = committed_flow(a0, dict(sim.state[0].observation.private["shed"]),
                                     float(sim.state[0].observation.farms[0]["money"]),
                                     tracker.inventory)
        pred = (np.array([model.expected_sell(g, t, int(tracker.prices[G_IX[g]]))
                          for g in GOODS]) if t else np.zeros(len(GOODS)))
        rec = tracker.observe(obs0)
        tracker.note_our_action(a0, obs0)
        if rec is not None:
            model.observe(rec, tracker)
            diff = np.abs(rec.rival_sales - true_rival)
            residual.append(float(diff.sum()))
            residual7.append(float(diff[seven].sum()))
            residual_dual.append(float(diff[duals].sum()))
            true_shed = np.array([float(dict(sim.state[1].observation.private["shed"]).get(g, 0))
                                  for g in GOODS])
            shed_err.append(float(np.abs(rec.rival_stock - true_shed).sum()))
            floor_turns += int(bool(np.any(rec.price_was_floor)))
            pred_err.append(float(np.abs(pred - true_rival).sum()))
        rival_moved += float(true_rival.sum())
        our_moved += float(our_sold.sum())
        sim.step([a0, a1])

    return dict(tracker=tracker, model=model, sim=sim, steps=days * 24,
                residual=np.array(residual), residual7=np.array(residual7),
                residual_dual=np.array(residual_dual), shed=np.array(shed_err),
                pred=np.array(pred_err), floor_turns=floor_turns,
                rival_moved=rival_moved, our_moved=our_moved,
                final_money=sim.money())


def slot_inference_experiment(true_index: int, our_slot: int = 6,
                              lot: int = 100, our_lot: int = 40) -> dict:
    """Controlled test of the hidden order: their average price is the measurement."""
    sim = FastSim(configuration={"seed": 5, "episodeSteps": 720}, validate="dev")
    tracker = MarketTracker(player=0)
    passive = {"farmer": ["PASS"], "hands": [], "market": []}
    for _ in range(12):
        tracker.observe(sim.observations()[0])
        tracker.note_our_action(passive, sim.observations()[0])
        sim.step([passive, passive])

    obs0 = sim.observations()[0]
    sim.state[0].observation.private["shed"]["WHEAT"] = our_lot
    sim.state[1].observation.private["shed"]["WHEAT"] = lot
    a0 = {"farmer": ["PASS"], "hands": [],
          "market": [["BUY_SEED", "WHEAT", 1]] * our_slot + [["SELL", "WHEAT", our_lot]]}
    a1 = {"farmer": ["PASS"], "hands": [],
          "market": [["BUY_SEED", "WHEAT", 1]] * true_index + [["SELL", "WHEAT", lot]]}

    inventory0 = float(obs0["market"]["inventory"]["WHEAT"])
    money0 = float(sim.state[1].observation.farms[1]["money"])
    tracker.observe(obs0)
    tracker.note_our_action(a0, obs0)
    sim.step([a0, a1])
    money1 = float(sim.state[1].observation.farms[1]["money"])
    spend = true_index * K.CROPS["WHEAT"]["seed"]
    their_avg = (money1 - money0 + spend) / lot
    return dict(true_index=true_index, our_slot=our_slot, their_avg=their_avg,
                with_us=infer_rival_slot({our_slot: our_lot}, lot, "WHEAT",
                                         inventory0, their_avg),
                without_us=infer_rival_slot({}, lot, "WHEAT", inventory0, their_avg))


def main(days: int = 6) -> int:
    grid = np.concatenate([np.arange(0, 60, 1.0), np.arange(100, 12000, 7.0)])
    worst = max(int(np.max(np.abs(price_vec(g, grid)
                                   - np.array([K.market_price(g, float(i)) for i in grid]))))
                for g in GOODS)
    print(f"[1] price parity: max |delta| = {worst} over {len(grid)} inventories x 9 goods")

    for label, silent in (("isolation (our seat trades nothing)", True),
                          ("both seats trading", False)):
        r = run_episode(days, seat0_silent=silent)
        res = r["residual"]
        print(f"\n[2] {label} — {r['steps']} turns, final money {r['final_money']}, "
              f"units sold: rival {r['rival_moved']:.0f}, ours {r['our_moved']:.0f}")
        print(f"      residual vs committed orders : exact {int((res == 0).sum())}/{len(res)} turns, "
              f"mean {res.mean():.4f}, max {res.max():.4f}")
        print(f"        the 7 one-way goods        : mean {r['residual7'].mean():.4f}, "
              f"max {r['residual7'].max():.4f}")
        print(f"        the 2 dual goods (net only): mean {r['residual_dual'].mean():.4f}, "
              f"max {r['residual_dual'].max():.4f}")
        print(f"      rival shed estimate vs true  : mean {r['shed'].mean():.2f}, "
              f"max {r['shed'].max():.2f}")
        print(f"      turns with a floor price     : {r['floor_turns']}")
        print(f"      model: predicted vs realised : mean {r['pred'].mean():.3f} units/turn")

    tracker = r["tracker"]
    obs = r["sim"].observations()[0]
    mean, sd = drain_forecast(obs, 24)
    print(f"\n[3] demand forecast, next 24 turns (open now: "
          f"{obs['town']['unlocked_shops'] or 'none'}):")
    for g in ("WHEAT", "MILK", "WOOL", "MELON"):
        print(f"      {g:<11} mean {mean[G_IX[g]]:6.2f}  sd {sd[G_IX[g]]:5.2f}")

    print("\n[4] hidden intra-turn order (rival's average price -> their slot):")
    for k in (0, 3, 9):
        e = slot_inference_experiment(true_index=k)
        side = "before" if e["with_us"]["estimate"] < e["our_slot"] else "after"
        print(f"      rival at index {k} -> inferred {e['with_us']['estimate']} "
              f"({side} our slot {e['our_slot']}), avg price {e['their_avg']:.2f}; "
              f"without our volume: unidentifiable={not e['without_us']['identifiable']}")

    lot, inventory = 100.0, float(K.MARKET_I0)
    schedules = default_schedules(lot)
    rival = np.array([[lot, 0.0], [0.0, lot], [lot / 2, lot / 2]])
    t0 = time.perf_counter()
    A = slot_game_matrix("WHEAT", inventory, schedules, rival, drain_per_turn=1.0, turns=24)
    mix_lp, value_lp = maximin_mixed_lp(A)
    mix_sq, worst_sq, mean_sq = maximin_mixed_slsqp(A)
    slot_ms = (time.perf_counter() - t0) * 1e3
    print(f"\n[5] slot game: {lot:.0f} wheat, inventory {inventory:.0f}, drain 1/turn")
    print(f"      payoff matrix:\n{np.round(A, 1)}")
    print(f"      maximin LP (highs)  : mix {np.round(mix_lp, 3)} security {value_lp:.1f}")
    print(f"      SLSQP risk-adjusted : mix {np.round(mix_sq, 3)} worst {worst_sq:.1f}, "
          f"mean {mean_sq:.1f}")
    print(f"      dump-now mean {A[0].mean():.1f} vs split-thirds mean {A[2].mean():.1f}")

    goods = ["WHEAT", "CARROT", "STRAWBERRY", "MILK", "WOOL"]
    prices0 = np.array([price_of(g, float(K.MARKET_I0)) for g in goods], dtype=float)
    drain = np.array([17.5, 10.9, 14.2, 10.9, 7.6])
    scenarios = np.array([[1.0] * 5, [0.6] * 5, [0.3, 0.5, 0.4, 0.5, 0.5]])
    rates = np.array([1.4, 1.33, 0.37, 0.4, 0.3])
    plan = season_plan_maximin(goods, 30, rates, 100.0, prices0, scenarios, drain)
    tiles = round_tiles(plan["tiles"])
    print(f"\n[6] season LP (highs): objective {plan['objective']:.0f} coins")
    for gi, g in enumerate(goods):
        cap = drain[gi] * 30 * scenarios[:, gi].min()
        print(f"      {g:<11} tiles {plan['tiles'][gi]:6.2f} -> {tiles[gi]:3d}, "
              f"season sales {plan['units'][gi].sum():6.1f} of {cap:.1f} absorbable")

    print(f"\n[7] timings: observe {tracker.__class__.__name__} per-turn work above; "
          f"slot matrix+solve {slot_ms:.1f} ms (an hour-0 decision)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(int(sys.argv[1]) if len(sys.argv) > 1 else 6))
