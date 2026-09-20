"""Guards for `belief/slot_circuit.py` — the closed slot circuit (#65 step 3).

The claim under test: choosing today's hourly sell schedule against the
trained rival scenarios earns MORE coins than the uniform spread — and the
circuit's revenue model is the engine's (ladder parity, drain sequencing).

Run:  .venv/bin/python -m tests.test_slot_circuit   (also under pytest)
"""

from __future__ import annotations

import numpy as np

from offline_lab.fast_sim import FastSim
from agent.belief.opponent import OpponentModel
from agent.belief.slot_circuit import (_candidate_schedules, _drain_per_hour,
                                       _revenue, plan_day_slots)
from agent.world.model import PRODUCTS

PASS = {"farmer": ["PASS"], "hands": [], "market": []}


def _sim_to(seed: int, day: int):
    sim = FastSim(configuration={"seed": seed, "episodeSteps": 720},
                  validate="dev")
    for _ in range(day * 24):
        sim.step([PASS, PASS])
    return sim.observations()[0]


def test_the_revenue_model_matches_the_ladder_identity() -> None:
    """With no rival and no drain, revenue = the ladder's own cumsum window."""
    from agent.belief.ladder import sell_coins
    ours = np.zeros(24); ours[0] = 7
    rev = _revenue("WHEAT", 9900.0, ours, np.zeros(24), np.zeros(24))
    assert abs(rev - sell_coins("WHEAT", 9900, 7)) < 1e-6, (rev,)


def test_the_rival_dropping_first_pushes_our_price_down() -> None:
    """The pessimistic interleaving: rival-first quotes us a WORSE ladder."""
    flat = np.zeros(24); flat[:] = 1.0
    none = np.zeros(24)
    without = _revenue("WHEAT", 9900.0, flat, none, np.zeros(24))
    with_rival = _revenue("WHEAT", 9900.0, flat, flat, np.zeros(24))
    assert with_rival < without, (without, with_rival)


def test_the_circuit_chooses_a_schedule_that_beats_the_uniform_spread() -> None:
    """The circuit's expected revenue beats the even spread against the
    trained rival scenarios — on several real observations."""
    model = OpponentModel(pretrained=True)
    wins = 0
    trials = 0
    for seed, day in ((0, 5), (1, 8), (2, 12), (5, 3), (7, 15)):
        obs = _sim_to(seed, day)
        shed = {"WHEAT": 50, "MILK": 30, "MELON": 10}
        for g, lot in shed.items():
            sched, exp = plan_day_slots(g, lot, obs, model)
            # the uniform spread's expected revenue, same scenarios:
            ours = _candidate_schedules(lot)
            from agent.belief.slot_circuit import (_rival_scenarios,
                                                   _drain_per_hour)
            theirs, w = _rival_scenarios(model, g, int(obs["step"]),
                                         int(obs["market"]["prices"][g]))
            uniform = np.full(24, lot / 24.0)
            drain = _drain_per_hour(obs)
            inv = float(obs["market"]["inventory"][g])
            e_uniform = float(sum(
                w[j] * _revenue(g, inv, uniform, theirs[j], drain)
                for j in range(len(theirs))))
            trials += 1
            wins += exp >= e_uniform - 1e-6
    assert wins == trials, f"the circuit lost {trials - wins}/{trials} slots"


def test_the_schedule_respects_the_remaining_hours() -> None:
    """A mid-day forecast schedules only into the hours that remain."""
    obs = _sim_to(0, 10)                    # ends at hour 23 of day 10
    obs = dict(obs)
    obs["step"] = 10 * 24 + 18              # hour 18: 6 hours left
    obs["market"] = obs["market"]
    model = OpponentModel(pretrained=True)
    sched, _rev = plan_day_slots("WHEAT", 20, obs, model)
    assert sched[:18].sum() == 0, sched
    assert abs(sched.sum() - 20) < 1e-6, sched


def test_the_candidate_family_is_not_flat() -> None:
    """The 6 candidates must disagree on revenue somewhere.

    A circuit whose candidates all price identically is not choosing
    anything — the argmax would be noise. Against a HEAVY rival (90 units
    front/spread/back of WHEAT) the shapes must spread by > 1%: the drain
    is small, so dumping beats spreading, and the rival's front-load makes
    the back shapes worse still.
    """
    obs = _sim_to(0, 5)
    model = OpponentModel(pretrained=True)
    g = "WHEAT"
    inv = float(obs["market"]["inventory"][g])
    drain = _drain_per_hour(obs)
    heavy_front = np.zeros(24); heavy_front[:8] = 90 / 8
    heavy_spread = np.full(24, 90 / 24)
    heavy_back = np.zeros(24); heavy_back[-8:] = 90 / 8
    scenarios = [(heavy_front, 0.32), (heavy_spread, 0.32), (heavy_back, 0.32),
                 (np.zeros(24), 0.04)]
    cands = _candidate_schedules(120)
    labels = ["dump", "even", "front", "back", "mid", "drip"]
    exp = []
    for i in range(len(cands)):
        sched = np.zeros(24)
        sched[0:] = cands[i]
        exp.append(sum(wt * _revenue(g, inv, sched, theirs, drain)
                       for theirs, wt in scenarios))
    spread = (max(exp) - min(exp)) / max(exp) * 100
    assert spread > 1.0, (
        f"the candidate family is flat ({spread:.2f}% spread): the argmax "
        "is noise, so the circuit is not choosing anything")


def main() -> int:
    tests = [(k, v) for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    failures = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except Exception as exc:                       # noqa: BLE001
            failures += 1
            print(f"FAIL {name}: {type(exc).__name__}: {exc}")
    print(f"{len(tests) - failures}/{len(tests)} slot-circuit checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
