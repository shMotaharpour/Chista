"""The rival-activity dimension: the model conditioned on the rival's OWN
recent behaviour — the PASS-rival gap, measured.

The gap the owner caught (2026-09-21): against a PASS rival the truth is
zero rival sells, but the corpus-primed model predicted ~12.4 WHEAT units
PER TURN (≈100/day) — the table averages every opponent's behaviour, and a
passive rival is out of that distribution. The market forecast then
carried a phantom rival supply.

The measured lever, from the store (DuckDB window over 5 dump dates): the
rival's own sell volume in the 24 turns BEFORE the current one buckets the
corpus into regimes —

    silent (0 units in the window)   15,102 turns, 0.65 sold/turn after
    low (1-10)                       88,438 turns, 1.19/turn
    mid (11-60)                     920,058 turns, 2.63/turn
    high (60+)                    2,591,152 turns, 7.03/turn

A PASS rival lives in the silent regime, where the corpus's own answer is
near-pure hold. This module carries the bucket function; the model's key
gains the bucket as its activity dimension (per-good, as #65 decided).
"""
from __future__ import annotations

import numpy as np

from agent.belief.tracker import MarketTracker

#: the rival's sell volume (all goods) over the last 24 turns, bucketed:
#: start (no history yet) / silent / low / mid / high.
ACTIVITY_EDGES = (0.0, 10.0, 60.0)


def activity_bucket(tracker: MarketTracker) -> int:
    """0 = start (no history yet), 1 = silent, 2 = low, 3 = mid, 4 = high."""
    if not tracker.records:
        return 0
    last = tracker.records[-24:]
    vol = sum(float(rec.rival_sales.sum()) for rec in last)
    b = 1
    for edge in ACTIVITY_EDGES:
        if vol > edge:
            b += 1
    return b


def main() -> int:
    """The PASS-rival measurement: the model vs the truth over 10 days."""
    from offline_lab.fast_sim import FastSim
    import numpy as np
    from agent.belief.opponent import OpponentModel
    PASS = {"farmer": ["PASS"], "hands": [], "market": []}
    sim = FastSim(configuration={"seed": 0, "episodeSteps": 720},
                  validate="dev")
    m = OpponentModel(pretrained=True)
    tr = MarketTracker(player=0)
    preds, truths = [], []
    for t in range(10 * 24):
        obs = sim.observations()[0]
        if t > 0:
            rec = tr.observe(obs)
            if rec is not None:
                preds.append(m.expected_sell("WHEAT", t, int(tr.prices[0])))
                truths.append(float(rec.rival_sales[0]))
        tr.note_our_action(PASS, obs)
        sim.step([PASS, PASS])
    print("PASS-rival 10-day check (WHEAT):")
    print(f"  truth: {sum(truths):.0f} units sold by the rival")
    print(f"  model predicted per-turn (last): {preds[-1]:.2f} "
          f"-> ~{preds[-1] * 24:.0f}/day")
    print(f"  sum of per-turn predictions over 10 days: {sum(preds):.0f}")
    print(f"  activity bucket now: {activity_bucket(tr)} "
          f"(1 = silent: the rival sold {sum(truths):.0f} in 10 days)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
