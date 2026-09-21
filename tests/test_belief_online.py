"""The rival model's online update, and whether the queue can read it (#95).

`OpponentModel` is primed from the corpus and then meant to keep counting on top
of it from live play — belief's own docstring says so, and its documented chain
is `tracker.observe() -> model.observe(rec, tracker) -> expected_sell(...)`.
Nothing walked the middle link, so the table was frozen for a whole season.

Walking it is not enough on its own, and that is what these guards pin: the
predictor reads the ACTIVITY-keyed row (`policy` / `expected_sell` are called
with the tracker's bucket), so the update has to write that same row. Writing
the plain `(good, day, bucket)` row instead leaves the state's own counts at the
artifact's and its prediction at the pretrained answer to four places — measured
on three days of live play: |move| 0.0001 against 0.8876.

R007: both guards were broken and seen red before they were trusted.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.belief.opponent import OpponentModel        # noqa: E402
from agent.belief.tracker import MarketTracker         # noqa: E402
from agent.config import Config                        # noqa: E402
from agent.manager import core as MC                   # noqa: E402
from agent.world.rules import TURNS_PER_DAY            # noqa: E402
from offline_lab.kaggle_env import new_environment     # noqa: E402

PASS = {"farmer": ["PASS"], "hands": [], "market": []}


def _watched(days: int = 3):
    """A tracker fed real observations, and the model that should learn from it."""
    reference = OpponentModel(pretrained=True)
    live = OpponentModel(pretrained=True)
    env = new_environment({"seed": 0})
    env.reset(2)
    tracker = MarketTracker(player=0)
    records = 0
    for _ in range(TURNS_PER_DAY * days):
        obs = env.state[0].observation
        rec = tracker.observe(obs)
        if rec is not None:
            live.observe(rec, tracker)
            records += 1
        env.step([dict(PASS), dict(PASS)])
    return reference, live, tracker, records


def test_the_online_update_writes_the_row_the_predictor_reads():
    """The update lands on the state's own row, not on one nobody reads.

    A plain-key write is not harmless: it feeds the good's marginal, so the
    prediction moves a little and the guard passes for the wrong reason. What
    separates the two is the state's OWN row, which is what the predictor reads
    when it is given a bucket.
    """
    reference, live, tracker, records = _watched()
    assert records > 0, "the tracker produced no flow records to learn from"

    changed = [k for k in live.counts
               if not np.array_equal(live.counts[k],
                                     reference.counts.get(
                                         k, np.zeros_like(live.counts[k])))]
    assert changed, "live play taught the model nothing at all"
    plain = [k for k in changed if len(k) != 4]
    assert not plain, (
        f"the update wrote a row the predictor cannot read: {plain[:3]}")

    # And the predictor's answer for such a state actually moves.
    moved = 0
    for key in changed[:200]:
        good, day, bucket, activity = key
        for step in range(day * TURNS_PER_DAY, day * TURNS_PER_DAY + 6):
            for price in (10, 25, 35, 60, 100, 120, 160, 200, 250):
                if live._key_activity(good, step, price, activity) != key:
                    continue
                a = live.policy(good, step, price, activity=activity)
                b = reference.policy(good, step, price, activity=activity)
                if not np.allclose(a, b):
                    moved += 1
                break
    assert moved, "the update reached a row, but no prediction changed"


def test_the_manager_hands_the_record_to_the_model(monkeypatch):
    """The middle link of belief's chain, walked by the shipped path."""
    seen: list[tuple] = []
    original = OpponentModel.observe

    def spy(self, rec, tracker):
        seen.append((rec.step, tracker.step))
        return original(self, rec, tracker)

    monkeypatch.setattr(OpponentModel, "observe", spy)
    env = new_environment({"seed": 0})
    env.reset(2)
    manager = MC.Manager(Config())
    manager.observe(env.state[0].observation, {"farmHandCostMult": 1})
    for _ in range(3):
        env.step([dict(PASS), dict(PASS)])
        manager.step(env.state[0].observation, budget_ms=0.0)

    assert seen, "the manager never handed a flow record to the model"
    assert len(seen) >= 3, f"only {len(seen)} of the watched turns reached the model"
    assert all(step >= 0 for step, _t in seen)
