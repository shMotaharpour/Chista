"""The trained rival model, wired into the queue that actually ships (#95).

#95's finding was a wiring gap, not a missing module: `OpponentModel` (24.87M
observations, 2,811 states), `MarketTracker` and the slot circuit were all built
and measured, and the shipped path called none of them. `market_queue` takes
`model=`/`activity=` — the day layer never passed them, so the queue spread the
day's sells uniformly across 24 hours and never read the rival's own behaviour.

What each guard is for:

- the model reaches the queue AT ALL (a spy on the call the manager makes);
- the tracker is fed every turn and for our own seat — a tracker built for the
  wrong seat reads the other farm's flows without saying so;
- the queue it produces is actually different (measured: on a shed over
  capacity, the plain spread sells 6 units in every one of the 24 hours and the
  model-timed queue sells nothing before hour 16);
- the 1,222 ms load happens at import and not inside a turn (the hour-0 budget
  is 965 ms, so a lazy load costs the season's first day its plan).

R007: each guard was broken and seen red before it was trusted.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = str(Path(__file__).resolve().parents[1])

from agent.belief import shed as SH          # noqa: E402
from agent.config import Config              # noqa: E402
from agent.manager import core as MC         # noqa: E402
from offline_lab.kaggle_env import new_environment   # noqa: E402

PASS = {"farmer": ["PASS"], "hands": [], "market": []}


def _board(days: int = 2, shed: dict | None = None):
    """A real observation, after `days` of nothing, with a shed if asked."""
    env = new_environment({"seed": 0})
    env.reset(2)
    for _ in range(24 * days):
        env.step([dict(PASS), dict(PASS)])
    obs = dict(env.state[0].observation)
    if shed is not None:
        private = dict(obs.get("private", {}))
        private["shed"] = dict(shed)
        obs["private"] = private
    return env, obs


def _sell_hours(rows) -> list[tuple[int, int]]:
    """(hour, units sold) for every hour that sells anything."""
    return [(hour, sum(int(o[2]) for o in row if o and o[0] == "SELL"))
            for hour, row in enumerate(rows)
            if any(o and o[0] == "SELL" for o in row)]


def test_the_manager_builds_one_forecast_per_turn(monkeypatch):
    """One curve, two consumers: the master's objective and the sell queue.

    Belief's `forecast` was built twice per turn — once for the master's price
    path (1.1 ms) and once inside `market_queue` (1.4 ms) — over two horizons
    for the same curve. The manager now builds it once and hands it to both, and
    this guard counts the builds rather than reading the call sites.
    """
    from agent.belief import market as BM
    from agent.belief import shed as SH

    calls: list = []
    original_forecast = BM.forecast

    def counting_forecast(*args, **kwargs):
        calls.append(kwargs.get("days"))
        return original_forecast(*args, **kwargs)

    handed: dict = {}
    original_queue = SH.market_queue

    def spying_queue(obs, *args, **kwargs):
        handed["forecast_obj"] = kwargs.get("forecast_obj")
        return original_queue(obs, *args, **kwargs)

    monkeypatch.setattr(BM, "forecast", counting_forecast)
    monkeypatch.setattr(SH, "market_queue", spying_queue)

    _env, obs = _board()
    manager = MC.Manager(Config())
    manager.observe(obs, {"farmHandCostMult": 1})

    assert len(calls) == 1, f"the turn built {len(calls)} forecasts: {calls}"
    assert handed.get("forecast_obj") is not None, (
        "the queue was left to build its own forecast again")


def test_handing_the_forecast_in_does_not_change_the_prices():
    """The reuse must be the same curve, not a different one.

    The master's price path is what its objective is optimised against, so a
    hand-off that shifted it by a coin would change every plan. Measured by
    building the path both ways on the same board.
    """
    import numpy as np

    from agent.belief.market import forecast
    from agent.planner import master as M

    _env, obs = _board()
    days = 20
    p_full, _w = M.dual_stand_in(obs)
    fc = forecast(obs, days=days, config={"farmHandCostMult": 1})

    handed, source_a = M._product_price_path(obs, days, p_full[:days],
                                             forecast_obj=fc)
    built, source_b = M._product_price_path(obs, days, p_full[:days])

    assert np.array_equal(handed, built), (
        "the handed-in forecast priced a different curve")
    assert source_a == source_b


def test_the_manager_hands_the_trained_model_to_the_queue(monkeypatch):
    """The shipped path passes `model=`/`activity=`, and they are the real ones.

    A model that never reaches the queue is the whole of #95: the pieces exist,
    the socket exists, and the day layer calls the socket without them.
    """
    seen: list[tuple] = []
    original = SH.market_queue

    def spy(obs, *args, **kwargs):
        seen.append((kwargs.get("model"), kwargs.get("activity")))
        return original(obs, *args, **kwargs)

    monkeypatch.setattr(SH, "market_queue", spy)
    _env, obs = _board()
    manager = MC.Manager(Config())
    manager.observe(obs, {"farmHandCostMult": 1})

    assert seen, "the manager's day plan never reached `market_queue`"
    model, activity = seen[0]
    assert model is MC.opponent_model() is not None, (
        "the queue was called without the trained model")
    assert isinstance(activity, int), f"activity must be the tracker's bucket: {activity!r}"


def test_the_tracker_is_fed_every_turn_and_reads_our_own_seat():
    """Not just hour 0: the bucket is a window over the last 24 TURNS.

    `activity_bucket(step, window=24)` counts the rival's inferred sales over
    the last 24 records, so a tracker fed once a day would read 24 DAYS of
    history and call it a day. It is 0.1 ms measured, so every turn can afford
    it, and `step` now takes the observation to do it.
    """
    env, obs = _board()
    manager = MC.Manager(Config())
    manager.observe(obs, {"farmHandCostMult": 1})
    assert manager.tracker is not None
    first = manager.tracker.step

    for _ in range(3):
        env.step([dict(PASS), dict(PASS)])
    manager.step(env.state[0].observation)

    assert manager.tracker.player == int(obs.get("player", 0)), (
        "the tracker is watching the wrong seat")
    assert manager.tracker.step == first + 3, (
        f"the tracker missed turns: {first} -> {manager.tracker.step}")
    assert isinstance(manager._activity(), int)


def test_the_model_re_times_the_sell_hours_on_a_board_that_must_sell():
    """The wiring has to BUY something, or it is a grep with extra steps.

    Driven through `planner.market.build` — the call the day layer actually
    makes — so a break anywhere on the chain (build -> sell_rows ->
    market_queue) is caught here, not just in belief's own entry point.

    Measured on this board (a shed over capacity, so the guard must release
    stock): the plain spread sells 6 units in every one of the 24 hours; the
    model-timed queue sells 17 units an hour from hour 16 on and nothing before
    it. The quantities are the guard's either way — only the hours move.
    """
    from agent.planner import market as K

    model = MC.opponent_model()
    assert model is not None, "no trained table: this guard cannot say anything"

    _env, obs = _board(shed={"WHEAT": 140, "CARROT": 90})
    plain = K.build(obs, (), hands=0).rows
    timed = K.build(obs, (), hands=0, model=model, activity=0).rows

    plain_hours, timed_hours = _sell_hours(plain), _sell_hours(timed)
    assert plain_hours, "this board was supposed to have stock to sell"
    assert timed_hours, "the model-timed queue sold nothing at all"
    assert timed != plain, "the model changed nothing — the circuit is not reached"
    assert sum(u for _h, u in timed_hours) == sum(u for _h, u in plain_hours), (
        "re-timing must not change how much is sold, only when")
    assert min(h for h, _u in timed_hours) > min(h for h, _u in plain_hours), (
        "the circuit is supposed to hold the stock to the peak, not spread it")


def test_the_entry_import_warms_the_model_in_a_fresh_process():
    """1,222 ms at import, never inside a turn.

    Run in a fresh interpreter on purpose: in-process the singleton is already
    warm from whichever test ran first, so the assertion would pass whether or
    not the entry module still does the warming.
    """
    code = (
        f"import sys; sys.path.insert(0, {ROOT!r})\n"
        "import agent.manager.core as MC\n"
        "assert MC._OPPONENT_TRIED is False, 'warm before the entry was imported'\n"
        "import agent.main\n"
        "print(int(MC._OPPONENT_TRIED), MC.opponent_model() is not None)\n"
    )
    done = subprocess.run([sys.executable, "-c", code],
                          capture_output=True, text=True)
    assert done.returncode == 0, done.stderr[-800:]
    assert done.stdout.strip() == "1 True", (
        f"importing agent.main did not warm the model: {done.stdout!r}")
