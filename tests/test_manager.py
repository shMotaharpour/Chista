"""The manager: the loop closes, the bracket converges, and a season runs.

What is asserted is what the engine and the search actually did, not what the
manager intended. The failure this suite exists to catch is the one that cost
four days: a path that answers every turn with something legal while the thing
it was supposed to be running is not wired at all. A manager that raises is
visible; a manager that quietly plays a worse policy is not.

R007: break the rule a test names and watch it go red before trusting it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.config import Config
from agent.manager.core import Manager
from agent.manager.prices import LABOR, cost_vector, value_vector
from agent.tile_dp.graph import TileGraph
from agent.world.model import N_RESOURCE, PRODUCTS
from agent.world.rules import TURNS_PER_DAY

GRAPH_PATH = Path(__file__).resolve().parents[1] / "agent" / "artifact" / "tile_graph.npz"
PASS_ACTION = {"farmer": ["PASS"], "hands": [], "market": []}


@pytest.fixture(scope="module")
def graph():
    return TileGraph.load(GRAPH_PATH)


# --- config: numbers, and the ones that are not legal ---------------------

def test_an_empty_wage_bracket_is_rejected():
    """A ceiling at or below the floor gives the bisection nothing to search."""
    with pytest.raises(ValueError, match="wage bracket is empty"):
        Config(wage_floor=4.0, wage_ceiling=4.0)


def test_a_probe_budget_inside_its_own_reserve_is_rejected():
    """A probe that cannot outlive its reserve leaves the turn no answer."""
    with pytest.raises(ValueError, match="leaves nothing after"):
        Config(probe_budget_ms=100.0, reserve_ms=100.0)


def test_an_unknown_config_key_is_an_error_not_a_shrug(tmp_path):
    """A number that is not read is a number that is not tuned.

    Falling back to the defaults on a malformed file is how a tuned agent
    plays untuned and nobody finds out, so the file is an error or it is
    honoured — never quietly half-applied.
    """
    bad = tmp_path / "config.json"
    bad.write_text('{"wage_growth": 3.0, "wage_grwoth": 9.0}')
    with pytest.raises(ValueError, match="unknown config keys"):
        Config.load(bad)


def test_a_missing_config_file_is_the_ordinary_case(tmp_path):
    """The tuned file is gitignored while the numbers move, so absence is normal."""
    assert Config.load(tmp_path / "nothing.json") == Config()


def test_a_round_trip_through_the_artifact_keeps_every_number(tmp_path):
    out = tmp_path / "config.json"
    Config(wage_growth=3.5, max_hands=6).dump(out)
    assert Config.load(out) == Config(wage_growth=3.5, max_hands=6)


# --- the two dual vectors -------------------------------------------------

def test_the_value_vector_prices_products_and_nothing_else():
    """`p` is the produce side: only the 9 products are ever produced.

    A non-zero entry on LABOR or a seed would pay the plan for consuming it,
    which is the sign error that makes every chain look profitable.
    """
    p = value_vector([[7.0] * len(PRODUCTS)], days=3)
    assert p.shape == (3, N_RESOURCE)
    assert int((p[0] > 0).sum()) == len(PRODUCTS)
    assert p[0, LABOR] == 0.0


def test_a_short_forecast_repeats_its_last_row_rather_than_extrapolating():
    """A made-up price is a made-up plan; repeating at least says where it came from."""
    p = value_vector([[1.0] * len(PRODUCTS), [5.0] * len(PRODUCTS)], days=4)
    assert p[1].max() == 5.0 and p[3].max() == 5.0


def test_the_cost_vector_puts_the_wage_on_labour_and_leaves_products_free():
    """`w` is the consume side: LABOR, the seeds, the animals, the two DUAL goods.

    Eleven columns and no more. A wage that also landed on CARROT would charge
    the plan for growing it.
    """
    w = cost_vector(3.0, {"WHEAT": 12, "FERTILIZER": 20}, days=2)
    assert w[0, LABOR] == 3.0
    assert int((w[0] > 0).sum()) == 11


def test_a_negative_wage_is_refused_here_not_by_the_contractor():
    """The contractor's R006 check would catch it; a clearer error is cheaper."""
    with pytest.raises(ValueError, match="negative"):
        cost_vector(-1.0, {"WHEAT": 12, "FERTILIZER": 20}, days=2)


# --- the bisection --------------------------------------------------------

def test_the_first_probe_asks_for_labour_free(graph):
    """Free labour is the most work the DP will ever ask for: the cheapest answer.

    Starting anywhere above it would settle on a dearer wage than the day needs
    and leave the farm idling for no reason.
    """
    m = Manager(graph, Config())
    m.view = object()                      # enough for `_next_wage`
    assert m._next_wage() == Config().wage_floor


def test_an_unbracketed_probe_climbs_and_a_bracketed_one_halves(graph):
    """Two different questions, and the same step must not answer both the same way."""
    cfg = Config()
    m = Manager(graph, cfg)
    m.view = object()
    m.probes = [object()]                  # one probe already spent
    m.lo, m.hi = 2.0, None
    assert m._next_wage() == pytest.approx(min(2.0 * cfg.wage_growth, cfg.wage_ceiling))
    m.hi = 10.0
    assert m._next_wage() == pytest.approx(6.0)


def test_the_bracket_settles_inside_its_own_tolerance(graph):
    cfg = Config(wage_tolerance=0.5)
    m = Manager(graph, cfg)
    m.view = object()
    m.lo, m.hi = 4.0, 4.4
    assert m.settled


# --- the whole season -----------------------------------------------------

def test_the_manager_plays_a_season_without_raising_once(graph):
    """Thirty days against a PASS opponent, and the manager answers every one.

    This is the test the old path could not have passed and did not have: its
    replanner could not import, so the ladder answered with `greedy` for four
    days and the agent looked like it worked. Here an exception is counted, not
    swallowed, and one is a failure.
    """
    from agent.dispatch import dispatch_plan
    from offline_lab.kaggle_env import new_environment

    state = {"fail": 0, "days": 0, "calls": 0, "planned": 0, "idle_days": 0,
             "error": None}
    m = Manager(graph, Config())

    def me(obs, config=None):
        state["calls"] += 1
        try:
            if int(obs["hour"]) == 0:
                if state["days"] and not m.fitted:
                    state["idle_days"] += 1
                m.observe(obs, config)
                state["days"] += 1
            if not m.settled:
                m.step()
            plan = m.best()
            if plan is None:
                return dict(PASS_ACTION)
            state["planned"] += 1
            return dispatch_plan(plan, obs)
        except Exception as exc:                       # noqa: BLE001
            state["fail"] += 1
            state["error"] = state["error"] or f"{type(exc).__name__}: {exc}"
            return dict(PASS_ACTION)

    env = new_environment()
    env.run([me, lambda obs, config=None: dict(PASS_ACTION)])

    assert state["fail"] == 0, f"manager raised {state['fail']}x: {state['error']}"
    assert state["days"] == 30, f"observed {state['days']} days, expected 30"
    # Every call, not a count: the grader calls a surviving seat 719 times,
    # not 720 (F058 - seat 0 played all 719 while seat 1 was stopped at 503),
    # and a test that hardcodes the wrong constant fails for the right reason
    # at the wrong place.
    assert state["planned"] == state["calls"], (
        f"only {state['planned']} of {state['calls']} turns were answered from "
        f"a plan: the manager is producing None and the dispatcher is passing")
    assert state["idle_days"] < 30, (
        f"every one of the 30 days went idle: the manager never found a wage "
        f"that fit, which is a broken loop and not a cautious one")
    assert [s.status for s in env.state] == ["DONE", "DONE"]


def test_a_probe_produces_a_plan_the_dispatcher_can_slice(graph):
    """One day, one probe, and every turn of it is a legal, sliceable row."""
    from agent.dispatch import dispatch_plan
    from offline_lab.kaggle_env import new_environment

    env = new_environment()
    env.reset(2)
    obs = env.state[0].observation
    m = Manager(graph, Config())
    m.observe(obs, env.configuration)
    probe = m.step()

    assert probe is not None and probe.fits, f"first probe did not fit: {probe}"
    plan = m.best()
    assert plan is not None
    assert len(plan["market"]) == TURNS_PER_DAY, (
        "the market queue is per TURN (F031: the ten-order cap is per turn, "
        "not per day), so it has one row for each of the day's hours")
    # Every worker's op list is the full day, so `ops[worker][hour]` is what the
    # dispatcher slices; a short list is a day that silently stops acting.
    for unit in plan["units"]:
        assert len(unit) == TURNS_PER_DAY, f"unit op list is {len(unit)} long"
    action = dispatch_plan(plan, obs)
    assert set(action) >= {"farmer", "hands", "market"}


def test_the_days_hires_are_paid_for_before_the_day_is_searched(graph):
    """The purse is checked BEFORE the plan, which is the arm repair.py never had.

    `planner/repair.py` built the day and then deleted whatever the money would
    not cover, so the plan that ran was never the plan that was priced.
    """
    from offline_lab.kaggle_env import new_environment

    env = new_environment()
    env.reset(2)
    obs = env.state[0].observation
    # A reserve above the whole purse leaves nothing to spend, so every wage
    # fails for cash and the bisection runs out of bracket rather than
    # producing a plan it cannot pay for.
    m = Manager(graph, Config(cash_reserve=1e9))
    m.observe(obs, env.configuration)
    while not m.settled:
        m.step()
    assert m.fitted is False, "a day was searched that the purse cannot pay for"
    assert m.best() == {"units": [[["PASS"]] * TURNS_PER_DAY], "market": []}, \
        "a farm that cannot afford to act must still answer, and idle is the answer"
    assert all(p.reason == "cash" for p in m.probes), \
        f"expected every probe to fail on cash, got {[p.reason for p in m.probes]}"
