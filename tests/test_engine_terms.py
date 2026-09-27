"""The engine's own numbers, resolved in ONE place.

`world/terms.EngineTerms` is the single reader of the keys the engine itself
takes out of a run's configuration (`farmHandCostMult`, `shedCapacity`, the
town's intervals — kaggriculture.py:552-553, :733-734, :867), and the single
place that decides which source answers: the observation's own `configuration`
when the harness carried one, else the world's transcription of the engine's
default. The guards below hold that resolution, the wiring from the manager's
turn into it, and the one-definition rule for the per-turn order cap and the
planner's own hand price.

None of these assert a machine's speed, and none of them need one: every claim
here is a structural fact about which number comes from where.
"""

from __future__ import annotations

import pytest

from agent.world import rules
from agent.world.terms import EngineTerms


def _env_obs():
    """A real day-start observation, from the engine, like the other guards."""
    from offline_lab.kaggle_env import new_environment

    env = new_environment({"seed": 0})
    env.reset(2)
    return env, env.state[0].observation


# --- the resolution -------------------------------------------------------


def test_without_a_configuration_the_transcription_answers():
    """The shipped path: no run configuration reaches the agent, so the world's
    transcription of the engine's defaults is the answer — and it is the SAME
    number, not a copy of it."""
    terms = EngineTerms.from_obs({})
    assert terms.hand_cost_mult == rules.FARM_HAND_COST_MULT
    assert terms.shed_capacity == rules.SHED_CAPACITY
    assert terms.shop_interval == rules.SHOP_SELL_INTERVAL_TURNS
    assert terms.center_interval == rules.CENTER_SELL_INTERVAL_TURNS
    assert terms.shop_unlock_interval == rules.SHOP_UNLOCK_INTERVAL_DAYS
    assert terms.board_size == rules.BOARD_SIZE
    assert terms.max_orders_per_turn == rules.MAX_MARKET_ORDERS_PER_TURN
    assert terms.turns_per_day == rules.TURNS_PER_DAY
    assert terms.episode_steps == rules.EPISODE_STEPS
    assert terms.act_timeout == rules.ACT_TIMEOUT_S
    assert terms.starting_money == rules.STARTING_MONEY
    assert terms.weed_spawn_chance == rules.WEED_SPAWN_CHANCE
    assert terms.hand_cost_mult == rules.FARM_HAND_COST_MULT == 1, (
        "one reference for what a hand costs — the engine's own default, not a "
        "planner 0 sitting beside it")


def test_the_observation_configuration_wins_for_what_it_names():
    """A harness (or a future two-argument entry) that hands the run's own
    configuration in moves the number — and only the number it names."""
    terms = EngineTerms.from_obs({"configuration": {"shedCapacity": 250,
                                                    "farmHandCostMult": 3}})
    assert terms.shed_capacity == 250
    assert terms.hand_cost_mult == 3
    assert terms.shop_interval == rules.SHOP_SELL_INTERVAL_TURNS   # untouched
    assert terms.shop_unlock_interval == rules.SHOP_UNLOCK_INTERVAL_DAYS


def test_a_callers_own_say_so_wins_over_the_observation():
    """A caller stating what THIS run is (a test, a probe, a harness that passes
    the configuration beside the observation) is not outvoted by the board."""
    terms = EngineTerms.from_obs({"configuration": {"shedCapacity": 250}},
                                 {"shedCapacity": 7, "farmHandCostMult": 0})
    assert terms.shed_capacity == 7
    assert terms.hand_cost_mult == 0


def test_get_speaks_the_engines_own_key_names():
    """The callers on this path arrived speaking the engine's vocabulary
    (`belief.market._get`), so one lookup shape answers both spellings."""
    terms = EngineTerms(hand_cost_mult=4)
    assert terms.get("farmHandCostMult") == 4
    assert terms.get("FARM_HAND_COST_MULT") == 4
    assert terms.get("SHED_CAPACITY") == rules.SHED_CAPACITY
    assert terms.get("notAKey", 12) == 12
    assert terms.get("notAKey") is None


def test_every_default_is_the_environments_own_default():
    """His rule: an env config's default IS the env's default — and we model all.

    Read from the INSTALLED `kaggle_environments` config file, key by key, so a
    drifted transcription or an env key the agent never modelled fails here
    instead of silently playing a different run than the one it was given.
    """
    import json
    import pathlib

    import kaggle_environments as KE

    from agent.world.terms import _KEYS

    spec_path = (pathlib.Path(KE.__file__).parent / "envs" / "kaggriculture"
                 / "kaggriculture.json")
    spec = json.loads(spec_path.read_text(encoding="utf-8"))["configuration"]
    assert "shedCapacity" in spec and len(spec) >= 14, spec

    terms = EngineTerms()
    unmodelled = [k for k in spec if k not in _KEYS]
    assert not unmodelled, (
        f"env config key(s) the agent does not model at all: {unmodelled}")

    for key, body in spec.items():
        if key in ("seed", "marketParams"):
            continue                    # carried as-is (None = the env's default)
        # `episodeSteps` and `actTimeout` are bare values in the file; every
        # other key is a spec object carrying its `default`.
        want = body["default"] if isinstance(body, dict) else body
        got = terms.get(key)
        assert got == want, f"{key}: agent default {got!r} != the env's {want!r}"
    assert terms.seed is None and terms.market_params is None, (
        "the env's own defaults for `seed` and `marketParams` are null/{} — "
        "the agent must not invent one")


def test_a_missing_or_odd_configuration_is_not_an_error():
    """An observation without one, and a configuration that is not a mapping,
    both fall back to the transcription: a forecast must never fail for this."""
    assert EngineTerms.from_obs(None) == EngineTerms()
    assert EngineTerms.from_obs({"configuration": None}) == EngineTerms()
    assert EngineTerms.from_obs({"configuration": 5}) == EngineTerms()


# --- the wiring -----------------------------------------------------------


def test_the_run_multiplier_reaches_the_hire_bill():
    """The multiplier the terms resolved is the one `hire_orders` is GIVEN.

    Structural, not arithmetic: the bill itself is zero while the planner's own
    `HAND_COST_MULT` is zero by the owner's order (the next guard pins that), so
    the only way to see the wire is to watch the call.
    """
    from agent.planner import market as K

    seen: list = []
    real = K.hire_orders

    def spy(hands, hires_today, multiplier=1):
        seen.append(int(multiplier))
        return real(hands, hires_today, multiplier)

    _env, obs = _env_obs()
    K.hire_orders = spy
    try:
        built = K.build(obs, (), hands=3, terms=EngineTerms(hand_cost_mult=7))
    finally:
        K.hire_orders = real

    assert seen == [7], f"the bill multiplier the terms resolved was {seen}"
    assert len([r for r in built.rows[0] if r and r[0] == "HIRE"]) == 3


def test_one_reference_for_what_a_hand_costs():
    """ONE number, and setting it moves the bill AND the LP's hour price.

    `rules.FARM_HAND_COST_MULT` is the reference (the engine's own default for
    `farmHandCostMult`), `EngineTerms.hand_cost_mult` defaults to it and carries
    the run's own value when the harness handed one in, and both readers go
    through `rules.hire_cost` — so there is no second knob that can disagree
    with the first, and a run that pays expensive hands is priced expensive.
    """
    from agent.planner import day as D
    from agent.planner.market import hire_orders
    from agent.world.rules import HIRE_SEQUENCE

    assert rules.FARM_HAND_COST_MULT == 1, "the engine's own default"
    assert EngineTerms().hand_cost_mult == rules.FARM_HAND_COST_MULT

    # Both readers, no argument: the reference answers for both.
    ladder = int(sum(HIRE_SEQUENCE[:3]))
    assert hire_orders(3, 0)[1] == D.hire_bill(3, 0) == rules.FARM_HAND_COST_MULT * ladder

    # ...and a run that says hands cost 3x fib moves both, once.
    assert hire_orders(3, 0, 3)[1] == D.hire_bill(3, 0, 3) == 3 * ladder, (
        "the bill and the LP's hour price disagree about what a hand costs: "
        "they must be one `rules.hire_cost` call, not two products")

    # The engine's own vocabulary answers the same reference.
    assert EngineTerms(hand_cost_mult=3).get("farmHandCostMult") == 3
    assert EngineTerms(hand_cost_mult=3).get("FARM_HAND_COST_MULT") == 3


def test_the_manager_holds_the_runs_terms():
    """The manager resolves the terms once per turn and keeps them, so nothing
    downstream has to re-read a configuration the entry point never gets."""
    from agent.config import Config
    from agent.manager.core import Manager

    _env, obs = _env_obs()
    manager = Manager(Config())
    manager.observe(obs, {"farmHandCostMult": 5, "shedCapacity": 64})
    assert manager.terms.hand_cost_mult == 5
    assert manager.terms.shed_capacity == 64
    # ...and the shipped path (no configuration in the observation) is the
    # engine's own defaults, not a second copy of them.
    manager.observe(obs)
    assert manager.terms == EngineTerms()


# --- one definition per number -------------------------------------------


def test_the_per_turn_order_cap_has_one_definition():
    """F031's ten reached the agent through five copies once (two in
    `world/rules.py`, one in `dispatch.py`, one in `belief/shed.py`, and the
    literals in `planner/market.py`). One definition now, and this is it."""
    from agent import dispatch
    from agent.belief import shed
    from agent.planner import market as K

    cap = rules.MAX_MARKET_ORDERS_PER_TURN
    assert cap == 10
    assert dispatch.MAX_MARKET_ORDERS_PER_TURN is cap
    assert shed.MAX_MARKET_ORDERS_PER_TURN is cap
    assert (K.merge.__kwdefaults__ or {}).get("cap") == cap, (
        "merge restates the cap instead of reading the world's")
    assert (K.build.__kwdefaults__ or {}).get("cap") is None, (
        "build must take the cap from the run's own terms (`terms."
        "max_orders_per_turn`), never restate it")
    assert not hasattr(rules, "MAX_ORDERS_PER_TURN"), (
        "the second spelling of the same cap is back")


def test_the_season_length_and_the_turn_length_have_one_definition():
    """`belief` spelled the season and the day itself once (`SEASON_DAYS`,
    `TURNS_PER_DAY = 24`); the manager path reads the world's, and both are the
    same object."""
    from agent.belief import market as BM
    from agent.belief import shed

    assert BM.TURNS_PER_DAY is rules.TURNS_PER_DAY
    assert not hasattr(shed, "SEASON_DAYS"), (
        "belief's own spelling of the season is back; the world has DAYS")
    assert rules.DAYS == 30
