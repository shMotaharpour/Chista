"""Replay-agent tests: bit-exact replay, seat independence, concurrency.

Run:  .venv/bin/python -m tests.test_replay_agent
      (also works under pytest: pytest tests/test_replay_agent.py)

Core contract — replaying two single-agent records of the SAME match against
each other, with the match's seed, must reproduce the recorded final rewards
bit-identically on BOTH engine paths (world.fast_sim and the real harness).
The record format itself carries no seat identity: the same ReplayAgent
instance logic must work at seat 0, seat 1, or both simultaneously.
"""

from __future__ import annotations

import copy
import json
import random
from pathlib import Path
from typing import Any

from offline_lab.fast_sim import FastSim
from agent.world.replay_agent import (EpisodeRecord, PASS_ACTION,
                                ReplayValidationError, ReplayAgent, SCHEMA)

STEPS = 96
SEED = 4242


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

UNIT_OPS = [["PASS"], ["NORTH"], ["SOUTH"], ["EAST"], ["WEST"], ["WATER"],
            ["HARVEST"], ["DIG"], ["FERTILIZE"], ["BUILD_COOP"],
            ["BUILD_PASTURE"], ["PLANT", "WHEAT"], ["PLANT", "CARROT"],
            ["PLANT", "MELON"]]
MARKET_OPS = [["SELL", "WHEAT", 1], ["BUY_SEED", "WHEAT", 1], ["HIRE"],
              ["BUY_LAND"], ["SELL", "CARROT", 1], ["BUY_ANIMAL", "GOOSE"],
              ["BUY_PRODUCT", "WHEAT", 1]]


def scripted_action(step: int, player: int) -> dict[str, Any]:
    """Deterministic messy action stream (crosses silent-failure paths)."""
    rng = random.Random(f"chista-replay-{step}-{player}")
    return {
        "farmer": list(rng.choice(UNIT_OPS)),
        "hands": [list(rng.choice(UNIT_OPS)) for _ in range(rng.randint(0, 2))],
        "market": [list(rng.choice(MARKET_OPS))
                   for _ in range(rng.randint(0, 3))],
    }


def make_record_data(player: int, steps: int = STEPS,
                     seed: int | None = SEED) -> dict[str, Any]:
    """A record dict for one agent, as the user would build it."""
    return {
        "schema": SCHEMA,
        "seed": seed,
        "configuration": {"episodeSteps": steps},
        "agent_name": f"scripted-p{player}",
        "turns": [{"step": s, "action": scripted_action(s, player)}
                  for s in range(steps)],
    }


def write_record(tmp: Path, player: int, **kw) -> Path:
    path = tmp / f"record_p{player}.json"
    path.write_text(json.dumps(make_record_data(player, **kw)),
                    encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
# load + validate
# --------------------------------------------------------------------------- #

def test_load_validates_before_construct(tmp_path: Path = None) -> None:
    good = write_record(tmp_path, 0)
    rec = EpisodeRecord.load(good)
    assert rec.seed == SEED
    assert rec.agent_name == "scripted-p0"
    assert len(rec.turns) == STEPS

    # a structurally-defective file (bad op shape) must never load with
    # validate=True, but loads with validate=False (ctor only checks schema)
    bad = make_record_data(0)
    bad["turns"][1]["action"] = {"farmer": ["WATER", "EXTRA_ARG"],
                                 "hands": [], "market": []}
    bad_path = tmp_path / "bad.json"
    bad_path.write_text(json.dumps(bad), encoding="utf-8")
    try:
        EpisodeRecord.load(bad_path)
        raise AssertionError("defective file accepted")
    except ReplayValidationError as exc:
        assert "turns[1]" in str(exc)

    # validate=False skips the deep check (trusted files)
    rec = EpisodeRecord.load(bad_path, validate=False)
    assert rec.seed == SEED

    # a wrong schema never becomes an object, even with validate=False
    wrong = dict(make_record_data(0))
    wrong["schema"] = "nope.v9"
    wrong_path = tmp_path / "wrong.json"
    wrong_path.write_text(json.dumps(wrong), encoding="utf-8")
    try:
        EpisodeRecord.load(wrong_path, validate=False)
        raise AssertionError("bad schema accepted")
    except ReplayValidationError:
        pass


def test_validate_soft_vs_hard(tmp_path: Path) -> None:
    # legacy BUILD form passes soft (shape ok), is flagged in soft problems
    data = make_record_data(0, steps=4)
    data["turns"][2]["action"] = {"farmer": ["BUILD", "COOP"], "hands": [],
                                  "market": []}
    p = tmp_path / "legacy.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    summary = EpisodeRecord.validate(p, mode="soft")
    assert summary["n_turns"] == 4
    assert any("legacy BUILD" in prob for prob in summary["problems"])

    # unknown unit op: hard raises with a precise location
    data2 = make_record_data(1, steps=4)
    data2["turns"][1]["action"] = {"farmer": ["FLY"], "hands": [], "market": []}
    p2 = tmp_path / "unknown_op.json"
    p2.write_text(json.dumps(data2), encoding="utf-8")
    try:
        EpisodeRecord.validate(p2, mode="hard")
        raise AssertionError("unknown op accepted in hard mode")
    except ReplayValidationError as exc:
        assert "turns[1]" in str(exc) and "FLY" in str(exc)

    # soft accepts it (shape-wise it's a list op)
    summary = EpisodeRecord.validate(p2, mode="soft")
    assert summary["n_turns"] == 4


def test_validate_rejects_bad_structure(tmp_path: Path) -> None:
    # steps must strictly increase
    data = make_record_data(0, steps=3)
    data["turns"][2]["step"] = 1
    p = tmp_path / "order.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    try:
        EpisodeRecord.validate(p, mode="soft")
        raise AssertionError("non-increasing steps accepted")
    except ReplayValidationError as exc:
        assert "strictly increasing" in str(exc)

    # hard mode: last step >= episodeSteps
    data = make_record_data(0, steps=5)  # steps 0..4, episodeSteps=5
    data["configuration"]["episodeSteps"] = 4
    p = tmp_path / "range.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    try:
        EpisodeRecord.validate(p, mode="hard")
        raise AssertionError("out-of-range step accepted")
    except ReplayValidationError as exc:
        assert "episodeSteps" in str(exc)

    # PLANT with an unknown crop — hard raises
    data = make_record_data(0, steps=2)
    data["turns"][0]["action"] = {"farmer": ["PLANT", "PINEAPPLE"],
                                  "hands": [], "market": []}
    p = tmp_path / "crop.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    try:
        EpisodeRecord.validate(p, mode="hard")
        raise AssertionError("unknown crop accepted")
    except ReplayValidationError as exc:
        assert "PLANT" in str(exc)


# --------------------------------------------------------------------------- #
# agent behaviour
# --------------------------------------------------------------------------- #

def test_agent_is_seat_agnostic() -> None:
    rec0 = EpisodeRecord(make_record_data(0), source="<t0>")
    rec1 = EpisodeRecord(make_record_data(1), source="<t1>")
    a0, a1 = ReplayAgent(rec0), ReplayAgent(rec1)

    for step in range(0, STEPS, 17):
        obs0 = {"player": 0, "step": step}
        obs1 = {"player": 1, "step": step}
        # seat 0 and seat 1 observations produce the SAME record action
        assert a0(copy.deepcopy(obs0)) == a0(copy.deepcopy(obs1))
        assert a1(copy.deepcopy(obs0)) == a1(copy.deepcopy(obs1))
        assert a0(copy.deepcopy(obs0)) == scripted_action(step, 0)
        assert a1(copy.deepcopy(obs0)) == scripted_action(step, 1)


def test_missing_step_plays_pass() -> None:
    data = make_record_data(0, steps=10)
    rec = EpisodeRecord(data, source="<t>")
    agent = ReplayAgent(rec)
    # beyond the record -> PASS
    assert agent({"player": 0, "step": 10}) == PASS_ACTION
    assert agent({"player": 1, "step": 999}) == PASS_ACTION
    # gaps inside the record -> PASS: a turn whose action is EXPLICITLY null.
    # The record is already indexed, so rewrite the index entry too (this
    # mirrors what a loader-side mutation of the file would produce).
    rec.turns[5]["action"] = None
    rec._index[5] = None
    assert agent({"player": 0, "step": 5}) == PASS_ACTION
    # 3 missing so far: steps 10, 999, and the nulled 5
    assert agent.missing_turns == 3
    # the earlier calls at steps 10 and 999 hit no record entries
    assert agent.replayed_turns == 0


def test_agent_copy_modes() -> None:
    """copy=False shares the recorded object (default); copy=True deep-copies."""
    data = make_record_data(0, steps=4)
    rec = EpisodeRecord(data, source="<t>")
    recorded = rec.action_at(1)

    # share mode (default): same object identity, zero copies
    shared_agent = ReplayAgent(rec)
    a = shared_agent({"player": 0, "step": 1})
    assert a is recorded

    # copy mode: fresh object every handout, record untouched by mutation
    copy_agent = ReplayAgent(rec, copy=True)
    a1 = copy_agent({"player": 0, "step": 1})
    a2 = copy_agent({"player": 0, "step": 1})
    assert a1 is not recorded and a2 is not recorded and a1 is not a2
    a1["farmer"][0] = "MUTATED"
    a1["market"].append(["SELL", "WHEAT", 999])
    assert rec.action_at(1) == scripted_action(1, 0)
    assert copy_agent({"player": 0, "step": 1}) == scripted_action(1, 0)


def test_engine_never_mutates_shared_actions(tmp_path: Path) -> None:
    """Share mode is safe: neither engine path writes into the handed-out
    action object. Probed here with a shared object across a full episode
    on fast_sim AND the harness (mixed with recorded actions so real ops
    execute)."""
    p0 = write_record(tmp_path, 0)
    p1 = write_record(tmp_path, 1)
    rec0 = EpisodeRecord.load(p0)
    rec1 = EpisodeRecord.load(p1)

    shared = {"farmer": ["PICKUP", "WHEAT", 3],
              "hands": [["WATER"], ["NORTH"]],
              "market": [["SELL", "WHEAT", 2], ["BUY_PRODUCT", "WHEAT", 1]]}
    orig = copy.deepcopy(shared)

    class MixedAgent:
        def __init__(self, record: EpisodeRecord) -> None:
            self._r = record

        def __call__(self, obs, configuration=None):
            if obs["step"] % 3 == 0:
                return shared  # same object every third turn
            a = self._r.action_at(obs["step"])
            return copy.deepcopy(a) if a else dict(PASS_ACTION)

    sim = FastSim({"episodeSteps": STEPS, "seed": SEED})
    sim.run([MixedAgent(rec0), MixedAgent(rec1)])
    assert shared == orig, "fast_sim mutated a submitted action"

    from offline_lab import kaggle_env
    env = kaggle_env.run_episode(
        [MixedAgent(rec0), MixedAgent(rec1)],
        configuration={"episodeSteps": STEPS, "seed": SEED})
    assert shared == orig, "harness mutated a submitted action"
    assert all(env.steps[-1][i].status == "DONE" for i in (0, 1))


# --------------------------------------------------------------------------- #
# bit-exact replay through both engine paths
# --------------------------------------------------------------------------- #

def _replay_via_fast_sim(rec0: EpisodeRecord, rec1: EpisodeRecord,
                         seed: int, steps: int) -> list[float]:
    sim = FastSim({"episodeSteps": steps, "seed": seed})
    sim.run([ReplayAgent(rec0), ReplayAgent(rec1)])
    return sim.rewards()


def _replay_via_harness(rec0: EpisodeRecord, rec1: EpisodeRecord,
                        seed: int, steps: int) -> list[float]:
    from offline_lab import kaggle_env
    env = kaggle_env.run_episode(
        [ReplayAgent(rec0), ReplayAgent(rec1)],
        configuration={"episodeSteps": steps, "seed": seed})
    return [float(env.steps[-1][i].reward) for i in (0, 1)]


def test_bit_exact_replay_both_paths(tmp_path: Path) -> None:
    p0 = write_record(tmp_path, 0)
    p1 = write_record(tmp_path, 1)
    rec0 = EpisodeRecord.load(p0)
    rec1 = EpisodeRecord.load(p1)

    # the reference: the scripted actions run directly on the engine
    sim = FastSim({"episodeSteps": STEPS, "seed": SEED})
    reference = sim.run([lambda obs, pl=0: copy.deepcopy(scripted_action(obs["step"], pl)),
                         lambda obs, pl=1: copy.deepcopy(scripted_action(obs["step"], pl))])

    rewards_fast = _replay_via_fast_sim(rec0, rec1, SEED, STEPS)
    assert rewards_fast == reference, "fast_sim replay diverged"

    rewards_harness = _replay_via_harness(rec0, rec1, SEED, STEPS)
    assert rewards_harness == reference, "harness replay diverged"

    # money must actually have moved (the stream is not all-PASS)
    assert any(r != 3000.0 for r in reference)


def test_both_seats_one_record_concurrent(tmp_path: Path) -> None:
    """One record driving BOTH seats: the same actions from both sides."""
    rec = EpisodeRecord(make_record_data(0), source="<both>")
    agent = ReplayAgent(rec)
    sim = FastSim({"episodeSteps": 48, "seed": 7})
    rewards = sim.run([agent, agent])
    # no crash, numeric rewards (the engine mirrors every action)
    assert all(isinstance(r, float) for r in rewards)
    assert agent.replayed_turns >= 48


def test_free_seed_still_runs(tmp_path: Path) -> None:
    """Same record, different world seed: episode runs, never crashes."""
    p0 = write_record(tmp_path, 0)
    rec = EpisodeRecord.load(p0)
    from kaggle_environments.envs.kaggriculture.kaggriculture import (
        random_agent,
    )
    sim = FastSim({"episodeSteps": 48, "seed": SEED + 1})
    rewards = sim.run([ReplayAgent(rec), random_agent])
    assert all(isinstance(r, float) for r in rewards)


if __name__ == "__main__":
    import tempfile

    failures = 0
    tests = [(k, v) for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        for name, fn in tests:
            try:
                import inspect
                sig = inspect.signature(fn)
                args = []
                if "tmp_path" in sig.parameters:
                    args.append(root / name)
                    args[0].mkdir(parents=True, exist_ok=True)
                fn(*args)
                print(f"PASS {name}")
            except Exception as exc:  # noqa: BLE001
                failures += 1
                print(f"FAIL {name}: {type(exc).__name__}: {exc}")
    raise SystemExit(1 if failures else 0)
