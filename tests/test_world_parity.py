"""Parity test: the agent-facing trajectory must equal the real harness.

Run:  .venv/bin/python -m tests.test_world_parity
      (also works under pytest: pytest tests/test_world_parity.py)

Method — the executable form of the AGENTS.md / R003 parity contract. The
primary comparison is the OBSERVATION STREAM: every observation each agent is
handed, at every decision turn, on both paths, with the same seed and the same
action sequence. That is the surface a policy actually consumes, and it is the
only comparison that catches the observation-aliasing bug class (a money-only
check cannot see it).

Secondary comparisons: the recorded end-of-episode state field-by-field, the
final rewards, and the number of decision turns.

The action sequence is deliberately messy (unknown ops, illegal moves, hires,
land, animals) so the engine's silent-failure paths are crossed too.
"""

from __future__ import annotations

import json
import random
from typing import Any

from kaggle_environments import make

from world.fast_sim import FastSim

STEPS = 96
SEED = 4242
WEED_CHANCE = 0.08  # default is 0.005: raised so the F045 weed->market RNG
                    # coupling is definitely exercised, not just read about

UNIT_OPS = [["PASS"], ["NORTH"], ["SOUTH"], ["EAST"], ["WEST"], ["WATER"],
            ["HARVEST"], ["DIG"], ["FERTILIZE"], ["BUILD", "COOP"],
            ["BUILD", "PASTURE"], ["PLANT", "WHEAT"], ["PLANT", "CARROT"],
            ["PLANT", "MELON"]]
MARKET_OPS = [["SELL", "WHEAT", 1], ["BUY_SEED", "WHEAT", 1], ["HIRE"],
              ["BUY_LAND"], ["SELL", "CARROT", 1], ["BUY_ANIMAL", "GOOSE"],
              ["BUY_PRODUCT", "WHEAT", 1], ["NONSENSE"]]


def action_for(step: int, player: int) -> dict[str, Any]:
    """Deterministic; identical for both simulations, no shared RNG state."""
    rng = random.Random(f"chista-parity-{step}-{player}")
    return {
        "farmer": list(rng.choice(UNIT_OPS)),
        "hands": [list(rng.choice(UNIT_OPS)) for _ in range(2)],
        "market": [list(rng.choice(MARKET_OPS)) for _ in range(rng.randint(0, 3))],
    }


def _clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    return value


def dump(value: Any) -> str:
    return json.dumps(_clean(value), sort_keys=True, default=str)


def config() -> dict[str, Any]:
    return {"seed": SEED, "episodeSteps": STEPS, "weedSpawnChance": WEED_CHANCE}


def drive_both_paths() -> dict[str, Any]:
    """Run the same actions through the harness and through fast_sim.

    Returns the observation streams each agent received, the final rewards and
    the two state containers.
    """
    actions = {p: [action_for(i, p) for i in range(STEPS + 2)] for p in (0, 1)}
    harness_seen: dict[int, list[Any]] = {0: [], 1: []}
    sim_seen: dict[int, list[Any]] = {0: [], 1: []}

    def harness_agent(player: int) -> Any:
        def agent(obs: Any) -> dict[str, Any]:
            harness_seen[player].append(_clean(dict(obs)))
            return actions[player][len(harness_seen[player]) - 1]
        return agent

    def sim_policy(player: int) -> Any:
        def policy(obs: dict[str, Any]) -> dict[str, Any]:
            sim_seen[player].append(_clean(obs))
            return actions[player][len(sim_seen[player]) - 1]
        return policy

    env = make("kaggriculture", configuration=config(), debug=True)
    rewards = env.run([harness_agent(0), harness_agent(1)])

    sim = FastSim(config(), validate="fast")
    sim_rewards = sim.run([sim_policy(0), sim_policy(1)])
    return {"env": env, "sim": sim, "harness_seen": harness_seen,
            "sim_seen": sim_seen, "rewards": rewards, "sim_rewards": sim_rewards}


def test_observation_stream_parity() -> None:
    run = drive_both_paths()
    h_seen, f_seen = run["harness_seen"], run["sim_seen"]
    for player in (0, 1):
        assert len(h_seen[player]) == len(f_seen[player]), (
            f"agent {player}: {len(h_seen[player])} harness turns vs "
            f"{len(f_seen[player])} fast_sim turns")
        assert len(h_seen[player]) > STEPS - 3, "episode ended unexpectedly early"
        for turn, (h_obs, f_obs) in enumerate(zip(h_seen[player], f_seen[player])):
            if dump(h_obs) != dump(f_obs):
                diff = [k for k in set(h_obs) | set(f_obs)
                        if dump(h_obs.get(k)) != dump(f_obs.get(k))]
                raise AssertionError(
                    f"agent {player}, turn {turn}: observation fields differ: {diff}\n"
                    f"  harness : {dump({k: h_obs.get(k) for k in diff})[:300]}\n"
                    f"  fast_sim: {dump({k: f_obs.get(k) for k in diff})[:300]}")


def test_final_rewards_and_recorded_state_parity() -> None:
    run = drive_both_paths()
    env, sim = run["env"], run["sim"]
    # env.run() returns the per-step reward matrix, whose entries are agent
    # state objects; take each agent's final payout.
    def payout(entry: Any) -> float:
        return float(getattr(entry, "reward", entry))

    harness_final = [payout(r) for r in run["rewards"][-1]]
    sim_final = [payout(r) for r in run["sim_rewards"]]
    assert harness_final == sim_final, (
        f"final rewards differ: harness={harness_final} fast_sim={sim_final}")

    # Recorded state: the framework stamps `step` on state[0] only (core.py:626),
    # so agent 1's recorded observation has no `step` key. Its act-time
    # observation does carry `step` for both agents, and that is what
    # test_observation_stream_parity compares.
    for player in (0, 1):
        h_obs = dict(env.state[player].observation)
        f_obs = dict(sim.state[player].observation)
        for key in ("day", "hour", "player", "farms", "private", "market", "town"):
            assert dump(h_obs.get(key)) == dump(f_obs.get(key)), (
                f"state[{player}].{key} differs\n"
                f"  harness : {dump(h_obs.get(key))[:300]}\n"
                f"  fast_sim: {dump(f_obs.get(key))[:300]}")
    assert dump(dict(env.state[0].observation)) == dump(dict(sim.state[0].observation)), \
        "state[0] differs field-for-field"

    weeds = sum(1 for row in sim.state[0].observation.farms[0]["tiles"]
                for tile in row if tile is not None and "WEED" in str(tile).upper())
    assert weeds > 0, ("no weed spawned: the occupancy-dependent RNG stream that "
                       "F045 couples to the market was never exercised")


def test_dev_and_fast_paths_agree() -> None:
    cfg = {"seed": 7, "episodeSteps": 48, "weedSpawnChance": 0.02}

    def policy(obs: dict[str, Any]) -> dict[str, Any]:
        return {"farmer": ["WATER"], "hands": [[]], "market": [["SELL", "WHEAT", 1]]}

    fast = FastSim(dict(cfg), validate="fast")
    fast.run([policy, policy])
    dev = FastSim(dict(cfg), validate="dev")
    dev.run([policy, policy])
    assert fast.money() == dev.money(), (fast.money(), dev.money())
    assert dump(fast.state) == dump(dev.state)


def test_observation_mutation_is_guarded_in_dev() -> None:
    """A policy must not be able to write into the world through its observation.

    The harness deep-copies the observation per agent, so mutation is impossible
    there; in fast mode these dicts ARE the state, so the dev guard exists to
    catch the bug before it silently corrupts a sweep.
    """
    cfg = {"seed": 11, "episodeSteps": 48}

    def cheater(obs: dict[str, Any]) -> dict[str, Any]:
        obs["farms"][1]["money"] = 0
        obs["private"]["seeds"]["WHEAT"] = 50
        return {"farmer": ["PASS"], "hands": [[], []], "market": []}

    dev = FastSim(dict(cfg), validate="dev")
    try:
        dev.run([cheater, cheater])
    except RuntimeError as exc:
        assert "mutated observation" in str(exc), exc
    else:
        raise AssertionError("dev guard did not fire on a mutating policy")
    assert dev.state[0].observation.farms[1]["money"] != 0, \
        "the mutating policy reached the episode state in dev mode"

    fast = FastSim(dict(cfg), validate="fast")
    fast.run([cheater, cheater])
    assert fast.state[0].observation.farms[1]["money"] == 0, (
        "fast mode is documented as live views; if that ever stops being true, "
        "the dev guard must be re-checked")


def test_rewards_are_numeric_mid_episode() -> None:
    """The harness reports 0 for a non-final agent; accumulating rewards must work."""
    sim = FastSim({"seed": 3, "episodeSteps": 24}, validate="fast")
    sim.reset()
    assert sim.rewards() == [0.0, 0.0]
    sim.step([{"farmer": ["PASS"], "hands": [[], []], "market": []},
              {"farmer": ["PASS"], "hands": [[], []], "market": []}])
    assert all(isinstance(r, float) for r in sim.rewards())
    assert sum(sim.rewards()) == 0.0


def test_run_parallel_returns_one_record_per_episode() -> None:
    from world.fast_sim import run_parallel

    cfg = {"episodeSteps": 48, "weedSpawnChance": 0.05}
    first = run_parallel(4, configuration=cfg, processes=2, master_seed=1)
    assert len(first) == 4, f"expected 4 records, got {len(first)}"
    assert len({r["seed"] for r in first}) == 4, "episodes must have distinct seeds"
    for record in first:
        assert set(record) == {"seed", "rewards", "money", "steps"}, record
        assert record["seed"] is not None
    second = run_parallel(4, configuration=cfg, processes=2, master_seed=1)
    assert dump(first) == dump(second), "master_seed must make the sweep reproducible"


def main() -> int:
    failures = 0
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception as exc:  # noqa: BLE001 - report every failure
                failures += 1
                print(f"FAIL {name}: {type(exc).__name__}: {exc}")
    print(f"\n{'all tests passed' if not failures else f'{failures} test(s) failed'}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
