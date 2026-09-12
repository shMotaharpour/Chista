# world/

This folder holds the **wrappers around and modifications to the original
game environment**, shaped so the rest of the project can reach it more
efficiently. It does not reimplement the game — per R002/R003, the shipped
`kaggle_environments` interpreter is the single source of truth and is
always the executor.

## Which wrapper to use

| need | use |
|---|---|
| submission-style evaluation, HTML replay of a full game | `world.kaggle_env` |
| single-step simulation, what-if branches (DP/MDP), parallel sweeps (RL) | `world.fast_sim` |
| debugging a new policy | `world.fast_sim` with `validate="dev"` |
| bulk runs (thousands of episodes) | `world.fast_sim` with `validate="fast"` (default) |

Rule of thumb: **evaluation → kaggle_env, computation → fast_sim.** Any
result used for a decision should be reproducible through the full path.

## world.kaggle_env — full official path

The genuine Kaggle harness (`make()`), schema-validated actions, timeout
accounting, per-turn snapshots.

```py
from world import kaggle_env

# run an episode with any configuration; agents passed at call time
env = kaggle_env.run_episode([my_agent, "random"],
                             configuration={"episodeSteps": 720, "seed": 42})

# render the game to an HTML file via the environment's own render()
env, html_path = kaggle_env.run_and_render([my_agent, "random"],
                                           configuration={"seed": 42})
```

- `configuration` accepts any key of the game's schema (episodeSteps,
  boardSize, startingMoney, marketParams, seed, ...).
- HTML output defaults to `/tmp/chistaagent/replays/` (ephemeral); pass
  `output_path=` for a persistent location.
- `env.steps`, `env.rewards`, `env.info["seed"]` carry the full record.

## world.fast_sim — fast path (R003)

Drives `kaggriculture.interpreter()` directly on structify-cloned state,
skipping only the harness bookkeeping (schema validation per turn,
stdout/stderr redirection, full-episode snapshot appends). Measured ~8x
faster than `env.run()` with bit-identical rewards.

```py
from world.fast_sim import FastSim, run_parallel

sim = FastSim({"episodeSteps": 720, "seed": 42})   # validate="fast" default
sim.step([action0, action1])                        # one turn
obs = sim.observations()                            # per-agent obs dicts
rewards = sim.run([policy0, policy1])               # full episode

branch = sim.what_if(action, horizon=48)            # hypothetical branch;
                                                    # sim itself unchanged
twin = sim.clone()                                  # independent copy point

results = run_parallel(100, configuration={"seed": None})  # multi-core sweep
```

- **validate switch (R004):** `FastSim(..., validate="dev")` checks action
  shapes and state invariants on every step; `validate="fast"` (default)
  bypasses all checks for speed-critical runs. Re-run the same inputs in
  dev mode to re-validate any fast result.
- **Parity contract:** same seed + same action sequence ⇒ bit-identical
  money across both paths. Verified at commit time; if it breaks, the
  change is wrong.
- RNG note: the interpreter seeds weed/shop draws per day from the episode
  seed — a cloned sim reproduces the original's future exactly unless you
  change actions (see F045: planting changes tomorrow's prices).
