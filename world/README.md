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
| playing back a recorded episode file | `world.replay_agent.ReplayAgent` |

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
- HTML output defaults to `artifacts/replays/` inside the repo (gitignored);
  the name carries the seed, the step count and a state fingerprint, so two
  episodes sharing a seed no longer overwrite each other's replay. Pass
  `output_path=` to override.
- `env.steps`, `env.rewards`, `env.info["seed"]` carry the full record.

## world.fast_sim — fast path (R003)

Drives `kaggriculture.interpreter()` directly on structify-cloned state,
skipping the harness bookkeeping around it. Measured here (720-step season,
PASS policies, 8-core box): 0.035 s/episode vs 1.785 s for `env.run()` =
**50.6x** (32.3x against the harness with its agent processes removed).
Reproduce with `.venv/bin/python -m bench.bench_paths` — the numbers are
machine-specific, so the script prints platform, cpu_count and the installed
kaggle-environments version.

```py
from world.fast_sim import FastSim, run_parallel

sim = FastSim({"episodeSteps": 720, "seed": 42})   # validate="fast" default
sim.step([action0, action1])                        # one turn
obs = sim.observations()                            # per-agent obs dicts
rewards = sim.run([policy0, policy1])               # full episode

branch = sim.what_if(action, horizon=48)            # hypothetical branch;
                                                    # sim itself unchanged
twin = sim.clone()                                  # independent copy point

results = run_parallel(100, configuration={})       # one record per episode
                                                    # (fresh seed each)
```

- **validate switch (R004):** `FastSim(..., validate="dev")` checks action
  shapes and state invariants on every step; `validate="fast"` (default)
  bypasses all checks for speed-critical runs. Re-run the same inputs in
  dev mode to re-validate any fast result.
- **Parity contract:** same seed + same action sequence ⇒ bit-identical
  money, per-turn observations and final rewards across both paths, enforced
  by `tests/test_world_parity.py` (the full agent-facing observation stream of
  both agents, every turn — not just money). If it breaks, the change is wrong.
- **Observation contract:** `observations()` returns detached deep copies in
  dev mode and LIVE views in fast mode. Fast callers must treat observations
  as read-only: writing into one mutates the episode — the harness never allows
  that (each agent gets its own copy), so a policy that writes to its
  observation can zero the opponent's money or grant itself free seeds. Dev
  mode hands out copies and raises if a policy mutates one. Cost of the copies:
  +265 us/turn, i.e. 0.270 s instead of 0.034 s for a 720-step season.
- **`run_parallel`:** returns one record per episode (`seed`, `rewards`,
  `money`, `steps`) sorted by seed, so a fixed master seed is reproducible.
  Worker policies must be importable at module level (they are pickled — no
  closures). It is a throughput helper, not an evaluator: with one fixed seed
  every record is the same episode; only the default `seed=None` draws a fresh
  seed per episode.
- **Branch purity / RNG invariant:** a `clone()` continued with a suffix equals
  a from-scratch replay of prefix+suffix, and exploration never touches the
  parent (`tests/test_world_branch_purity.py`). This holds because the
  interpreter rebuilds its RNG per day from `(seed, day)`; if a dependency bump
  ever moves it to a single advancing stream, these tests fail instead of every
  DP/what-if branch going quietly wrong.
- RNG note: the interpreter seeds weed/shop draws per day from the episode
  seed — a cloned sim reproduces the original's future exactly unless you
  change actions (see F045: planting changes tomorrow's prices).

## world.replay_agent — playing back recorded episodes

`EpisodeRecord` loads a single-agent episode file (schema
`chistaagent.replay.v1`, format documented in
[REPLAY_SCHEMA.md](REPLAY_SCHEMA.md)); `ReplayAgent(record)` is a callable
policy that plays it back in any seat, on either engine path. Missing steps
play PASS; the record is never mutated by the engine (deep copies handed out).
Validation is a class method (`EpisodeRecord.validate(path, mode="soft"|"hard")`)
run only on demand — never during an episode (R004).

```py
from world.replay_agent import EpisodeRecord, ReplayAgent

rec = EpisodeRecord.load("my_episode.json")   # validates first
agent = ReplayAgent(rec)
env = kaggle_env.run_episode(["starter", agent],
                             configuration={"seed": rec.seed,
                                            **rec.configuration})
```

Seat-agnostic by construction: the file carries no player identity; the agent
picks the action purely by `obs["step"]`, so the same record plays at seat 0,
seat 1, or both seats concurrently. Bit-exactness of a full-match replay
requires a record per seat, both replayed against each other with the match's
seed (covered by `tests/test_replay_agent.py`).

