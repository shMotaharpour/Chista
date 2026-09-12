# world/

This folder holds the **wrappers around and modifications to the original
game environment**, shaped so the rest of the project can reach it more
efficiently. It does not reimplement the game — per R002/R003, the shipped
`kaggle_environments` interpreter is the single source of truth and is
always the executor.

Two access paths live here:

- `kaggle_env.py` — the **full Kaggle path**: the real `make()` environment
  with any custom configuration, agents supplied at call time, episodes run
  inside the genuine Kaggle harness, and the game rendered to an HTML file
  via the environment's own `render` method.
- `fast_sim.py` — the **fast path**: bypasses the Kaggle harness wrappers
  (`Environment.step()`'s per-turn JSON-schema validation, stdout/stderr
  redirection, and full per-turn snapshot appends) and drives
  `kaggriculture.interpreter()` directly on structify-cloned state. It
  provides single-step simulation, what-if branching from snapshots, and
  optional process-parallel episode sweeps for DP / MDP / RL analyses.

Both paths share one configuration object and honor R004: a `validate`
switch with `dev` mode (all checks on) and `fast` mode (checks bypassed for
speed-critical runs).
