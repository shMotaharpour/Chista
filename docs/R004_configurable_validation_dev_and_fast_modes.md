# R004 — Validation is a config switch: dev mode validates, fast mode bypasses

**Summary (<=50 words):** Every piece of code carries a validation
configuration. The development mode runs the full harness and validators;
the fast mode — the main parse-and-run path for heavy processes (DP, RL,
MDP) — bypasses every check via the same config, so no time is wasted on
validation when algorithm speed matters.

## Decision

Every module ships with a validation configuration switch with two modes:

- **dev (default)** — the harness and validators run: every check executes.
- **fast** — the heavy-compute mode, used for the main parse-and-run: with
  this config set, all tests and validations on the hot path are bypassed.

The bypass is a single configuration value, not scattered `if` statements:
validators guard themselves with the flag, so the same code serves both
modes without maintaining two versions.

## How to use it (project wiring)

- `world.FastSim(sim, validate="dev" | "fast")` is the reference
  implementation of the switch. Construct with `validate="dev"` while
  debugging; `validate="fast"` (the constructor default) for bulk runs.
- In dev mode the action shapes and state invariants are checked on every
  step; in fast mode they are skipped entirely.
- Dev mode is deliberately *stricter* than the real harness: the harness only
  requires an action to be an object (no typed properties in its schema, and
  the interpreter no-ops unknown ops), so `{"hands": "nope"}` or a junk op
  string passes silently there while dev raises. A dev failure therefore means
  "caller bug", not necessarily "would fail on Kaggle". The useful direction
  still holds: a run that passes dev is a run the harness accepts, because it
  is the same code doing the checking (R002).
- In dev mode `FastSim.observations()` returns detached deep copies and raises
  if a policy mutates one. The harness gives every agent its own copy, so
  writing into an observation is relying on behavior that does not exist on the
  evaluation platform (it can zero the opponent's money or grant free seeds).
  Cost, measured on a 720-step season: +265 us per turn, i.e. 0.270 s in dev
  vs 0.034 s in fast — still far below the harness.
- A fast run's result is always re-checkable: re-run the same inputs with
  `validate="dev"` and compare.
- New wrappers and analysis modules in this repo follow the same pattern:
  one `validate` config value, validators self-guarded by the flag — never
  a second, check-free copy of the code.

## Why

The same code is used twice with different budgets. While developing,
correctness checking *is* the point. While running the heavy processes —
DP, RL, MDP sweeps, self-play — the main run executes thousands of episodes
and validation overhead is pure waste. Making validation a property of
configuration, not of code edits, means speed mode is entered deliberately
per run and can never silently become the default.

## Risk accepted

In fast mode the code runs unchecked: invalid actions, malformed states, and
contract violations pass silently and surface only as wrong numbers
downstream. The mitigation is deliberate entry — fast mode is selected per
run through config — and re-validation: any fast run's result can be
re-checked by rerunning the same inputs in dev mode.

Fast mode also hands out LIVE observation views, so a policy that writes into
an observation mutates the episode; dev mode's copies and its mutation guard do
not apply. The mitigation is the same deliberate entry plus the read-only
contract stated in `agent/world/README.md` — a fast-mode policy must treat the
observation it is handed as read-only. The copies dev mode makes cost
+265 us/turn (0.270 s vs 0.034 s per 720-step season), which is why "fast"
stays the default for sweeps.
