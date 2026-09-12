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
