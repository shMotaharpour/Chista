# R003 — Simulator wraps the real interpreter, it does not reimplement it

**Summary (≤50 words):** Fast game simulation must call
`kaggle_environments.envs.kaggriculture.kaggriculture.interpreter()` directly
on a real, structify-cloned state — never reimplement the game rules.
Measured 8.2× faster than `env.run()` with bit-identical rewards, and no
second rule implementation to keep in sync.

## Decision

Everywhere that needs a fast game simulator calls
`kaggle_environments.envs.kaggriculture.kaggriculture.interpreter()` directly
on a real, structify-cloned state, skipping only the harness bookkeeping
around it (`Environment.step()`'s per-turn JSON-schema validation, its
`redirect_stdout`/`redirect_stderr` context manager, and appending a full
snapshot to `self.steps` every turn for the whole episode). It is not a
second implementation of the game's rules.

## Why

"If wrapping the real interpreter with a cheap state clone is fast enough
for offline sweeps, a reimplementation is pure risk. Measure
episodes/second first." So it was measured before building either option,
3 episodes each, playbook vs pass:

| path | s/episode |
|---|---:|
| `env.run()` (the harness path `lab/eval/arena.py` already uses) | 5.20 |
| `agrioracle.sim.run_episode()` (interpreter called directly) | 0.61 |

8.2× faster, with the exact reward the harness produced at the same seed
(127,425.0000 == 127,425.0000) — not approximately the same, the same
number, because it is the same code computing it, per R002.

A reimplementation was the other option on the table. That buys nothing
here that wrapping doesn't already have, and costs a real, ongoing risk:
two implementations of weed spawning, market pricing, plant/animal decay,
and shed capacity that must be kept in agreement by hand — the same failure
mode R002 already rejected once for the rules constants themselves ("a
copied table can disagree with the game — silently").

## Risk accepted

This module is exactly as correct as `kaggriculture.interpreter()` is —
which is to say, exactly as correct as the real competition, by
construction — but it inherits every private-API dependency R002 already
accepted (`interpreter`, `structify`, the `Environment` class's
state/reset shape). A kaggle-environments version bump that changes
`Environment.step()`'s own bookkeeping (not just the rules) could silently
desync the simulator from the harness path; the mitigation is the same
version pinning R002 accepted.
