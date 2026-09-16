# Agent Guide — ChistaAgent

## Rules

- English only in all repo files.
- Never commit directly to `main` — feature branch → PR → squash-merge.
- Every commit by the agent MUST use `~/.local/bin/agent-commit -m "..."`,
  which appends the `Co-authored-by: Hermes (GLM via OpenRouter)
  <hermes@local>` trailer. Never plain `git commit`.

## Findings & rules files (hard rules — always follow)

Research output lives in two kinds of markdown files:

- **Findings**: `F<NNN>_<slug>.md` — `NNN` is a zero-padded number (F001,
  F002, …); slug is kebab/snake case, **max 10 words**.
- **Rules**: `R<NNN>_<slug>.md` — same numbering scheme.

Each file starts with a summary of **max 50 words**. `docs/INDEX.md` lists
them: **rules first, then findings**, one line per file — the filename as a
link to the file, followed by only that file's 50-word summary. IDs are
permanent: never renumber, never reuse. These conventions are hard rules and
must always be respected.

Before writing any simulation, evaluation, or training code, read the rules
files — R002 (never transcribe game rules; import from
`kaggle_environments`), R003 (the simulator wraps the real interpreter, it
never reimplements it), and R004 (validation is a config switch: dev mode
validates, fast mode bypasses) — they dictate how `world/` must be used.

R005 (every number has a source) binds wider than `world/`: it applies to every
threshold, budget and constant written anywhere in this project, including
issues and commit messages. A number that cannot name a finding, the engine, or
a measurement someone recorded is a defect, not a detail.

R007 (a guard must be seen to fail) binds every test that claims to cover a
bug: re-introduce the defect, watch the test go red, put it back, and say in
the PR which bug you broke and what the failure said. Five guards in this
repository passed while guarding nothing — each exercised a path the production
code never takes, and every one survived review by reading.

## Environment

Python 3.11 venv at `.venv/`, CPU-only PyTorch, built from `requirements.txt`:

`requirements.txt` lists the core stack (kaggle-environments, ortools, scipy,
torch CPU) plus analysis/plotting extras.

## Game environment access — `world/`

All interaction with the game goes through the wrappers in `world/` (see
`world/README.md`). Do not call `kaggle_environments` or the interpreter
ad hoc elsewhere; import one of the two paths:

- **`world.kaggle_env`** — the full, official path. Real harness
  (`make()`), any configuration, agents passed at call time, HTML replay
  via the environment's own `render()`. Outputs go to `artifacts/replays/`
  (gitignored, in-repo) unless another `output_path` is passed. Use for
  submission-style evaluation and replays.
- **`world.fast_sim`** — the fast path. `FastSim` drives
  `kaggriculture.interpreter()` directly on structify-cloned state:
  `step()` for single turns, `run()` for episodes, `clone()`/`what_if()`
  for hypothetical branches (DP/MDP), and `run_parallel()` for multi-core
  sweeps (RL/evaluation). Construct with `validate="dev"` while debugging
  and `validate="fast"` (default) for bulk runs — R004.

Parity contract: with the same seed and the same action sequence, both paths
produce bit-identical money and bit-identical per-turn observations and final
rewards for both agents — enforced by `tests/test_world_parity.py`, which
compares the full agent-facing stream. If a change breaks that, the change is
wrong.

## Vocabulary

**The arena** — the evaluation loop that plays our agent against the vendored
competitors in `opponents/` over paired seeds, and reports win rate and coin
margin. It is `offline/evaluate.py` (issue #18) driving `offline/pool/`
(issue #20).

**It does not exist yet.** Both are specified and neither is built, so every
sentence in this repo that says "the arena runs this in our interpreter" is
describing a contract the code must honour once it is written, not something
that happens today. Read those as requirements.

Two things it is not:

- It is not the game harness. `world/` reserves *harness* for
  `kaggle_environments` itself — "the real harness (`make()`)" — and the two
  must not be called the same thing.
- It is not a Kaggle leaderboard. Only Kaggle scores the competition; the arena
  is our own measurement, and its numbers are ours alone.

The word arrived with the vendored `opponents/` directory, which came from
**AgriOracle**, this project's predecessor, where it was a real module at
`lab/eval/arena.py` (still cited in R003's evidence table). It travelled into
Chista's own tests and docs before anything here answered to it. It stays
because it is a good name for the thing, and now it has a referent — which is
the point of this section, and the reason it is here rather than assumed.

## Docs

- `docs/player_agent.md` — getting started: build, test, and submit an agent.
- `docs/README.md` — full game rules (crops, animals, market, town).
- `docs/kaggriculture-source.md` — environment source notes.
- `docs/INDEX.md` — one-line index of all rules and findings files.
