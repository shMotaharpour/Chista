# Agent Guide — ChistaAgent

## Rules

- English only in all repo files.
- Never commit directly to `main` — feature branch → PR → squash-merge.
- Every commit by the agent MUST use `agent-commit-trailer`. Never plain `git commit`.
- Use world definitios for uniform naming.
- The submission is `agent/`, self-contained: everything the entry point loads
  lives inside it (`agent/world/` for the definitions, `agent/artifact/` for
  the model artifacts, and one folder per layer). Nothing inside `agent/` may import
  from outside it. `offline_lab/` and `tests/` import `agent/` as they need, and the
  builders write their artifacts into `agent/`.

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

**One vocabulary, one belief.** `docs/ARCHITECTURE.md` is the system's shape:
the canonical names live in `world/model.py` (engine-derived, never typed), the
DP's chains and the WSR's major tasks are the same object with one expansion
(`world/model.py::compile_chain`), and `belief/` is the only reader of the market
and the rival. Before adding a name, a set or a market read, read that document —
and if a module disagrees with it, the module is the bug.

## Environment

Python 3.11 venv at `.venv/`, CPU-only PyTorch, built from `requirements.txt`:

`requirements.txt` lists the core stack (kaggle-environments, ortools, scipy,
torch CPU) plus analysis/plotting extras.

## Game environment access — `offline_lab/`

All interaction with the game goes through the wrappers in `offline_lab/` (see
`offline_lab/README.md`). `world/` is the reference of definitions and holds no tools.
Do not call `kaggle_environments` or the interpreter ad hoc elsewhere; import one of
the two paths:

- **`offline_lab.kaggle_env`** — the full, official path. Real harness
  (`make()`), any configuration, agents passed at call time, HTML replay
  via the environment's own `render()`. Outputs go to `artifacts/replays/`
  (gitignored, in-repo) unless another `output_path` is passed. Use for
  submission-style evaluation and replays.
- **`offline_lab.fast_sim`** — the fast path. `FastSim` drives
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
margin. It is `offline_lab/evaluate.py` (issue #18) driving `offline_lab/pool/`
(issue #20).

`offline_lab/evaluate.py` exists (issue #18): paired-seed comparison of two
versions, paired coin margin as the development signal and win rate against
the pool as the ship gate, ties counted explicitly, loss autopsy, a measured
seed-count line, and a serial `--timing` path for the F046 budget. Opponent
selection consumes the measured dev/held-out split from
`offline_lab/pool/registry.py` (#20: 11 dev / 5 held-out over the canonical 16 —
amendment A dropped 3 byte-identical duplicates); the sorted-prefix fallback
fires only if the registry module is missing, and every report names which
selection ran. The ship gate's win rate is seat-0-only (measured seat effect
in the module docstring).

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

## The sell side — `belief/`

`belief/` holds what the market and the rival are doing, and what to sell (issue
#54). It is the only place that reads the market's own arithmetic:

- `tracker.py` — the flow residual. Exact for the seven goods that cannot be
  bought; **net-only** for WHEAT and FERTILIZER, because a buy is quoted at
  `price(I-1)` and a sale earns `price(I)`, so both channels enter the identity
  with the same sign. Its docstring carries the measured error.
- `opponent.py` — the rival's action counts, the closed-form demand forecast
  (already-open shops are facts; only the future unlocks are random, so both
  moments are exact without sampling), and the inference of the order they filled
  their ten slots in, from their realised average price.
- `solvers.py` — the slot game's exact maximin mix (`linprog(method="highs")`) and
  its risk-adjusted continuous sibling (`minimize(method="SLSQP")`), plus the
  season LP whose absorption row is what stops it selling 4,000 units into a
  525-unit drain.
- `schemas.py` — the messages the layers exchange. Nobody commands an op across a
  boundary: a message says what must be true and by when, and the receiving layer
  owns how.
- `stubs.py` — what runs until each of those units exists, each naming the issue
  that retires it.

## Docs

- `docs/player_agent.md` — getting started: build, test, and submit an agent.
- `docs/README.md` — full game rules (crops, animals, market, town).
- `docs/kaggriculture-source.md` — environment source notes.
- `docs/INDEX.md` — one-line index of all rules and findings files.
- `docs/ARCHITECTURE.md` — **the** system shape: layers and ownership, the one
  vocabulary (`world/model.py`), the DP ↔ WSR alignment, the one belief
  (`belief/`), and the migration order that retires the duplicates.
- `world/model.py` — the canonical, engine-derived vocabulary every layer imports.

## Games run on FastSim — MANDATORY
Every season, episode or match in this repo runs on `offline_lab/fast_sim.py`, never
on the kaggle harness, unless the owner says otherwise. The harness caps a turn at
one second (`kaggriculture.json`: `"actTimeout": 1`), so a plan's round count follows
the machine's speed and two runs of one seed disagree. `run_episode` stays for the
submission-path checks the owner asks for by name.

## Debugging: ask the graph before you grep — `graphify-out/`

This repo carries a knowledge graph of its own code and docs, built with
`graphify` (a uv tool: `uv tool install graphifyy`; the interpreter it runs is
recorded in `graphify-out/.graphify_python`). Its outputs are TRACKED here, not
gitignored: `graphify-out/graph.json`, `graph.html` (opens without a server) and
`GRAPH_REPORT.md`. Rebuild or extend with `graphify <repo> --code-only` (code
only — the doc pass wants an LLM key) and `--update` to fold the docs back in
from the cache; the rebuildable parts (`cache/`, dated backups, the raw
`.graphify_*` intermediates) stay gitignored.

The rule: for a debugging question — what calls this, what does a change here
ripple into, which layer owns this name — read the graph FIRST and grep second.
It answers "what touches this" across layers in one shot, which grep cannot: the
god nodes (`FastSim`, `forecast()`, `TileGraph`, `compile_route()`, `Config`) are
exactly the junctions every change propagates through. Query `graphify-out/graph.json`
with `graphify query` / `path` / `explain`; `docs/ARCHITECTURE.md` still owns the
INTENDED shape while the graph shows the realised one.

This is a development tool, not a runtime dependency: nothing in `agent/` imports
it, so it does not belong in `requirements.txt`.
