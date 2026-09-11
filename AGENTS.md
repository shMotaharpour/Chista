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

## Environment

Python 3.11 venv at `.venv/`, CPU-only PyTorch, built from `requirements.txt`:

`requirements.txt` lists the core stack (kaggle-environments, ortools, scipy,
torch CPU) plus analysis/plotting extras.

## Docs

- `docs/player_agent.md` — getting started: build, test, and submit an agent.
- `docs/README.md` — full game rules (crops, animals, market, town).
- `docs/kaggriculture-source.md` — environment source notes.
- `docs/INDEX.md` — one-line index of all rules and findings files.
