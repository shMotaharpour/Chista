# Agent Guide — ChistaAgent

Workspace of the "Chista" Telegram topic (thread 123): the Kaggriculture
competition agent project (Kaggle). The parent folder `~/Chista` holds shared
rules; this folder is the project root.

## Rules

- English only in all repo files; Persian is for chat replies only.
- Never commit directly to `main` — feature branch → PR → squash-merge.
- Every commit by the agent MUST use `~/.local/bin/agent-commit -m "..."`.
  Never plain `git commit`.

## Environment

Python 3.11 venv at `.venv/`, CPU-only PyTorch, built from `requirements.txt`:

```bash
uv venv .venv --python 3.11
uv pip install --python .venv/bin/python "torch==2.14.0+cpu" \
    --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv/bin/python -r requirements.txt
```

`requirements.txt` lists the core stack (kaggle-environments, ortools, scipy,
torch CPU) plus analysis/plotting extras.

## Docs

- `docs/AGENT.md` — getting started: build, test, and submit an agent.
- `docs/README.md` — full game rules (crops, animals, market, town).
- `docs/kaggriculture-source.md` — environment source notes.
