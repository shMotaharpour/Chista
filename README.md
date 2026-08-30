# ChistaAgent

AI Agent for **Kaggriculture** — a two-player farming competition on Kaggle.

## Layout

```
ChistaAgent/
├── agent/      # Agent code (main.py submittable to Kaggle)
├── lab/        # Independent design/optimization environment (free deps: kaggle-environments, ...)
│   ├── opponents/  # 19 vendored top-player opponents (ladder reference)
│   ├── runner.py   # Paired-seed match runner (sides swapped to cancel first-mover advantage)
│   ├── ladder.py   # Opponent strength ranking (sorting tournament w/ binary insertion)
│   ├── prices.py   # Exact port of env market_price() — env-verified
│   ├── economics.py# Crop/animal/land economics tables
│   └── sell_impact.py # Price-crash curves per product
├── docs/       # Project documentation
│   ├── README.md, AGENT.md, kaggriculture-source.md   # Game docs
│   ├── rules/    # Project rules
│   └── research/ # Findings: economics (003), mechanics verification (006),
│                 # opponent ladder (005), roadmap (004), strategy plan (000)
└── .hermes/plans/  # Implementation plans
```

## Two separated environments

1. **Runtime (agent)** — Kaggle Python image. The agent only uses what that image provides (output: `main.py` with `agent(obs)`).
2. **Lab** — on the host, free dependencies. Runs the game env, evaluates, analyzes replays, optimizes strategy.

## Quick start (lab)

```bash
. .venv/bin/activate
python -m lab.runner --a starter --b random --seeds 1..5 --out lab/results
python -m lab.report lab/results/<run-dir>
python -m lab.ladder          # rank the 19 vendored opponents
```
