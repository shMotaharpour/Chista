# ChistaAgent Project Rules

## Runtime
- The agent must run on the Kaggle Python image (V1.63); only libraries available in that image.
- The lab is independent and may use any Python library.
- Final submission output for Kaggle: `main.py` at the root with an `agent(obs)` function.

## Game rules (summary)
- Two players, 720 turns (24 turns × 30 days), winner = most money in bank.
- Per turn: one farmer action + hand actions + up to 10 market orders.
- Plants must be watered daily (2 unwatered days = weed); animals fed wheat daily (2 days = escape).
- Sell prices are dynamic: selling lowers the price — premium products (strawberry, melon, milk, wool) crash hard on oversupply.
- Harvest the moment yield > 0: decay is 1 unit per 2 TURNS after max lifespan (see docs/research/006).
- Animals cannot be resold — animal purchase is a sunk, durable investment.
- Fertilizer only helps when applied inside the bonus window (from ceil(max_yield_day/2) to max_yield_day).

## Workflow
- Every change committed atomically.
- Research docs in `docs/research/`, new rules in `docs/rules/`.
- Strategy is evaluated in the lab first, then moved to `agent/`.
- **All repo files must be written in English only.**
- Plans are presented step-by-step and implemented only after explicit user approval.
