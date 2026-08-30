# ChistaAgent Project Rules

## Runtime
- The agent must run on the Kaggle Python image (V1.63); only libraries available in that image.
- The lab is independent and may use any Python library.
- Final submission output for Kaggle: `main.py` at the root with an `agent(obs)` function.

## Game rules (summary)
- Two players, 720 turns (24 turns × 30 days), winner = most money in bank.
- Per turn: one farmer action + hand actions + up to 10 market orders.
- Plants must be watered daily (2 unwatered days = weed); animals fed wheat daily (2 days = escape).
- Watering every other day is safe ONLY for ongoing crops (tomato/strawberry); one-time crops need daily watering.
- Sell prices are dynamic: selling lowers the price — premium products (strawberry, melon, milk, wool) crash hard on oversupply.
- `first_yield_day` hard-gates harvest: no crop can be harvested before it regardless of yield (melon: day 10).
- Decay after max lifespan = 1 unit per 2 TURNS → harvest the same day yield exists (1-day delay = −25%, 2-day = total loss).
- Animals cannot be resold — animal purchase is a sunk, durable investment.
- Fertilizer only helps when applied inside the bonus window (from ceil(max_yield_day/2) to max_yield_day); worthless for day-0 melon planting.
- HIRE cost is exact fibonacci: 1+1+2+3+5... per day; hands act independently every turn.

## Workflow
- Every change committed atomically.
- Research docs in `docs/research/`, new rules in `docs/rules/`.
- Strategy is evaluated in the lab first, then moved to `agent/`.
- **All repo files must be written in English only.**
- Plans are presented step-by-step and implemented only after explicit user approval.
