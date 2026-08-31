# Phase (a) — MILP Season Planner: Detailed Implementation Plan

> Status: plan awaiting approval. Feeds the rule-based agent a proven-optimal
> season plan instead of hand-written rules. VRP-TW (space c) is assumed to exist
> as an independent service that, given the day's task list and farm geometry,
> returns per-worker tours and the minimum crew size.

## 1. Goal

Solve the full 30-day season as one deterministic optimization problem:
*which crops to plant on which day, when to buy land, how many hands to hire,
what to sell and when* — maximizing final bank, respecting every verified
mechanic (docs/replays Analysis/000-003, docs/research/006).

Output is a **season plan** (`plan.json`) the agent executes day-by-day.
Re-planning happens only at day boundaries (v1: fixed plan; v2: rolling replan).

## 2. Why MILP

Dynamics are deterministic and piecewise-linear → MILP is provably near-optimal.
RL/DL is banned (000-strategy-plan). HiGHS solves ~1,200 constraints in seconds.

## 3. Index sets and constants (all verified)

| Set | Values | Source |
|---|---|---|
| Days | d = 0..29 (24 turns/day) | config |
| Crops | WHEAT, CARROT, TOMATO, STRAWBERRY, MELON | env source |
| Products (sellable) | WHEAT, CARROT, TOMATO, STRAWBERRY, MELON, EGG, MILK, WOOL, FERTILIZER | env source |
| Animals | GOOSE(coop), COW(pasture), SHEEP(pasture) | env source |
| Quadrants | NE ($1k), SW ($2k), SE ($4k) | env config |

Crop constants (per crop): seed cost, first_yield_day, max_yield_day, interval,
ongoing flag, base price, unfertilized cap. From `CROPS` in env source; verified
in docs/research/003 and 006.

Animal constants: cost, structure, first_yield_day, production interval,
max_held, product, care-banking rule (verified 006 §V6).

Market: price = piecewise function of inventory around I0=10,000 with per-item
shape functions and T anchors (env `MARKET_PARAMS`); floor $1; sells at floor
don't add inventory. Verified exact in `lab/prices.py`.

## 4. Decision variables

| Variable | Domain | Meaning |
|---|---|---|
| `plant[c,d]` | integer ≥ 0 | seeds of crop c planted on day d |
| `land[q]` | binary | quadrant q purchased |
| `hire[d]` | integer ≥ 0 | hands hired on day d (reset daily) |
| `crew[d]` | integer ≥ 0 | total workers available on day d (= 1 + hired + carried) |
| `sell[p,d]` | continuous ≥ 0 | units of p sold on day d |
| `stock[p,d]` | continuous ≥ 0 | units of p in shed end of day d |
| `buy_wheat[d]` | continuous ≥ 0 | wheat bought for animal feed on day d |
| `harvest[p,d]` | continuous ≥ 0 | units of p harvested on day d (determined by plant schedule) |
| `animals[a,d]` | integer ≥ 0 | animals of type a alive on day d |
| `land_tiling[d]` | continuous ≥ 0 | helper for tile-count constraint |

Sizes: plant ≈ 150 int; land 3 bin; sell/stock/harvest ≈ 500 cont; crew ≈ 30 int.
Total ≈ 700 vars, ~1,200 constraints → HiGHS solves in seconds.

## 5. Constraints (each mechanic verified)

1. **Cash path** — money never negative:
   `money[d+1] = money[d] + Σ revenue[sell[p,d]] − seed_cost − land_cost − hire_cost − feed_cost − animal_cost`
   Hire cost = FIB(n) cumulative (1,1,2,3,...) — verified engine notes.
2. **Tile capacity** — plants on day d ≤ 25 × (1 + Σ land[q] bought by d):
   `Σ_c live_plants[c,d] ≤ 25 × (1 + Σ_q land[q]·[buy_day_q ≤ d])`
3. **Watering capacity (VRP link)** — plantings needing water ≤ crew capacity:
   `live_plants[d] ≤ crew[d] × CREW_RATE` (CREW_RATE = tiles/unit/day, swept = 6;
   VRP service will later replace this constant with exact tour capacity).
4. **Shed cap** — `stock[p,d] ≤ 100` (overflow discarded → model forbids it).
5. **Market absorption (piecewise-linear)** — selling n units moves inventory by n;
   revenue computed per tranche. Linearized: sell split into ≤ K tranches per day
   with per-tranche marginal price (K ≈ 8 covers the curve).
6. **Harvest schedule** — deterministic from plant: first units at
   `plant_day + first_yield_day`, ongoing crops produce every `interval` days up to
   max_yield; one-time crops yield once at max_yield_day (with watering bonus
   included as verified constants: unfert cap / fert cap).
7. **Animals** — bought on day d produce first at `d + first_yield`, then every
   `interval`; feed = 1 wheat/day (from `buy_wheat` or shed); CARE banking adds
   +1/day fed-and-cared to next production (verified 006 §V6); max_held caps
   unharvested product; escape if unfed 2 days (model forbids).
8. **Terminal value** — inventory at end of season worth ZERO:
   `maximize money[30]`; constraint: `sell[p,29] + sell[p,30] ≥ stock[p,28]`
   (everything sold, even below base).
9. **Land timing** — NE/SW/SE buy days are chosen by the optimizer (replay
   evidence: winners buy NE@6, SW@11 — the model may find better under its own
   cash-flow; acceptance will compare against this baseline).

## 6. Objective

```
maximize  money[30]
```
(final bank; unsold inventory worthless — the terminal constraint forces sale).

## 7. Market revenue linearization

Price function per product: `P(inv) = base + amp·f(|inv − I0|)` with shapes
linear / sq / sqrt / log / hinge. For the LP we approximate each side of I0 with
K=6-8 line segments over the reachable inventory band (inventory only moves
~±2,000 in practice). Segment marginal prices precomputed from `lab/prices.py`.

## 8. Implementation steps (bite-sized)

| # | Task | Output | Test |
|---|---|---|---|
| 1 | Install HiGHS (`highspy`) in venv | import works | trivial |
| 2 | `lab/milp/constants.py` — port CROPS/ANIMALS/MARKET_PARAMS + verified mechanics table | constants module | assert vs lab/prices.py |
| 3 | `lab/milp/model.py` — build MILP for a given config (board, money, start day) | model builder | model builds, HiGHS solves toy instance |
| 4 | Solve baseline instance (25 tiles, $3k, day 0) | plan.json | plan is feasible; cash path ≥ 0 all days |
| 5 | `lab/milp/execute.py` — convert plan.json → per-day agent ops | executor | executor follows plan in real env for 3 days |
| 6 | Full season execution vs 'pass' on 3 seeds | money | compare vs MILP-predicted money → gap report |
| 7 | Gap analysis doc (docs/replays Analysis/005-milp.md) | report | documented reasons for any gap |
| 8 | Iterate: fix biggest gap (usually watering capacity → VRP dependency) | improved plan | repeat 6 |

## 9. Acceptance criteria

- MILP solves the season in < 60s
- Executed season money ≥ 2× current v1.2 ($28k → target $55k+)
- Predicted vs executed gap < 15% (measures model fidelity)
- Plan respects ALL verified mechanics (no weeds, no animal escapes, zero residue)

## 10. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Piecewise price linearization error | K=8 segments; compare LP revenue vs exact for sample paths |
| Watering capacity approximation (CREW_RATE constant) | VRP-TW (space c) later replaces constant with exact tours; for now swept value |
| Model too rigid (fixed plan breaks on weeds) | executor has safety layer: weeds/water still handled by rule layer (v1.2 dispatcher) — plan only drives PLANT/SELL/BUY decisions |
| HiGHS int variables slow | relax plant to continuous (yield is linear anyway), keep land binary |

## 11. What stays rule-based (v1.2 dispatcher reused)

- Watering, harvesting, feeding, CARE, DIG — short-horizon tile ops (priority ladder)
- PICKUP batching, movement (greedy step; VRP later)
- FERTILIZE inside bonus window
The plan dictates: what/when to PLANT, BUY (seeds/land/animals/wheat), SELL.

## 12. Deliverables

- `lab/milp/` (constants, model, execute, tests)
- `plan.json` schema: per-day {plant: {crop: n}, sell: {item: n}, buy: {...}, land: bool, hire: n}
- `docs/replays Analysis/005-milp.md` — predicted vs executed gap report
