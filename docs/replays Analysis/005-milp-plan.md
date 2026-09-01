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


## Amendment 1 — Animal care mechanics (verified) & feeding modes REJECTED

### CARE bonus exact rule (from engine source, _daily_refresh_animals):
- `pending_care_bonus` banks +1 ONLY on days with BOTH fed AND cared.
- Missing CARE one day loses only that day's +1; the bank survives.
- **On a production day, the bank is CONSUMED even if the animal is hungry** (source
  sets pending_care_bonus = 0 unconditionally inside the production branch) — a hungry
  production day burns the entire bank AND yields no bonus.
- Two consecutive unfed days = animal escapes.

### Feeding modes tested (live env, 16 days, seed 70):
| mode | COW milk | GOOSE eggs | SHEEP wool | alive? |
|---|---|---|---|---|
| full_care (feed+care daily) | **15-18** | **25-29** | **17** | yes |
| keep_alive (feed only on/around production days) | 4 | 0 (escaped) | 0 (escaped) | cow only, 27% yield |

- GOOSE (interval 1): production every day → keep_alive ≡ full_care, but missing ANY day
  kills it → full_care mandatory.
- SHEEP (interval 3): alternating feed cannot align with production days 9,12,15
  (production-day hunger burns bank + no product) → dead.
- COW (interval 2): odd-day feeding survives but yields only 27% of full_care
  (no care banking, misses half the bonus).

**Verdict: full_care dominates for all three animals. The MILP model should NOT include
a feeding-mode decision variable — always feed + care daily (1 action/day/animal each).**

### Shop demand (corrected — per-shop product list, coverage count)
Each shop consumes 1-4 specific products (single-product shops consume ×2):
- WHEAT covered by 6 of 8 shops (highest demand coverage, near-guaranteed sell floor)
- STRAWBERRY by 4; MILK by 3; TOMATO/EGG by 2; CARROT/WOOL by 1 each (×2 consumption)
Model demand: for unlocked shops s, demand(item) = Σ_units(s) — deterministic from obs.
Products with widest shop coverage have the most predictable demand.

## Amendment 2 — Price-conditioned animal modes (supersedes "full_care dominates")

Amendment 1 concluded "full_care dominates" — **that was wrong**. It treated animal
upkeep as action-free and ignored that (a) feed is a PURCHASED input (wheat $25-50),
(b) actions have opportunity cost when the crew is expensive, and (c) fertilizer is
produced by ANY living animal, even unfed ones (source: fertilizer_available = True
in _daily_refresh_animals for every surviving animal, fed or not).

### The real per-animal economics (example: GOOSE at low prices)

Given: EGG = $40, WHEAT (feed) = $50, FERTILIZER = $70.

| mode | actions/day | inputs | revenue/day | net | net per action |
|---|---|---|---|---|---|
| full_care | 2 (FEED+CARE) | wheat $50 | egg $40 + fert $70 = $110 | $60 | $30 |
| fert_only (keep alive, skip feed) | 1.5 (COLLECT + feed every 2nd day) | wheat $25 | fert $70 | $45 | **$60** |
| no_feed_collect (no feed at all, collect until escape) | 1 (COLLECT) | none | fert $70 (2 days, then animal gone) | varies | highest short-term |

When EGG ($40) < WHEAT ($50), feeding is a NET LOSS on the egg alone — the only
reason to feed is the CARE bank, which is worthless if eggs sell below feed cost.

### Mode hierarchy (price-conditioned, per animal)

Three operating modes, chosen per animal by CURRENT market prices:

1. **PROFIT mode** (feed + care daily): worth it when
   `product_price + fert_price − wheat_price > 0` AND crew actions are not scarce
   (i.e. marginal hire cost is cheap). Full yield incl. care bank.
2. **FERT_ONLY mode** (keep alive: 1 feed every 2nd day + daily COLLECT_FERTILIZER):
   when product_price < wheat_price but fert_price > wheat_price/2.
   Yields: fertilizer only (~$70/day/animal at 1 action every other day).
   Constraint: feed days must never allow 2 consecutive misses (survival), and
   NEVER let a production day go hungry (bank burn — Amendment 1).
   GOOSE: impossible (interval 1 → production daily → must feed daily anyway;
   but CARE can be skipped in low-egg markets → "half_care": feed daily, skip care).
   COW: feed on production days 9,11,13,... (interval 2 aligns with production).
   SHEEP: production days 9,12,15,... (interval 3) do NOT align with 2-day feeding
   cadence → sheep cannot run FERT_ONLY safely → always full_care or sell/skip.
3. **RETIRE mode** (stop feeding entirely): when even fertilizer doesn't justify
   the actions. Animal escapes in 2 days; tile freed for crops. Do this when the
   tile is worth more than the fert income.

### MILP formulation change

Per animal a and day d:
- `mode[a,d]` ∈ {PROFIT, FERT_ONLY, RETIRE} (integer 1-3, monotone: can only step down)
- actions_used[a,d] = 2 (PROFIT) | 1.5 (FERT_ONLY) | 0 (RETIRE)
- yield[a,d] = production schedule × (1 + bank if PROFIT) — bank only accumulates in PROFIT
- fert[a,d] = 1 per day alive (all modes until escape)
- survival constraint: no two consecutive days without feed unless RETIRE

Decision drivers (from obs): product price vs wheat price vs fert price, marginal
crew cost (current FIB hire price), remaining season days (RETIRE near end).

### Worked example (Hossein's scenario)

EGG $40, WHEAT $50, FERT $70 → GOOSE runs FERT_ONLY: 1.5 actions/day → $45 net/day
≈ $60/action vs full_care $60/day ÷ 2 actions = $30/action. FERT_ONLY wins ×2.
If EGG rises above WHEAT + care value → switch to PROFIT mode.

### Agent implementation (rule layer, before MILP)

In `decide_unit`, animal branch priority becomes price-aware:
```
if fert_available: COLLECT_FERTILIZER            # always, 1 action, pure profit
if product_price > wheat_price: FEED + CARE       # PROFIT mode
elif product_price > 0 and wheat cheap: FEED only # half_care (no CARE action)
else: skip FEED (accept escape countdown if RETIRE decided)
```
MELON/wheat planting decisions unchanged.

## Amendment 3 — Production is UNCONDITIONAL; feeding-mode economics CORRECTED

### Engine fact (source line 828, verified live):
`yield_units += base(1) + bonus` happens on every production day REGARDLESS of fed state.
`fed_today` gates ONLY the care bonus, not base production. Survival = no 2 consecutive
unfed days (consecutive_unfed resets on any fed day).

### Verified per-animal feeding economics (live env):

| mode | GOOSE (interval 1) | COW (interval 2) | SHEEP (interval 3) |
|---|---|---|---|
| full_care (feed+care daily) | 8 eggs/8d + bank | 15-18 milk + bank | 17 wool + bank |
| alternating (feed every 2nd day) | **8 eggs/8d** (survives!) | survives if aligned to prod days | MISALIGNED → escapes |
| base yield penalty of alternating | ZERO (base unconditional) | zero base penalty, only bank lost | — |

### Corrected model:
- Survival constraint: feed at least every 2nd day (never 2 consecutive misses).
- Base production = 1/production-day REGARDLESS of feeding.
- CARE adds banked bonus (+1 per fed+care day) — an OPTIONAL yield multiplier.
- GOOSE alternating: feed days 0,2,4... production daily (fed or not) → FULL base eggs.
  Cost: half feed actions. Loss: care bank only.
- COW alternating: align feed days to production days (interval 2 aligns with 2-day
  cadence) → survives, full base, no bank.
- SHEEP: interval 3 does NOT align with 2-day cadence → production days would go hungry
  (bank burn + production day base is still paid though!). Sheep CAN run alternating but
  some production days fall on unfed days — base still produced (engine fact), so sheep
  survives AND produces base — only the bank is lost on misaligned days.

### Corrected mode economics (GOOSE example: egg $40, wheat $50, fert $70):
- full_care: 2 actions/day (feed+care), wheat $1/day → egg+daily, bank bonus.
- alternating: ~1 feed action/day (pickup+feed every 2nd day), wheat $0.5/day,
  same base eggs, no bank.
- Per-action efficiency: alternating is ~2x better when actions are scarce.
- CARE becomes worthwhile only when product price > feed cost (care is pure
  extra actions for extra banked yield).

### MILP model change (final):
- `mode[a]` ∈ {FULL_CARE, KEEP_ALIVE} per animal (decision var).
- KEEP_ALIVE: feed actions = 0.5/day, care actions = 0, base production only.
- FULL_CARE: feed = 1/day, care = 1/day, base + banked bonus.
- Survival: both modes satisfy (KEEP_ALIVE feeds every 2nd day = no 2 consecutive misses).
- Choose by prices: if (product_price − wheat_price) × interval_yield_gain > care_action_cost
  → FULL_CARE else KEEP_ALIVE.
