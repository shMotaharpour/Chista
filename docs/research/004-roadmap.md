# Roadmap — ChistaAgent

> Process: each step is first presented as a plan (steps + sub-steps); implementation only after explicit user approval.

## V0 — Lab Harness ✅ (done)

- [x] metrics / registry / paired-seed runner / Wilson CI report
- [x] 19 vendored opponents installed + ladder ranking (`docs/research/005-opponent-ladder.md`)
- [ ] ⬜ Baseline runs vs starter (superseded: ladder vs 19 real opponents is the reference now)

---

## V1 — Rule-based agent that beats starter

**Goal:** delta ≥ +$200 vs starter over 18 paired games (p<0.05)

### 1.1 Base economics (research + doc) ✅
- 1.1.1 Theoretical profit per crop: (yield/tile/day from table) × base price − seed cost ✅
- 1.1.2 Animal profit incl. wheat feed cost (1 wheat/day) ✅
- 1.1.3 Price table reproduction: port `market_price()` into lab + price-inventory curves ✅
- 1.1.4 Results → `docs/research/003-crop-economics.md` ✅
- 1.1.5 Mechanics verification (chain effects) → `docs/research/006-mechanics-verification.md` ✅
  - fertilizer works only inside the bonus window; melon harvestable day 8 with fert (first_yield_day does not gate harvest)
  - daily watering mandatory; alternating OK for ongoing crops; decay = 1 unit per 2 TURNS (harvest same day!)
  - animals cannot be sold; shed overflow: buys rejected, DROP/PICKUP overflow discarded
  - simultaneous selling = lockstep; fast arbitrage nets zero

### 1.2 Agent v1
- 1.2.1 Skeleton `agent/main.py`: modular decision structure (market / plant / care / harvest)
- 1.2.2 Planting logic: best available crop given liquidity and season day
- 1.2.3 Care loop: watering (priority per 006 findings), DIG weeds, harvest the moment yield > 0
- 1.2.4 Selling: full shed sale at end of day (no timing optimization yet) with safe-pace caps
- 1.2.5 Simple terminal-value: no new late plantings after ~day 26; forced selling days 29-30
- 1.2.6 Evaluate with harness: 18 paired seeds vs starter → report

### 1.3 Rule-based optimization loop
- 1.3.1 Analyze lost games (money-path and residue from harness JSONs)
- 1.3.2 Fix rules and re-evaluate (iterate until target)

---

## V2 — Market simulator and predictive selling

**Goal:** premium sell price ≥ +25% vs immediate dumping (stockpiler-opponent scenario)

- 2.1 Price curve in lab (copy from source) + equivalence test vs live env ✅ (done in 1.1.3)
- 2.2 Predict impact of a large sell order on price (piecewise simulation)
- 2.3 Sell timing: drip selling at daily/weekly price peaks
- 2.4 A/B: v1 + smart selling vs v1 — 18 paired seeds

---

## V3 — MILP mix (strategy plan, 🔴 priority)

**Goal:** ≥ +10% paired money, p<0.05, 71 seeds

- 3.1 Install HiGHS in venv (light library, no heavy deps)
- 3.2 MILP modeling: plant[crop,day], land bins, continuous sell/stock, cash/tile/shed constraints
- 3.3 Solve offline for liquidity profiles → policy table (lookup)
- 3.4 Distill into a rule/lookup executable in <10ms inside the agent
- 3.5 Evaluate 71 seeds vs the best previous version

## V4 — Opponent prediction + CVaR

- 4.1 Collect vendored opponents from competition data / replays ✅ (19 in lab/opponents)
- 4.2 Nearest-neighbor identification on first 2-3 days
- 4.3 Three opponent sell scenarios (early/late/stockpiler) + CVaR choice
- 4.4 Evaluate against all three scenarios

## V5 — Remaining strategy-plan items

- Liquidity/land-timing DP (🟠)
- Full daily VRP (🟡)
- Systematic tuning with anti-overfit holdout league

---

## Fixed rules for every step

1. Plan before code — user approval required
2. Every sub-step = atomic commit
3. Every strategy change = paired-seed evaluation + report before merge
4. Findings documented in docs/research/ with sequential numbers
5. **All repo files in English only**
