# ChistaAgent Optimization Strategy — Reference Plan (source: research agent)

> Derived from another agent's analysis. Architecture decisions and execution priority.

## Key decision: RL banned

Game dynamics are deterministic and model-able → **DP/MILP beats RL**.

The only permitted "learning": supervised distillation of the optimizer's output into a lookup table / decision tree (no torch). Executable policy <10ms; deep replanning only at day boundaries.

## Architecture — four sub-problems

### (a) Mix / capital = MILP deterministic-equivalent
- Variables: `plant[crop,day]` ~150 int, `land` 3 bins, crew, `sell/stock` ~500 cont
- ~1200 constraints → trivial for HiGHS/CBC
- Constraints: cash path, 100 tiles, action-per-day capacity, shed≤100, piecewise-linear market absorption, terminal constraint
- Liquidity DP for land-timing

### (b) Sell timing = piecewise-linear LP
- SELL accepts unlimited volume in one turn → daily granularity is sufficient
- Hard constraint: shed = 100

### (c) Daily labor = VRP-TW
- The unit count should be self-determined: minimal h such that `tour ≤ 24(h+1)`
- Old numbers 6/8/11 from finding 0010 belong to an older mix — invalid

### (d) Opponent prediction
- Identification: nearest-neighbor on first 2-3 days against 20 vendored opponents
- Deterministic simulation; uncertainty only on the opponent's sell timing → 3 scenarios (early-seller / late-seller / stockpiler)
- Decision via **CVaR** (not minimax, not expectation)

## Full DP banned
Exact DP over the full tile×age-class vector is explosive.

## Execution priority: 2 → 3 → 1 → 6 → 4 → 5 → 8 → 7

| # | Priority | Task | Acceptance criteria | Seeds |
|---|---|---|---|---|
| 1 | 🔴 | Opponent-conditioned MILP mix (replaces fixed numbers of findings 0004/0005) | ≥+10% paired money, p<0.05 | 71 (discovery 18) |
| 2 | 🔴 | Predictive selling (SELL timing — time-shifting shelved product) | Premium avg sell price ≥+25% vs stockpiler-opponent scenario | 18 |
| 3 | 🔴 | Terminal-value: end-of-season inventory worth zero; plant only if sellable by ~day 29; forced selling days 29-30 | zero residue in 95% of episodes; ≥+$500 | — |
| 4 | 🟠 | Liquidity/land-timing DP (task 1.3) | third land by day ≤6 in ≥80% of episodes without negative cash | — |
| 5 | 🟠 | Systematic tuning (4.4) with paired seeds + Wilson CI + separate holdout league (5 opponents outside the tuning loop — anti-overfit vs 20 vendored) | — | — |
| 6 | 🟡 | Fertilizer jackpot hypothesis test (A/B with/without COLLECT_FERTILIZER) | ≥+$2k paired | 18 |
| 7 | 🟡 | Full daily VRP — only after the new mix | — | — |
| 8 | ⚪ | Opponent identification (6.3) | ≥90% accuracy on 20 vendored by day 3 | — |
