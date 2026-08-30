# Project status — where we are (Audit 2026-08-29)

## What we have ✅

- **Game env works:** `kaggriculture` is registered inside `kaggle-environments 1.32.7` and runs (no local source needed)
- **Paired seeds are reproducible:** two runs with `seed: 42` gave identical results → statistical evaluation base (paired seeds) ready
- **Parameters:** `episodeSteps`, `boardSize`, `startingMoney`, `seed`, `marketParams`... all configurable via configuration
- **Docs:** game README, AGENT.md, env source, strategy plan (`docs/research/000-strategy-plan.md`)
- **Infrastructure:** git repo + venv + dedicated 100G disk (`/chista`)

## What we don't have ❌

- **No agent code** — `agent/` is empty; built-in baselines: `pass`, `random`, `starter`
- **No lab code** — no harness, evaluator, or evaluation tooling (resolved later same day: harness v0 built)
- **Old numeric baselines (findings 0004/0005/0010, numbers 6/8/11)** — lost with the previous project; must be regenerated from zero
- **Vendored opponents** — not yet identified/available (needed for opponent prediction and holdout league) (resolved: 19 opponents installed from user archive)
- **Leaderboard and competition logs** — must be pulled from Kaggle (CLI needed)

## Ambiguities from the strategy plan (to resolve)

- DP/DL/RL ban — the plan requires a liquidity DP while banning full DP; the exact boundary will be determined by testing
- Reference numbers (best/worst mix, expected prices) must be built with our own simulation

## Track 1 — the problem itself (what to beat)

1. **Baseline ladder:** `pass` → `random` → `starter` → our agent
   - `starter` (single-tile carrot loop) ~$3400 over 720 turns = baseline (superseded: 19 real opponents, $60k–$158k level)
2. **Game economics:** profit per product/animal (yield/tile/day × expected price − cost) via simulation
3. **Price dynamics:** how high-volume SELL breaks prices; spread and timing strategy
4. **Money growth curve:** which mix gives which liquidity path (input to MILP)

## Track 2 — development (how to build)

- v0: **lab code** — match runner, agent registry, metric logging (money, residue, prices), JSON storage, comparison report ✅ done
- v1: **rule-based agent** (improved carrot loop + simple terminal-value) → must beat starter
- v2: **market simulation** in lab for sell decisions (LP/piecewise regression)
- v3: **MILP mix** with HiGHS (per strategy plan)
- v4: **opponent prediction + CVaR**
- Each step: acceptance criteria + paired seeds + results report (per 000-strategy-plan)

## Immediate order this week

1. ✅ Lab scaffold + evaluation harness (paired-seed runner)
2. ✅ Metric logging: money-path, residue, price-history
3. ⬜ Rule-based agent v1 that beats starter
4. ✅ Economics report: profit per tile per day per product/animal at real game prices
