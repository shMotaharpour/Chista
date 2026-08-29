# V0 Implementation Plan — Lab Harness (English, step-by-step)

**Goal:** Build the lab evaluation infrastructure: a paired-seed runner, agent registry, metric logging, and comparison reporting. No agent intelligence yet — just the measuring stick.

**Architecture:** `lab/` is independent (own deps allowed). Agents live in `agent/` as single-file callables compatible with kaggle-environments (`agent(obs) -> action dict`). The runner executes games via `kaggle_environments.make("kaggriculture")`, extracts metrics from the final step + price history, and writes structured JSON results to `lab/results/`.

**Tech Stack:** Python 3.11, kaggle-environments 1.32.7 (already installed in `.venv`), stdlib only (json, csv, argparse, dataclasses). No heavy deps needed for V0.

---

## Step 1: Lab package skeleton

**Files:**
- Create: `lab/__init__.py` (empty)
- Create: `lab/metrics.py` — metric extraction + dataclasses

**Do:**
1. Create empty `lab/__init__.py`.
2. In `lab/metrics.py` define a `GameResult` dataclass:
   - fields: `agent_a`, `agent_b`, `seed`, `rewards: [float, float]`, `statuses: [str, str]`, `steps_used: int`, `price_history: list[dict]` (day → market prices), `money_path_a: list[float]`, `money_path_b: list[float]`, `residue_a: float` (unsold inventory value at end), `residue_b: float`, `meta: dict`
3. Add `to_dict()` and `save_json(path)` methods.
4. Money path: sample `env.steps[step][p].observation.farms[p].money` every 24 steps (daily). Residue: sum `private.shed` counts × current price at final step.
5. Commit: `feat(lab): metrics module`

## Step 2: Agent registry

**Files:**
- Create: `lab/registry.py`

**Do:**
1. Map of built-in agents by name: `"pass"`, `"random"`, `"starter"` (these come from the environment itself).
2. Map of local file agents: `{"<name>": "path/to/main.py"}` resolved from `agent/` directory; loader uses `env.run([path, opp])` which kaggle-environments supports natively (paths to .py files with an `agent` function).
3. Function `resolve(name) -> str | callable` that raises a clear error for unknown names.
4. Commit: `feat(lab): agent registry`

## Step 3: Paired-seed runner

**Files:**
- Create: `lab/runner.py`

**Do:**
1. Function `run_game(agent_a, agent_b, seed=None, episode_steps=720) -> GameResult`:
   - `env = make("kaggriculture", configuration={"episodeSteps": episode_steps, "seed": seed})`
   - `env.run([resolve(agent_a), resolve(agent_b)])`
   - Extract metrics into `GameResult`.
2. Function `run_paired(agent_a, agent_b, seeds: list[int]) -> list[GameResult]` — same seed list for both sides; each seed run twice with sides swapped (a,b then b,a) to cancel first-mover advantage. Return all results.
3. CLI: `python -m lab.runner --a starter --b random --seeds 1..18 --out lab/results/` — prints a summary table (mean reward A, mean reward B, mean delta, count of wins).
4. Verify: run `python -m lab.runner --a starter --b random --seeds 1..5` → completes without error, writes 10 result JSONs (5 seeds × 2 side-swaps).
5. Commit: `feat(lab): paired-seed runner with CLI`

## Step 4: Reporting / comparison

**Files:**
- Create: `lab/report.py`

**Do:**
1. Function `summarize(results: list[GameResult]) -> dict`: mean/median delta (A−B), win rate, standard error, Wilson 95% CI on win rate.
2. Function `print_table(summary)` — aligned text table for terminal.
3. Load result JSONs from a directory, aggregate, print.
4. Verify: `python -m lab.report lab/results/<run-dir>` prints a table with non-zero stats.
5. Commit: `feat(lab): report with Wilson CI`

## Step 5: Baseline ladder measurement

**Files:**
- Create: `docs/research/002-baseline-ladder.md` (generated from report output)

**Do:**
1. Run: pass vs starter, random vs starter, starter vs starter (18 paired seeds, sides swapped).
2. Record all numbers in the research doc — this is the numeric baseline that replaces the lost findings 0004/0005.
3. Commit: `docs: baseline ladder numbers`

## Step 6: Money-path logging for analysis

**Files:**
- Modify: `lab/metrics.py` (if daily money sampling needs a tweak based on Step 5 output)
- Create: `lab/plots.py` (optional, matplotlib only if space allows — otherwise text sparkline)

**Do:**
1. Verify money_path arrays are non-trivial (not all constant) for random agent.
2. If matplotlib fits in disk budget: plot money path per agent for one seed, save PNG under `lab/results/<run>/plots/`. Else: ASCII sparkline in report.
3. Commit: `feat(lab): money-path visualization (png or sparkline)`

---

## Verification checklist (definition of done for V0)

- [ ] `python -m lab.runner --a starter --b random --seeds 1..5` completes, writes JSONs
- [ ] Reproducibility: same seed twice → identical rewards
- [ ] `python -m lab.report <run-dir>` prints mean delta, win rate, Wilson CI
- [ ] Baseline ladder numbers recorded in `docs/research/002-baseline-ladder.md`
- [ ] Every step committed atomically
- [ ] No new heavy dependencies; venv still works

## Risks / notes

- Disk: results JSONs are small (price history per game ~720×9 ints ≈ manageable). If matplotlib install threatens disk, fall back to ASCII sparkline (disk is at 87% on root; keep pip cache off or redirect to /chista).
- `env.steps` observation access: farms/market are dicts on observation objects; verify attribute vs dict access at runtime (already validated in smoke test: `s.reward` works).
- First-mover advantage: side-swap pairing in Step 3 is designed to neutralize it — verify delta sign flips when sides swap on a symmetric matchup (starter vs starter should be ≈ 0 delta).
