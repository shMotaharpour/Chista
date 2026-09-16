r"""The arena: paired-seed evaluation of two agent versions (issue #18).

One command answering "is this version better than that one, against whom,
by how much, and is the difference real?":

    .venv/bin/python -m offline.evaluate --a <slug-or-ref> --b <slug-or-ref> \
        --tier smoke|ladder|full

Design (issue #18, corrections from #20's brief and addendum):

- TWO METRICS, TWO JOBS. The paired coin margin (A - B, per opponent per
  seed) is the development signal; win rate against the class-R pool is
  the ship gate. They are reported separately and never conflated. Ties
  are counted explicitly (docs/player_agent.md: "ties are possible") -
  never silently as losses, never as halves.
- PAIRED COMPARISON. Both versions meet the SAME opponent on the SAME
  seed, and the per-seed differences are compared - not means of
  independent runs (the AgriOracle lesson in issue #18: a single-seed
  +4.6 % became +7.1 % under 16 paired seeds). F045 (weed RNG depends on
  both farms' planting) means same-seed pairing is NOT common random
  numbers past the first behavioural divergence; the shared prefix is
  still real variance reduction. The per-opponent spread this harness
  measures IS the F045-contaminated variance - the seed-count line at
  the bottom of the report turns it into the measured n that replaces
  the provisional 16 (brief 6.3).
- SEAT EFFECTS CANCEL IN THE PAIRED DIFFERENCE. Both versions meet
  every opponent in the SAME seat (seat 0, the seat the harness calls
  first), so the seat term is common to both sides and subtracts out.
  The paired margin is clean; the WIN RATE, however, is seat-0-only —
  a known limitation of the ship gate as built, not a property of the
  board (brief #18 §3 asked for both orders; the full-tier gate should
  play both). Measured seat effect on this box, same pair both orders
  (2026-09-16): seed 0 delta 0, seed 1 delta 4,171 coins (seat-0
  advantage ≈ 2,086 mean) — noise at today's −57.9k margins, material
  the moment margins approach a few thousand coins.
- REF NAMES. `--a`/`--b` take either a vendored pool slug or an
  agent-side reference: "main" (repo agent/main.py) or
  "agent.main:agent". Pool slugs make any two competitors comparable
  with the same instrument. Baselines are versioned by git SHA (R005).
- TIERED, BECAUSE COMPUTE IS THE CONSTRAINT (~12-15 s per episode, the
  planning-dominated estimate from issue #18 - corrected 20 -> 19
  opponents per #20 C1, so the full tier is 608 agent-episodes, not 640):

      tier   opponents            seeds  agent-episodes  when
      smoke  5 representative     4      40              every commit
      ladder 15 dev pool          16     480             per milestone
      full   19 incl. held-out    16     608             M5 only

  The smoke five are the first five of the dev split (sorted, stable) -
  a placeholder until #20's measured 14/5 split lands; --opponents
  overrides. `offline/pool/registry.py` (the measured split, #20) is
  used when present; until then every pool agent is treated as dev and
  the report says so.
- PARALLELISM IS THROUGHPUT ONLY. All coin/win numbers come from
  `run_episode_process` children (one process per episode, brief 5);
  per-turn wall time collected in that mode is reported as NOT a budget
  number (A4: a parallel sweep's timing reads the load, not the agent).
  The F046 budget check is the --timing path: serial, one process,
  `timing_solo` (vs PASS) and `timing_contended` (vs a real opponent),
  with the addendum's direction assertion (contended >= solo - a
  contended reading faster than solo is a broken harness, not fast load).
- LOSS AUTOPSY. The widest paired losses are dumped as JSON with both
  action streams and both money series - the raw material for "the day
  the gap opened". `--autopsy N` (default 2, 0 disables). Every number
  in every file names where it came from (R005).

Scoreboard: each run appends one row to `offline/scoreboard.csv` and
writes the full report + records under `artifacts/evaluate/<id>/`, so
the trajectory across milestones is visible rather than remembered.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# --- tier table (issue #18; #20's amendment A landed: 19 slugs, 3
# byte-identical duplicates, 16 canonical; registry.py holds the
# measured 11/5 dev/held-out split, which _dev_split consumes).
TIER_OPPONENTS = {"smoke": 5, "ladder": 11, "full": 19}
TIER_SEEDS = {"smoke": 4, "ladder": 16, "full": 16}
TIER_AGENT_EPISODES = {"smoke": 40, "ladder": 352, "full": 608}

HELD_OUT_NOTE = ("dev/held-out split not measured yet (#20 in progress); "
                 "every pool agent treated as dev")

# The agent-side reference names understood by --a/--b.
AGENT_REFS = {"main": "agent.main:agent"}

# progress post (live-progress-post skill); never fatal.
_PROGRESS = Path.home() / ".hermes" / "scripts" / "live_progress_post.py"
_PROGRESS_LABEL = "Issue 18: evaluation harness"


def _progress(detail: str) -> None:
    if not _PROGRESS.is_file():
        return
    try:
        subprocess.run([sys.executable, str(_PROGRESS), "update",
                        _PROGRESS_LABEL, detail],
                       capture_output=True, text=True, timeout=20)
    except Exception:                            # noqa: BLE001 - never fatal
        pass


# --- reference resolution ------------------------------------------------

def _resolve_ref(ref: str) -> str:
    """'main' / 'agent.main:agent' / a pool slug -> a runner-side ref.

    Pool slugs pass through; agent-side refs become "ref:<module>:<attr>"
    which offline.runner._load_agent imports directly from the repo (no
    shim under opponents/ - the vendored tree is SHA-pinned by
    tests/test_opponents.py and must not gain entries).
    """
    if (REPO / "opponents" / ref / "agent.py").is_file():
        return ref
    ref = AGENT_REFS.get(ref, ref)
    if ref.startswith("ref:"):
        return ref
    if ":" in ref:
        module, attr = ref.split(":", 1)
    else:
        module, attr = ref, "agent"
    src = REPO / (module.replace(".", "/") + ".py")
    if not src.is_file():
        raise SystemExit(f"unknown --a/--b ref {ref!r}: no such pool slug "
                         f"and no such module file ({src})")
    return f"ref:{module}:{attr}"


# --- opponent selection --------------------------------------------------

def _dev_split() -> tuple[list[str] | None, str]:
    """The measured dev / held-out split from #20's registry (11/5 over
    the canonical 16 - amendment A dropped the three duplicates)."""
    try:
        from offline.pool.registry import dev_and_heldout
    except Exception:                                # noqa: BLE001 - not yet
        return None, HELD_OUT_NOTE
    dev, _held = dev_and_heldout()
    if not dev:
        return None, HELD_OUT_NOTE
    return sorted(dev), ("measured 11-dev/5-held-out split from "
                         "offline/pool/registry.py (#20, canonical 16)")


def _pick_opponents(tier: str, override: str | None) -> tuple[list[str], str]:
    """The tier's opponent list, and the sentence describing where it came
    from (R005: every selection is named)."""
    if override:
        slugs = [s.strip() for s in override.split(",") if s.strip()]
        return slugs, "--opponents override"
    dev, note = _dev_split()
    from offline.pool.loader import slugs
    if tier == "full":
        return sorted(slugs()), ("all 19 vendored opponents, M5 gate - "
                                 "#20 amendment A: 16 canonical, 3 "
                                 "byte-identical duplicates included for "
                                 "provenance; canonical gate = ladder")
    if dev is None:
        return sorted(slugs())[:TIER_OPPONENTS[tier]], \
            f"first {TIER_OPPONENTS[tier]} sorted slugs - {HELD_OUT_NOTE}"
    if tier == "ladder":
        return dev, note
    return dev[:TIER_OPPONENTS["smoke"]], \
        f"first {TIER_OPPONENTS['smoke']} of the dev split - {note}"


# --- the paired schedule -------------------------------------------------

def _paired_jobs(opponents: list[str], seeds: list[int]) -> list[tuple[str, int]]:
    """(opponent, seed) jobs, interleaved so no seed strands.

    Seed s starts at opponent index s mod n and the pair walks forward
    diagonally: the first len(opponents) jobs touch every seed (and
    every opponent) once, so a run cut short leaves every seed with
    similar coverage instead of the first opponents holding all seeds.
    """
    n = len(opponents)
    return [(opponents[(col + row) % n], seeds[row % len(seeds)])
            for col in range(n)
            for row in range(len(seeds))]


# --- one paired comparison ----------------------------------------------

def _run_pair(a_slug: str, b_slug: str, opp: str, seed: int,
              episode_steps: int, timeout_s: float) -> dict:
    """Both versions against the same opponent on the same seed.

    Two physical episodes (one per version); the opponent's seat record
    is not part of the comparison. Returns one merged paired record.
    """
    from offline import runner as R
    rec_a = R.run_episode_process(a_slug, opp, seed, episode_steps, timeout_s)
    rec_b = R.run_episode_process(b_slug, opp, seed, episode_steps, timeout_s)
    sa, sb = rec_a["seats"][0], rec_b["seats"][0]
    ok = (rec_a.get("status") == "DONE" and rec_b.get("status") == "DONE"
          and sa.get("rewards") is not None and sb.get("rewards") is not None)
    rec = {"opponent": opp, "seed": seed, "status": "ok" if ok else "abandoned",
           "a_slug": a_slug, "b_slug": b_slug,
           "a_coins": float(sa["rewards"]) if sa.get("rewards") is not None else None,
           "b_coins": float(sb["rewards"]) if sb.get("rewards") is not None else None,
           "margin": (float(sa["rewards"]) - float(sb["rewards"]))
                     if ok else None,
           "a_guard": sa.get("guard"), "b_guard": sb.get("guard"),
           "error": rec_a.get("error") or rec_b.get("error")}
    if ok:
        m = rec["margin"]
        rec["outcome"] = "win" if m > 0 else ("loss" if m < 0 else "tie")
    return rec


def _run_all(a_slug: str, b_slug: str, jobs: list[tuple[str, int]],
             episode_steps: int, timeout_s: float, parallel: int,
             records_path: Path) -> list[dict]:
    """The paired jobs, newest progress to the Telegram post, records
    appended to a JSONL file as they land (a killed run keeps its data)."""
    records: list[dict] = []
    total = len(jobs)
    t0 = time.perf_counter()
    if parallel <= 1:
        for i, (opp, seed) in enumerate(jobs):
            rec = _run_pair(a_slug, b_slug, opp, seed, episode_steps, timeout_s)
            records.append(rec)
            _append_jsonl(records_path, rec)
            _progress(f"{i + 1}/{total} paired episodes "
                      f"({(i + 1) * 100 // total}%)")
    else:
        with ThreadPoolExecutor(max_workers=parallel) as ex:
            futs = {ex.submit(_run_pair, a_slug, b_slug, opp, seed,
                              episode_steps, timeout_s): (opp, seed)
                    for opp, seed in jobs}
            for i, fut in enumerate(as_completed(futs)):
                rec = fut.result()
                records.append(rec)
                _append_jsonl(records_path, rec)
                dt = time.perf_counter() - t0
                eta = dt / (i + 1) * (total - i - 1)
                _progress(f"{i + 1}/{total} paired episodes "
                          f"({(i + 1) * 100 // total}%), "
                          f"eta {eta / 60:.0f}m")
    records.sort(key=lambda r: (r["opponent"], r["seed"]))
    return records


def _append_jsonl(path: Path, rec: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        fh.write(json.dumps(rec, sort_keys=True) + "\n")


# --- statistics ----------------------------------------------------------

def _stats(margins: list[float]) -> dict:
    """Paired-mean summary. The 95% CI is the normal approximation
    (1.96 sd / sqrt n) - at n=4 it is indicative, at n=32 solid; the sd
    itself is the measured quantity the seed-count line consumes."""
    n = len(margins)
    if n == 0:
        return {"n": 0}
    mean = statistics.fmean(margins)
    sd = statistics.stdev(margins) if n >= 2 else float("nan")
    half = 1.96 * sd / math.sqrt(n) if n >= 2 else float("nan")
    return {"n": n, "mean": mean, "sd": sd,
            "ci_lo": mean - half, "ci_hi": mean + half}


def _seeds_needed(sd: float, mean: float) -> float:
    """Measured seed count replacing the provisional 16 (#20 brief 6.3):
    paired seeds so that the 95% CI half-width (1.96 sd/sqrt n) no longer
    exceeds the observed mean margin. Normal approximation, R005-named."""
    if not (sd == sd) or sd <= 0 or abs(mean) < 1e-9:
        return float("nan")
    return (1.96 * sd / abs(mean)) ** 2


# --- timing mode (F046 budget check) ------------------------------------

def _timed_episode_worker_code(slug: str, opp: str, seed: int,
                               episode_steps: int) -> str:
    return (
        "import sys, json\n"
        "sys.path.insert(0, '.')\n"
        "from offline.runner import _timed_episode_worker\n"
        f"rec = _timed_episode_worker({slug!r}, {opp!r}, {seed}, "
        f"{episode_steps})\n"
        "print(json.dumps(rec))\n"
    )


def _timed_episode(slug: str, opp: str, seed: int,
                   episode_steps: int = 720) -> dict:
    """Serial timing run: one process, per-turn wall time for `slug`."""
    from offline import runner as R
    proc = subprocess.run(
        [sys.executable, "-c",
         _timed_episode_worker_code(slug, opp, seed, episode_steps)],
        cwd=REPO, capture_output=True, text=True, timeout=1200.0)
    if proc.returncode != 0:
        raise RuntimeError(f"timed episode failed: "
                           f"{proc.stderr.strip().splitlines()[-1:] or '?'}")
    rec = json.loads(proc.stdout.strip().splitlines()[-1])
    rec["mode"] = "timing"
    rec["reading"] = "timing_solo" if opp == "PASS-proxy" else "timing_contended"
    return rec


def _timing_block(a_slug: str, b_slug: str, records: list[dict],
                  episode_steps: int) -> list[str]:
    """The two A4 readings, plus the direction assertion.

    - timing_solo: vs PASS - the floor.
    - timing_contended: vs the slowest opponent observed in THIS run
      (highest guard self-p95); the honest number the bank policy
      derives from (A4: a competing opponent took 59% of compute).
    """
    from offline.runner import run_episode_process  # noqa: F401 - pattern only
    lines = ["", "timing (serial, per-turn wall time; F046 floor)"]

    solo = {}
    for slug in (a_slug, b_slug):
        rec = _timed_episode(slug, "PASS-proxy", seed=0,
                             episode_steps=episode_steps)
        solo[slug] = rec
    contended_opp = _slowest_opponent(records)
    contended = {}
    for slug in (a_slug, b_slug):
        rec = _timed_episode(slug, contended_opp, seed=0,
                             episode_steps=episode_steps)
        contended[slug] = rec

    lines.append(f"{'slug':<44} {'reading':<18} {'p50':>8} {'p95':>8} "
                 f"{'max':>8} {'bank':>6} {'turns':>6}")
    for slug in (a_slug, b_slug):
        for rec in (solo[slug], contended[slug]):
            t = rec["timing_ms"]
            lines.append(f"{slug:<44} {rec['reading']:<18} "
                         f"{t['p50']:>8.1f} {t['p95']:>8.1f} "
                         f"{t['max']:>8.1f} {rec['bank_drawn_s']:>6.3f} "
                         f"{rec['turns']:>6d}")
    for slug in (a_slug, b_slug):
        s, c = solo[slug]["timing_ms"]["p95"], \
            contended[slug]["timing_ms"]["p95"]
        if c < s:
            raise AssertionError(
                f"A4 direction assertion failed for {slug}: contended p95 "
                f"{c:.1f} ms < solo p95 {s:.1f} ms - the harness is broken "
                f"(a contended reading faster than solo is an inverted "
                f"measurement, addendum A4), not fast load")
    lines.append("direction assertion: contended >= solo - holds for both "
                 "versions")
    return lines


def _slowest_opponent(records: list[dict]) -> str:
    """The opponent with the highest observed guard self-p95 this run."""
    best, best_p95 = None, -1.0
    for rec in records:
        for seat in ("a_guard", "b_guard"):
            g = rec.get(seat) or {}
            p95 = g.get("self_p95_ms") or 0.0
            if p95 > best_p95:
                best_p95, best = p95, rec["opponent"]
    if best is None:
        # no guard data (abandoned run): a named default, to be replaced
        # by the measured slowest once #20's registry lands
        from offline.pool.loader import slugs
        return sorted(slugs())[0]
    return best


# --- loss autopsy --------------------------------------------------------

_AUTOPSY_WORKER = (
    "import sys, json\n"
    "sys.path.insert(0, '.')\n"
    "from offline.runner import _episode_worker\n"
    "rec = _episode_worker({slug!r}, {opp!r}, {seed}, {steps})\n"
    "print(json.dumps(rec))\n"
)


def _autopsy_worker_code(slug: str, opp: str, seed: int, steps: int) -> str:
    return _AUTOPSY_WORKER.format(slug=slug, opp=opp, seed=seed,
                                  steps=steps)


def _run_autopsy(a_slug: str, opp: str, seed: int, episode_steps: int,
                 out_dir: Path, b_slug: str) -> Path:
    """Re-run one paired episode with full per-turn capture and dump JSON.

    The autopsy question (#18): our plan vs realised, their sell calendar
    vs ours, the day the gap opened. This file is the raw material - the
    readable report over it is the analysis layer's job.
    """
    from offline import runner as R

    def _full(slug: str) -> dict:
        proc = subprocess.run(
            [sys.executable, "-c",
             _autopsy_worker_code(slug, opp, seed, episode_steps)],
            cwd=REPO, capture_output=True, text=True, timeout=1200.0)
        if proc.returncode != 0:
            return {"error": proc.stderr.strip().splitlines()[-1:]}
        return json.loads(proc.stdout.strip().splitlines()[-1])

    rec_a = _full(a_slug)
    rec_b = _full(b_slug)
    rec_opp = R.run_episode_process(opp, a_slug, seed, episode_steps, 1200.0)
    ra = rec_a.get("seats", {}).get("0", rec_a.get("seats", {}).get(0, {}))
    rb = rec_b.get("seats", {}).get("0", rec_b.get("seats", {}).get(0, {}))
    opp_seats = rec_opp.get("seats", {})
    opp0 = opp_seats.get("0", opp_seats.get(0, {}))
    opp1 = opp_seats.get("1", opp_seats.get(1, {}))
    doc = {
        "source": "offline/evaluate.py loss autopsy (issue #18)",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "pairing": {"a": a_slug, "b": b_slug, "opponent": opp, "seed": seed},
        "a_actions": ra.get("actions"),
        "a_money": ra.get("money_series"),
        "b_actions": rb.get("actions"),
        "b_money": rb.get("money_series"),
        "opponent_actions_seat0_vs_a": opp0.get("actions"),
        "opponent_money_seat0_vs_a": opp0.get("money_series"),
        "rewards": {"a": ra.get("rewards"), "b": rb.get("rewards"),
                    "opponent_vs_a": opp1.get("rewards")},
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"autopsy_{a_slug}_vs_{opp}_seed{seed}.json"
    path.write_text(json.dumps(doc, indent=1, default=list))
    return path


# --- the scoreboard ------------------------------------------------------

SCOREBOARD = REPO / "offline" / "scoreboard.csv"
_COLUMNS = ["run_id", "utc", "label", "tier", "a", "b", "git_sha",
            "opponents_note", "n_jobs", "n_ok", "n_abandoned", "wins",
            "losses", "ties", "win_rate", "margin_mean", "margin_sd",
            "ci_lo", "ci_hi", "seeds_needed", "records"]


def _git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              cwd=REPO, capture_output=True,
                              text=True).stdout.strip()
    except Exception:                            # noqa: BLE001
        return "unknown"


def _append_scoreboard(row: dict) -> None:
    new = not SCOREBOARD.is_file()
    SCOREBOARD.parent.mkdir(parents=True, exist_ok=True)
    with SCOREBOARD.open("a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=_COLUMNS, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow(row)


# --- report --------------------------------------------------------------

def _fmt(x: float, spec: str = "+.1f") -> str:
    return f"{x:{spec}}" if x == x else "nan"


def _report(a_slug: str, b_slug: str, tier: str, records: list[dict],
            opponents: list[str], seeds: list[int], opp_note: str,
            extra_lines: list[str]) -> str:
    ok = [r for r in records if r["status"] == "ok"]
    abandoned = [r for r in records if r["status"] != "ok"]
    lines = [
        f"=== paired evaluation - {tier} tier ===",
        f"A: {a_slug}    B: {b_slug}",
        f"git: {_git_sha()}    utc: {datetime.now(timezone.utc).isoformat()}",
        f"jobs: {len(opponents)} opponents x {len(seeds)} seeds = "
        f"{len(records)} paired comparisons ({2 * len(records)} "
        f"agent-episodes)",
        f"opponents: {opp_note}",
        f"ok {len(ok)}   abandoned {len(abandoned)}"
        + (f"  <- {abandoned[0]['error']}" if abandoned else ""),
        "",
        "per-opponent paired coin margin (A - B, same seed, seat 0):",
        f"{'opponent':<44} {'n':>3} {'w/l/t':>7} {'mean':>10} {'sd':>9} "
        f"{'95% CI':>22}",
    ]
    for opp in opponents:
        rs = [r for r in ok if r["opponent"] == opp]
        m = [r["margin"] for r in rs]
        s = _stats(m)
        if s["n"] == 0:
            lines.append(f"{opp:<44} {0:>3} {'-':>7} {'-':>10} {'-':>9} "
                         f"{'-':>22}")
            continue
        w = sum(1 for r in rs if r["outcome"] == "win")
        l = sum(1 for r in rs if r["outcome"] == "loss")
        t = sum(1 for r in rs if r["outcome"] == "tie")
        lines.append(f"{opp:<44} {s['n']:>3} {f'{w}/{l}/{t}':>7} "
                     f"{_fmt(s['mean']):>10} {_fmt(s['sd'], '.1f'):>9} "
                     f"[{_fmt(s['ci_lo'])}, {_fmt(s['ci_hi'])}]")
    all_m = [r["margin"] for r in ok]
    s = _stats(all_m)
    wins = sum(1 for r in ok if r["outcome"] == "win")
    losses = sum(1 for r in ok if r["outcome"] == "loss")
    ties = sum(1 for r in ok if r["outcome"] == "tie")
    ties_m = [r["margin"] for r in ok if r["outcome"] == "tie"]
    lines += [
        f"{'ALL':<44} {s.get('n', 0):>3} {f'{wins}/{losses}/{ties}':>7} "
        f"{_fmt(s.get('mean', float('nan'))):>10} "
        f"{_fmt(s.get('sd', float('nan')), '.1f'):>9} "
        f"[{_fmt(s.get('ci_lo', float('nan')))}, "
        f"{_fmt(s.get('ci_hi', float('nan')))}]",
        "",
        f"win rate vs pool: {wins}/{s['n']} = "
        f"{wins / s['n']:.3f}" if s.get("n") else "win rate: n/a",
        f"ties: {ties} (counted explicitly; margin on ties "
        f"{_fmt(statistics.fmean(ties_m)) if ties_m else 'n/a'} coins)",
        "",
        f"seed count (paired, 95%, normal approx): with the pooled sd "
        f"{_fmt(s.get('sd', float('nan')), '.1f')} and mean "
        f"{_fmt(s.get('mean', float('nan')))}, resolving this margin needs "
        f"n >= {_seeds_needed(s.get('sd', float('nan')), s.get('mean', float('nan'))):.1f} "
        f"paired seeds per opponent-meeting (measured here; the 16 in "
        f"issue #18 was provisional - #20 brief 6.3)",
    ] + extra_lines
    return "\n".join(lines)


# --- CLI -----------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m offline.evaluate",
        description="Paired-seed evaluation of two agent versions (#18).")
    ap.add_argument("--a", required=True,
                    help="version A: 'main', 'agent.main:agent', or a pool slug")
    ap.add_argument("--b", required=True, help="version B (same forms)")
    ap.add_argument("--tier", default="smoke",
                    choices=sorted(TIER_OPPONENTS))
    ap.add_argument("--opponents", default=None,
                    help="comma-separated pool slugs (overrides the tier)")
    ap.add_argument("--seeds", type=int, default=None,
                    help="override the tier's seed count (seeds = range(n))")
    ap.add_argument("--seed", type=int, default=None,
                    help="a single seed (overrides --seeds and the tier)")
    ap.add_argument("--parallel", type=int, default=4,
                    help="concurrent episodes (throughput mode only; 0/1 = serial)")
    ap.add_argument("--episode-steps", type=int, default=720)
    ap.add_argument("--episode-timeout", type=float, default=600.0)
    ap.add_argument("--autopsy", type=int, default=2,
                    help="dump full per-turn JSON for the N widest B-wins "
                         "(0 disables)")
    ap.add_argument("--timing", action="store_true",
                    help="add the serial F046 timing readings (solo + "
                         "contended) after the main run")
    ap.add_argument("--label", default="", help="free text for the scoreboard")
    args = ap.parse_args(argv)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = REPO / "artifacts" / "evaluate" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    a_slug = _resolve_ref(args.a)
    b_slug = _resolve_ref(args.b)
    opponents, opp_note = _pick_opponents(args.tier, args.opponents)
    seeds = ([args.seed] if args.seed is not None
             else list(range(args.seeds if args.seeds is not None
                             else TIER_SEEDS[args.tier])))
    jobs = _paired_jobs(opponents, seeds)

    _progress(f"0/{len(jobs)} paired episodes ({args.tier} tier)")
    records = _run_all(a_slug, b_slug, jobs, args.episode_steps,
                       args.episode_timeout, args.parallel,
                       out_dir / "records.jsonl")

    extra: list[str] = []
    ok = [r for r in records if r["status"] == "ok"]

    # guard labels for our two versions (raises/malformed/slow are pool
    # instrumentation, but our agents run under the same guard)
    for name, key in (("A", "a_guard"), ("B", "b_guard")):
        gs = [r[key] for r in ok if r.get(key)]
        if gs:
            raises = sum(g.get("raises", 0) for g in gs)
            malformed = sum(g.get("malformed", 0) for g in gs)
            p95 = max(g.get("self_p95_ms", 0.0) for g in gs)
            extra.append(
                f"version {name} guard: raises {raises}, malformed "
                f"{malformed}, worst self-p95 {p95:.1f} ms "
                f"(throughput-mode reading - not a budget number, A4)")

    # loss autopsy: the N widest B-wins (B beats A by the most)
    autopsies: list[str] = []
    if args.autopsy > 0:
        b_wins = sorted((r for r in ok if r["outcome"] == "loss"),
                        key=lambda r: r["margin"])
        for r in b_wins[:args.autopsy]:
            try:
                path = _run_autopsy(a_slug, r["opponent"], r["seed"],
                                    args.episode_steps,
                                    out_dir / "autopsies", b_slug)
                autopsies.append(str(path.relative_to(REPO)))
            except Exception as exc:             # noqa: BLE001
                autopsies.append(f"autopsy failed for "
                                 f"{r['opponent']} seed {r['seed']}: {exc}")
        if autopsies:
            extra.append("loss autopsy (widest B-wins, re-run with full "
                         "per-turn capture):")
            extra += [f"  {p}" for p in autopsies]

    if args.timing:
        extra += _timing_block(a_slug, b_slug, ok, args.episode_steps)

    report = _report(a_slug, b_slug, args.tier, records, opponents, seeds,
                     opp_note, extra)
    (out_dir / "report.md").write_text(report + "\n")
    print(report)

    wins = sum(1 for r in ok if r.get("outcome") == "win")
    losses = sum(1 for r in ok if r.get("outcome") == "loss")
    ties = sum(1 for r in ok if r.get("outcome") == "tie")
    margins = [r["margin"] for r in ok]
    s = _stats(margins)
    _append_scoreboard({
        "run_id": run_id, "utc": datetime.now(timezone.utc).isoformat(),
        "label": args.label, "tier": args.tier, "a": a_slug, "b": b_slug,
        "git_sha": _git_sha(), "opponents_note": opp_note,
        "n_jobs": len(records), "n_ok": len(ok),
        "n_abandoned": len(records) - len(ok),
        "wins": wins, "losses": losses, "ties": ties,
        "win_rate": round(wins / len(ok), 4) if ok else "",
        "margin_mean": round(s["mean"], 2) if s.get("n") else "",
        "margin_sd": round(s["sd"], 2) if s.get("n", 0) >= 2 else "",
        "ci_lo": round(s["ci_lo"], 2) if s.get("n", 0) >= 2 else "",
        "ci_hi": round(s["ci_hi"], 2) if s.get("n", 0) >= 2 else "",
        "seeds_needed": (round(_seeds_needed(s["sd"], s["mean"]), 1)
                         if s.get("n", 0) >= 2 else ""),
        "records": str((out_dir / "records.jsonl").relative_to(REPO)),
    })
    print(f"\nscoreboard row appended: {SCOREBOARD.relative_to(REPO)}")
    print(f"records + report: {out_dir.relative_to(REPO)}")

    abandoned = len(records) - len(ok)
    if abandoned == 0 and ok:
        return 0
    return 3 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
