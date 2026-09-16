"""Arena runner: ONE episode, ONE process, records per SEAT.

Issue #20 brief §5 + addendum A1/A4. Three requirements collapse into
one decision:

1. **Thread isolation** - Kaggle-like CPU limits are environment
   variables, and a variable set after numpy is imported does nothing;
   so this module sets them at the very top, before any import, and is
   the ONLY module in the repository permitted to (a test enforces it).
2. **Agent state isolation** - vendored agents keep module-level state;
   a fresh process per episode is a fresh agent, no reload trickery.
3. **Crash isolation** - a vendored agent that segfaults a C extension
   takes down its process, not the sweep.

And the addendum's corrections, both measured on the real harness:

- **Per-seat records (A1)**: an episode is two agents; every pool-vs-pool
  episode characterises TWO agents, so the runner returns
  `{0: SeatRecord, 1: SeatRecord}` instead of one seat's view.
- **Timing has two readings (A4)**: `timing_solo` (vs PASS — the floor)
  and `timing_contended` (vs a real pool agent — the honest number the
  bank policy derives from; a competing opponent took 59 % of compute
  in the owner's measurement). The harness calls seat 0 BEFORE seat 1
  within a step, and step distance is not wall distance — the timing
  wrappers therefore assert the direction (contended >= solo), because
  an inverted measurement is a broken harness, not fast load.
"""

from __future__ import annotations

# --- thread caps: set before ANY third-party import (numpy/torch land in
# _episode_worker's imports, below), and this module is the only one in
# the repository permitted to set them (a test enforces it).
# Hard-set, not setdefault (review 1 F5): a pre-set shell variable
# silently overriding the cap would make every timing number a reading
# of the wrong machine. The effective count rides on every record.
import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
             "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_var] = "1"

import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def run_selfplay_probe(slug: str, seed: int = 0,
                       episode_steps: int = 720,
                       timeout_s: float = 600.0) -> dict:
    """The §A2 self-play P-filter: slug vs slug, same seed, one episode.

    The two seats play the SAME slug; the worker compares their action
    sequences by exact equality and returns the verdict directly
    (brief part 2 §1: classify inside the worker - 719 action tuples do
    not ride through JSON for a boolean). Identical sequences = strong P
    candidate; different = S or R, needs the full four-run table.
    """
    rec = run_episode_process(slug, slug, seed, episode_steps,
                              timeout_s=timeout_s)
    rec["selfplay"] = True
    return rec


def run_episode_process(slug0: str, slug1: str, seed: int,
                        episode_steps: int = 720,
                        timeout_s: float = 600.0) -> dict:
    """One episode in a fresh python process. Returns a per-seat record.

    The worker is `_episode_worker`, driven by -c so the env caps
    above hold in the child too. `timeout_s` bounds the whole episode:
    a pool agent that wedges aborts the episode (`abandoned`), it is
    never silently retried.
    """
    worker_code = (
        "import sys, json\n"
        "sys.path.insert(0, '.')\n"
        "from offline.runner import _episode_worker\n"
        f"rec = _episode_worker({slug0!r}, {slug1!r}, {seed}, "
        f"{episode_steps})\n"
        "print(json.dumps(rec))\n"
    )
    try:
        proc = subprocess.run(
            [sys.executable, "-c", worker_code],
            cwd=REPO, capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired as exc:
        # F3: a wedged pool agent aborts ITS episode as `abandoned`,
        # it never takes the sweep down
        return {"seats": {
                    0: {"slug": slug0, "status": "abandoned"},
                    1: {"slug": slug1, "status": "abandoned"}},
                "seed": seed, "status": "abandoned",
                "error": f"episode exceeded {timeout_s}s (abandoned)"}
    if proc.returncode != 0:
        return {"seats": {
                    0: {"slug": slug0, "status": "abandoned"},
                    1: {"slug": slug1, "status": "abandoned"}},
                "seed": seed, "status": "abandoned",
                "error": (proc.stderr.strip().splitlines() or ["?"])[-1][:200]}
    rec = json.loads(proc.stdout.strip().splitlines()[-1])
    # json round-trip turns the int seat keys into strings - normalise
    rec["seats"] = {int(k): v for k, v in rec["seats"].items()}
    rec["mode"] = "throughput"
    return rec


def run_episode_timed(slug0: str, slug1: str = "PASS-proxy", seed: int = 0,
                      episode_steps: int = 720) -> dict:
    """Throughput-shaped record tagged mode="timing" — NO per-turn readings.

    What this BUILDS: one process, one episode, the record tagged
    mode="timing" so the reporting layer can refuse on the tag. The
    per-turn readings live in `_timed_episode_worker` (used by
    `offline.evaluate`'s timing block): `timing_solo` against
    "PASS-proxy" and `timing_contended` against a real pool opponent,
    with the addendum's direction assertion (contended >= solo — a
    contended reading faster than solo is a broken harness, not fast
    load). Still open, named: the bank policy DERIVING from the
    contended number (#9's flip; F046's 1 s + 60 s accounting is in
    `bank_seconds` but nothing gates on it yet).
    """
    rec = run_episode_process(slug0, slug1, seed, episode_steps,
                              timeout_s=1200.0)
    rec["mode"] = "timing"
    return rec


def _pass_agent(obs, config=None):
    """The built-in PASS policy (the timing floor's opponent)."""
    return {"farmer": ["PASS"], "hands": [], "market": []}


def _load_agent(slug: str):
    """Resolve one runner-side agent reference (shared by both workers).

    - "PASS-proxy": the built-in PASS policy;
    - "chista-m1": OUR agent (the sweep/baseline subject, main's
      offline/pool/sweep.py slugs it); kept from main — the arena
      rebase must not drop it;
    - "ref:<module>:<attr>": an in-repo agent (issue #18 --a/--b refs);
    - otherwise: a vendored pool slug via offline.pool.loader.
    """
    from offline.pool.loader import LoadedAgent, load

    if slug == "PASS-proxy":
        return LoadedAgent(slug="PASS-proxy", fn=_pass_agent,
                           fn_name="_pass_agent", arity=2,
                           rule="builtin", module=None)
    if slug == "chista-m1":                 # OUR agent (the baseline subject)
        from agent.main import agent as our_agent
        return LoadedAgent(slug="chista-m1", fn=our_agent,
                           fn_name="agent", arity=2,
                           rule="builtin-our-agent", module=None)
    if slug.startswith("ref:"):
        # issue #18: --a/--b accept "main" / "agent.main:agent". The
        # module is OUR code, importable from the repo root (the child's
        # cwd + sys.path); no shim under opponents/ (vendored tree is
        # SHA-pinned by tests/test_opponents.py).
        import importlib
        import inspect as _inspect
        _, module_name, attr = slug.split(":")
        module_obj = importlib.import_module(module_name)
        fn = getattr(module_obj, attr)
        arity = 2 if len(list(_inspect.signature(fn).parameters.values())) >= 2 else 1
        return LoadedAgent(slug=slug, fn=fn, fn_name=attr, arity=arity,
                           rule="in-repo-ref", module=module_obj)
    return load(slug)


def _episode_worker(slug0: str, slug1: str, seed: int,
                    episode_steps: int, capture: bool = False) -> dict:
    """The child-process body: two guard-wrapped agents, one episode.

    Returns a record PER SEAT (addendum A1) - every pool-vs-pool episode
    characterises two agents. Seat 0's agent is called before seat 1's
    within a step (measured harness behaviour); both receive COPIES of
    the observation, never the live view.

    `capture=True` (issue #18 loss autopsy) additionally records the raw
    per-turn action dicts and each seat's money series - used only by
    single-episode re-runs, never in the bulk sweep (record size).
    """
    from offline.pool.guard import guarded_call, GuardStats

    agents = {0: _load_agent(slug0), 1: _load_agent(slug1)}
    stats = {0: GuardStats(), 1: GuardStats()}

    from world.fast_sim import FastSim
    sim = FastSim({"episodeSteps": episode_steps, "seed": seed,
                   "weedSpawnChance": 0.005}, validate="fast")

    actions = {0: [], 1: []}
    selfplay = slug0 == slug1        # the A2 self-play probe condition
    raw_actions: dict[int, list] = {0: [], 1: []}
    money: dict[int, list] = {0: [], 1: []}
    # one DETACHED view per seat (addendum safety + review 1 F9): the
    while not sim.done:
        # DETACHED views per seat, EVERY turn (review 2, N1): the first
        # fix detached turn 0 only - copy_state defaults off in fast
        # mode, so turns 1..719 handed out LIVE views and a mutating
        # vendored agent wrote straight into the episode. Never rely on
        # the default here: in this module's validate="fast" mode the
        # default IS the unsafe one. The harness deep-copies per agent
        # through the world/ parameter (R003), so third-party code never
        # touches live episode state.
        views = sim.observations(copy_state=True)
        # seat 0 then seat 1, the harness's own order; each sees only
        # its own view and its own private state
        a0 = guarded_call(agents[0].fn, views[0],
                          {"episodeSteps": episode_steps}, stats[0],
                          copy=False, arity=agents[0].arity)
        a1 = guarded_call(agents[1].fn, views[1],
                          {"episodeSteps": episode_steps}, stats[1],
                          copy=False, arity=agents[1].arity)
        actions[0].append(_freeze(a0))
        actions[1].append(_freeze(a1))
        if capture:
            raw_actions[0].append(a0)
            raw_actions[1].append(a1)
            m = sim.money()
            money[0].append(m[0])
            money[1].append(m[1])
        sim.step([a0, a1])
    rewards = sim.rewards()
    record = {"seed": seed, "status": "DONE", "selfplay": selfplay,
              "seats": {}}
    for seat, slug in ((0, slug0), (1, slug1)):
        record["seats"][seat] = {"slug": slug, "rewards": rewards[seat],
                                 "guard": stats[seat].labels(),
                                 "actions_hash": _stable_hash(actions[seat]),
                                 "n_actions": len(actions[seat])}
        if capture:
            record["seats"][seat]["actions"] = raw_actions[seat]
            record["seats"][seat]["money_series"] = money[seat]
    if selfplay:
        # the A2 self-play filter: identical sequences across the two
        # seats = strong P candidate (classify.py's pure comparison,
        # applied inside the worker per brief part 2 section 1)
        from offline.pool.classify import classify_from_sequences
        record["class_verdict"] = classify_from_sequences(
            actions[0], actions[1], actions[0], actions[0])
    return record


def _timed_episode_worker(slug: str, opp: str, seed: int,
                          episode_steps: int) -> dict:
    """Serial per-turn timing of ONE agent in the child process (issue #18).

    Timing is part of the score (F046): every run records per-turn wall
    time for `slug` only. The opponent may be the PASS proxy (the
    `timing_solo` floor) or a real pool agent (`timing_contended`, the
    A4-honest number). The caller tags mode/reading and enforces the
    direction assertion (contended >= solo).
    """
    from offline.pool.guard import guarded_call, GuardStats

    me = _load_agent(slug)
    other = _load_agent(opp)
    stats_me = GuardStats()
    stats_other = GuardStats()

    from world.fast_sim import FastSim
    sim = FastSim({"episodeSteps": episode_steps, "seed": seed,
                   "weedSpawnChance": 0.005}, validate="fast")

    turn_ms: list[float] = []
    while not sim.done:
        views = sim.observations(copy_state=True)
        a1 = guarded_call(other.fn, views[1],
                          {"episodeSteps": episode_steps}, stats_other,
                          copy=False, arity=other.arity)
        t0 = time.perf_counter()
        a0 = guarded_call(me.fn, views[0],
                          {"episodeSteps": episode_steps}, stats_me,
                          copy=False, arity=me.arity)
        turn_ms.append((time.perf_counter() - t0) * 1000.0)
        sim.step([a0, a1])
    turn_ms.sort()
    n = len(turn_ms)
    p50 = turn_ms[n // 2]
    p95 = turn_ms[min(n - 1, int(0.95 * n))]
    mx = turn_ms[-1]
    drawn_s, worst_over_s = bank_seconds(turn_ms)
    return {"seed": seed, "status": "DONE", "slug": slug, "opponent": opp,
            "turns": n,
            "timing_ms": {"p50": round(p50, 3), "p95": round(p95, 3),
                          "max": round(mx, 3)},
            "bank_drawn_s": round(drawn_s, 4),
            "worst_turn_over_s": round(worst_over_s, 4),
            "guard": stats_me.labels()}


def bank_seconds(turn_ms: list[float]) -> tuple[float, float]:
    """F046's runtime-budget accounting over one episode's turn times.

    Returns `(bank_drawn_s, worst_turn_over_s)`:
    - `bank_drawn_s` is the POLICY number: F046 gives 1 free second per
      turn and a 60-second bank for the episode, and bills
      `max(0, duration - 1.0)` per turn, so the draw is the SUM over
      turns (a forfeit is this exceeding 60 s). The bench's single
      `max(0, max_turn - 1.0)` is a different, weaker reading — it only
      ever equals the draw when one turn overruns and the rest do not;
    - `worst_turn_over_s` keeps that per-turn worst case, which is what
      the bench prints (labelled "from the max turn").
    """
    drawn = sum(max(0.0, ms / 1000.0 - 1.0) for ms in turn_ms)
    worst = max(0.0, (max(turn_ms) / 1000.0 - 1.0)) if turn_ms else 0.0
    return drawn, worst


def _freeze(action: dict) -> tuple:
    """Action dict -> hashable tuple (for sequence comparison)."""
    return (tuple(action["farmer"]),
            tuple(tuple(h) for h in action["hands"]),
            tuple(tuple(m) for m in action["market"]))


def _stable_hash(seq) -> str:
    """A hash that does NOT vary across processes: builtin hash() randomizes
    string hashing per process, so identical sequences compared unequal
    between worker processes. sha256 of the json dump instead."""
    import hashlib
    return hashlib.sha256(
        json.dumps(seq, sort_keys=True, default=list).encode()).hexdigest()[:16]
