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
import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
             "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# NOTE: `from __future__ import annotations` must precede other code; placed below
# the env block on purpose (docstring + env caps first). Python allows it anywhere
# before usage; keep it adjacent to the typing imports that need it.

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
    proc = subprocess.run(
        [sys.executable, "-c", worker_code],
        cwd=REPO, capture_output=True, text=True, timeout=timeout_s)
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
    """Timing mode: exactly one process, serial, per-seat readings.

    `timing_solo` = the seat's turn time against the PASS proxy (the
    floor); `timing_contended` = against a real pool agent (the honest
    number the bank policy derives from - addendum A4). The reporting
    layer refuses contended < solo: an inverted reading means the
    harness is broken, not that load helps.
    """
    rec = run_episode_process(slug0, slug1, seed, episode_steps,
                              timeout_s=1200.0)
    rec["mode"] = "timing"
    return rec


def _episode_worker(slug0: str, slug1: str, seed: int,
                    episode_steps: int) -> dict:
    """The child-process body: two guard-wrapped agents, one episode.

    Returns a record PER SEAT (addendum A1) - every pool-vs-pool episode
    characterises two agents. Seat 0's agent is called before seat 1's
    within a step (measured harness behaviour); both receive COPIES of
    the observation, never the live view.
    """
    from offline.pool.loader import load
    from offline.pool.guard import guarded_call, GuardStats

    def _pass_agent(obs, config=None):
        return {"farmer": ["PASS"], "hands": [], "market": []}

    def _load(slug):
        if slug == "PASS-proxy":      # the timing floor's opponent
            from offline.pool.loader import LoadedAgent
            return LoadedAgent(slug="PASS-proxy", fn=_pass_agent,
                               fn_name="_pass_agent", arity=2,
                               rule="builtin", module=None)
        return load(slug)

    agents = {0: _load(slug0), 1: _load(slug1)}
    stats = {0: GuardStats(), 1: GuardStats()}

    from world.fast_sim import FastSim
    sim = FastSim({"episodeSteps": episode_steps, "seed": seed,
                   "weedSpawnChance": 0.005}, validate="fast")

    actions = {0: [], 1: []}
    obs = sim.observations()[0]
    while not sim.done:
        # seat 0 then seat 1, the harness's own order; each receives a
        # copy (guarded_call) - neither sees the other's live mutation
        a0 = guarded_call(agents[0].fn, obs,
                          {"episodeSteps": episode_steps}, stats[0])
        a1 = guarded_call(agents[1].fn, obs,
                          {"episodeSteps": episode_steps}, stats[1])
        actions[0].append(_freeze(a0))
        actions[1].append(_freeze(a1))
        sim.step([a0, a1])
        obs = sim.observations()[0]
    rewards = sim.rewards()
    return {"seed": seed, "status": "DONE",
            "seats": {
                0: {"slug": slug0, "rewards": rewards[0],
                    "guard": stats[0].labels(),
                    "actions_hash": _stable_hash(actions[0]),
                    "n_actions": len(actions[0])},
                1: {"slug": slug1, "rewards": rewards[1],
                    "guard": stats[1].labels(),
                    "actions_hash": _stable_hash(actions[1]),
                    "n_actions": len(actions[1])}}}


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
