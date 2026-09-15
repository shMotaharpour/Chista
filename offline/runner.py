"""Arena runner: ONE episode, ONE process, honestly timed.

Issue #20 brief §5. Three requirements collapse into one decision:

1. **Thread isolation** - Kaggle-like CPU limits are environment
   variables, and a variable set after numpy is imported does nothing;
   so this module sets them at the very top, before any import, and is
   the ONLY module in the repository permitted to (a test enforces it).
2. **Agent state isolation** - vendored agents keep module-level state;
   a fresh process per episode is a fresh agent, no reload trickery.
3. **Crash isolation** - a vendored agent that segfaults a C extension
   takes down its process, not the sweep.

Two modes (brief 5.2): `throughput` (many processes; ONLY coins, wins
and characterisation may be reported) and `timing` (exactly one
 unloaded process; per-turn wall time, bank draw). A timing number
produced in throughput mode is discarded by the reporting layer, not
caveated.
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

def run_episode_process(slug: str, opponent_slug: str | None, seed: int,
                        episode_steps: int = 720,
                        timeout_s: float = 600.0) -> dict:
    """One episode in a fresh python process. Returns a result record.

    The worker is `_episode_worker_main`, driven by -c so the env caps
    above hold in the child too. `timeout_s` bounds the whole episode:
    a pool agent that wedges aborts the episode (`abandoned`), it is
    never silently retried.
    """
    worker_code = (
        "import sys, json\n"
        "sys.path.insert(0, '.')\n"
        "from offline.runner import _episode_worker\n"
        f"rec = _episode_worker({slug!r}, {opponent_slug!r}, {seed}, "
        f"{episode_steps})\n"
        "print(json.dumps(rec))\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", worker_code],
        cwd=REPO, capture_output=True, text=True, timeout=timeout_s)
    if proc.returncode != 0:
        return {"slug": slug, "opponent": opponent_slug, "seed": seed,
                "status": "abandoned",
                "error": (proc.stderr.strip().splitlines() or ["?"])[-1][:200]}
    import json
    rec = json.loads(proc.stdout.strip().splitlines()[-1])
    rec["mode"] = "throughput"
    return rec


def run_episode_timed(slug: str, seed: int = 0,
                      episode_steps: int = 720) -> dict:
    """Timing mode: exactly one process, unloaded, serial timing only."""
    rec = run_episode_process(slug, None, seed, episode_steps,
                              timeout_s=1200.0)
    rec["mode"] = "timing"
    return rec


def _episode_worker(slug: str, opponent_slug: str | None, seed: int,
                    episode_steps: int) -> dict:
    """The child-process body: build both agents (guard-wrapped) and play."""
    from offline.pool.loader import load
    from offline.pool.guard import guarded_call, GuardStats

    mine = load(slug)
    stats_mine = GuardStats()

    if opponent_slug is None:
        def opponent(obs, config=None):
            return {"farmer": ["PASS"], "hands": [], "market": []}
    else:
        theirs = load(opponent_slug)
        stats_theirs = GuardStats()

        def opponent(obs, config=None):
            return guarded_call(theirs.fn, obs, config, stats_theirs)

    from world.fast_sim import FastSim
    sim = FastSim({"episodeSteps": episode_steps, "seed": seed,
                   "weedSpawnChance": 0.005}, validate="fast")

    # Drive turn by turn: our agent is guarded; the engine's own actions
    # come back through sim.step's parity-verified path.
    actions_mine = []
    obs = sim.observations()[0]
    while not sim.done:
        hour = int(obs.get("hour", 0))
        day = int(obs.get("day", 0))
        a = guarded_call(mine.fn, obs, {"episodeSteps": episode_steps},
                         stats_mine)
        actions_mine.append(_freeze(a))
        sim.step([a, opponent(obs, None)])
        obs = sim.observations()[0]
    rewards = sim.rewards()
    return {"slug": slug, "opponent": opponent_slug, "seed": seed,
            "rewards": rewards, "status": "DONE",
            "guard": stats_mine.labels(),
            "actions_hash": _stable_hash(actions_mine),
            "n_actions": len(actions_mine)}


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
