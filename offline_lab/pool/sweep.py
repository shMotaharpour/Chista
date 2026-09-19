"""offline_lab.pool sweep: the characterisation sweep driver (issue #20 part 2).

Produces the registry: classification (P/S/R) via the self-play
pre-filter + the full four-run table for reactive candidates, then the
chista-m1 baseline against the whole pool.

Run:  .venv/bin/python -m offline_lab.pool.sweep
Writes: /tmp/pool_final.json (the registry source), /tmp/baseline_*.json
"""

from __future__ import annotations

import json

from offline_lab.pool.loader import slugs
from offline_lab.pool.registry import REGISTRY
from offline_lab.runner import run_episode_process


def selfplay_prefilter(episode_steps: int = 96) -> dict:
    """One slug-vs-slug episode per slug (A2 pre-filter, ~30 s each)."""
    results = {}
    for slug in slugs():
        rec = run_selfplay(slug, episode_steps)
        results[slug] = rec.get("class_verdict", "ERROR")
    return results


def run_selfplay(slug: str, episode_steps: int = 96) -> dict:
    from offline_lab.runner import run_selfplay_probe
    return run_selfplay_probe(slug, seed=0, episode_steps=episode_steps,
                              timeout_s=240)
