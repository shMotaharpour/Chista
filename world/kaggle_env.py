"""Full-path wrapper: the genuine Kaggle environment for kaggriculture.

Runs episodes inside the real kaggle-environments harness (schema-validated
actions, timeout accounting, per-turn snapshots) with arbitrary
configuration and agents supplied at call time, then renders the game to an
HTML file through the environment's own render() method.

This is the slow, official path — use it for submission-style evaluation
and replays. For bulk simulation (DP/MDP/RL), use world.fast_sim instead
(per R003; measured on a 720-step season: 1.785 s/episode here vs 0.035 s in
world.fast_sim, i.e. 50.6x — reproduce with bench/bench_paths.py).
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Any, Callable

from kaggle_environments import make as ke_make
from kaggle_environments.core import Environment

# Ensure our own repo root is importable so "agent:<name>" style file agents
# or helper modules resolve consistently regardless of the caller's cwd.
REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Replays are evidence and must outlive a reboot, so the default lives inside the
# repo (gitignored) rather than /tmp; pass output_path to override.
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "artifacts" / "replays"

AgentLike = Callable[[Any], Any] | str


def new_environment(configuration: dict[str, Any] | None = None,
                    debug: bool = False) -> Environment:
    """Build a fresh kaggriculture Environment via the official make().

    `configuration` accepts any subset of the game's configuration schema
    (episodeSteps, boardSize, startingMoney, seed, marketParams, ...).
    """
    return ke_make("kaggriculture", configuration=configuration or {}, debug=debug)


def run_episode(agents: list[AgentLike],
                configuration: dict[str, Any] | None = None,
                debug: bool = False) -> Environment:
    """Run a full episode with the real harness and return the Environment.

    `agents` is a list of two callables or strings (e.g. ["random", my_agent]);
    each callable receives the observation and returns the action dict.
    The returned Environment carries env.steps (per-turn snapshots),
    env.rewards, and env.info — including any resolved episode seed.
    """
    if len(agents) != 2:
        raise ValueError("kaggriculture is a two-player game: pass exactly 2 agents")
    env = new_environment(configuration, debug=debug)
    env.run(agents)
    return env


def render_episode_html(env: Environment,
                        output_path: str | Path | None = None,
                        overwrite: bool = False) -> Path:
    """Render a finished episode to an HTML file via env.render(mode="html").

    Default output is a tmp dir (/tmp/chistaagent/replays) per project
    convention: outputs are ephemeral unless the caller asks for a
    persistent path by passing `output_path` explicitly.
    """
    path = Path(output_path) if output_path is not None else _default_html_path(env)
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists; pass overwrite=True to replace it")
    html = env.render(mode="html")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")
    return path


def _default_html_path(env: Environment) -> Path:
    """Collision-free default replay name.

    Keyed on the seed and on a fingerprint of the episode's outcome, so two runs
    that share a seed (the normal case in a sweep) no longer overwrite each
    other's replay. Two runs collide only when the episode is genuinely
    identical, which makes run_and_render's overwrite=True idempotent instead of
    lossy.
    """
    final = env.steps[-1] if env.steps else []
    signature = "|".join(str(getattr(s, "reward", None)) for s in final)
    fingerprint = hashlib.sha1(f"{len(env.steps)}:{signature}".encode()).hexdigest()[:8]
    stem = (f"kaggriculture_s{env.info.get('seed', 'NA')}"
            f"_{len(env.steps)}steps_{fingerprint}")
    return DEFAULT_OUTPUT_DIR / f"{stem}.html"


def run_and_render(agents: list[AgentLike],
                   configuration: dict[str, Any] | None = None,
                   output_path: str | Path | None = None,
                   overwrite: bool = True,
                   debug: bool = False) -> tuple[Environment, Path]:
    """Convenience: run a full episode, then write its HTML replay.

    Returns (env, html_path).
    """
    env = run_episode(agents, configuration=configuration, debug=debug)
    html_path = render_episode_html(env, output_path=output_path, overwrite=overwrite)
    return env, html_path
