"""Pool guard: wrap a vendored agent so nothing it does takes the sweep down.

Issue #20 §2.3. A vendored agent is untrusted code: it may raise, return
None, return a malformed action, or run long - and none of that may end
an episode. The wrapper contains all four and turns them into LABELS:

- `raises`: count + first traceback (an agent that raised on turn 300
  still produced 300 turns of data; how OFTEN it raises is itself a
  characterisation number);
- `malformed`: the returned action fails offline_lab.actions.validate_action;
- `slow`: per-call wall time p95 above the 1 s free turn (F046) - it
  would time out on Kaggle, so its score carries the label;
- `abandoned`: the episode's process timeout tripped (offline_lab.runner) -
  M1 has no in-call ceiling; the per-call one is a named TODO.

The substituted action on any failure is the same PASS our own agent's
bottom rung uses.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from offline_lab.actions import validate_action

# Hard per-turn ceiling for a pool agent: named TODO (R005) - the
# number below has NO measurement behind it; the M1 bench distribution
# under the grader's measured conditions sets the real ceiling. It is
# unused in M1 (the runner's episode timeout owns the hard stop).
TURN_CEILING_S = 2.0
PASS = {"farmer": ["PASS"], "hands": [], "market": []}


def copy_observation(obs):
    """A deep copy of the observation, handed to a third-party agent.

    Pool-analysis safety note (issue #20 comment 1): six vendored agents
    exec payloads at import and one installs sys.modules entries; fast
    mode hands out LIVE observation views, and a stray mutation by a
    competitor corrupts the evaluation silently (F047). The wrapper
    therefore hands out copies, never the live view. The copy cost is
    measured once per season by the bench, not assumed.
    """
    import copy
    return copy.deepcopy(obs)


@dataclass
class GuardStats:
    raises: int = 0
    first_traceback: str | None = None
    malformed: int = 0
    first_malformed: str | None = None
    slow_calls: int = 0
    call_times_ms: list = field(default_factory=list)
    abandoned: bool = False

    def labels(self) -> dict:
        """The per-agent label set the registry stores."""
        times = sorted(self.call_times_ms)
        p95 = times[min(len(times) - 1, int(0.95 * len(times)))] if times else 0.0
        return {
            "raises": self.raises,
            "first_traceback": self.first_traceback,
            "malformed": self.malformed,
            "first_malformed": self.first_malformed,
            "slow": p95 > 1000.0,
            "self_p95_ms": round(p95, 3),
            "self_max_ms": round(times[-1], 3) if times else 0.0,
            "abandoned": self.abandoned,
        }


def guarded_call(fn, obs, configuration, stats: GuardStats,
                 copy: bool = True, arity: int | None = None) -> dict:
    """One guarded call: contain, validate, time. Returns a legal dict.

    `arity` comes from the LOADER's resolution (LoadedAgent.arity) - the
    decision is made once at load, not re-inspected per call (720 x 2
    per episode). Callers holding a bare callable may omit it; the
    signature is inspected then. The observation handed over is the
    caller's responsibility (offline_lab.runner passes detached per-seat
    views); `copy=True` deep-copies here for callers holding a live
    view. The per-episode hard stop is the RUNNER's process timeout -
    `abandoned` there - because a thread cannot be killed.
    """
    t0 = time.perf_counter()
    agent_obs = copy_observation(obs) if copy else obs
    try:
        wants_two = _wants_two(fn) if arity is None else arity == 2
        action = (fn(agent_obs, configuration) if wants_two
                  else fn(agent_obs))  # noqa: kept for plain callables
    except Exception as exc:                     # noqa: BLE001 - containment IS the job
        stats.raises += 1
        if stats.first_traceback is None:
            import traceback
            stats.first_traceback = traceback.format_exc(limit=4)
        return dict(PASS)
    dt_ms = (time.perf_counter() - t0) * 1000.0
    stats.call_times_ms.append(dt_ms)
    if dt_ms > 1000.0:
        stats.slow_calls += 1
    try:
        validate_action(0, action)
    except (TypeError, KeyError) as exc:
        stats.malformed += 1
        if stats.first_malformed is None:
            stats.first_malformed = f"{type(exc).__name__}: {exc}"
        return dict(PASS)
    return action


def _wants_two(fn) -> bool:
    """Arity by inspection (never by parameter name): 2+ params -> pass config."""
    import inspect
    return len(inspect.signature(fn).parameters.values()) >= 2
