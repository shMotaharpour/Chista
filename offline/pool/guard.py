"""Pool guard: wrap a vendored agent so nothing it does takes the sweep down.

Issue #20 §2.3. A vendored agent is untrusted code: it may raise, return
None, return a malformed action, or run long - and none of that may end
an episode. The wrapper contains all four and turns them into LABELS:

- `raises`: count + first traceback (an agent that raised on turn 300
  still produced 300 turns of data; how OFTEN it raises is itself a
  characterisation number);
- `malformed`: the returned action fails world.actions.validate_action;
- `slow`: per-call wall time p95 above the 1 s free turn (F046) - it
  would time out on Kaggle, so its score carries the label;
- `abandoned`: a call hit the hard per-turn ceiling and the episode was
  given up rather than silently retried.

The substituted action on any failure is the same PASS our own agent's
bottom rung uses.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from world.actions import validate_action

# Hard per-turn ceiling for a pool agent (F046's 1 s free turn plus a
# measured margin - brief 5.1 labels this a conservative decision, and
# every timing number carries the note that it is not a Kaggle reading).
TURN_CEILING_S = 2.0
PASS = {"farmer": ["PASS"], "hands": [], "market": []}


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
            "malformed": self.malformed,
            "slow": p95 > 1000.0,
            "self_p95_ms": round(p95, 3),
            "self_max_ms": round(times[-1], 3) if times else 0.0,
            "abandoned": self.abandoned,
        }


def guarded_call(fn, obs, configuration, stats: GuardStats,
                 ceiling_s: float = TURN_CEILING_S) -> dict:
    """One guarded call: contain, validate, time. Returns a legal dict.

    The ceiling is enforced by the RUNNER's process timeout (a thread
    cannot be killed); this function is what the runner's worker calls
    and `abandoned` is set by the runner when the ceiling trips at the
    process level.
    """
    t0 = time.perf_counter()
    try:
        action = fn(obs, configuration) if _wants_two(fn) else fn(obs)
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
