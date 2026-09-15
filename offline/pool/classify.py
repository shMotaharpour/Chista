"""Class probe: P (open-loop) / S (self-reactive) / R (reactive), behaviourally.

Issue #20 brief §3. Static scans lie (adaptive-replay-agent shows zero
obs-reads in 550 lines), so class is MEASURED: the agent's full action
sequence is recorded under four conditions and compared by exact
equality of the action dicts at every step.

Vendored agents keep module-level state across turns, so each trajectory
runs in a FRESH process (the runner's one-process-per-episode design
provides that; classify.py drives the runner).

| run | seat | seed | opponent |
|-----|------|------|----------|
| S1  | 0    | A    | PASS     |
| S2  | 0    | A    | active pool agent |
| S3  | 0    | B    | PASS     |
| S4  | 1    | A    | PASS     |

P: S1=S2=S3=S4 · S: S1=S2 but S1!=S3 or S1!=S4 · R: S1!=S2
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ClassVerdict:
    slug: str
    cls: str                    # "P" | "S" | "R"
    s1_eq_s2: bool
    s1_eq_s3: bool
    s1_eq_s4: bool


def classify_from_sequences(s1, s2, s3, s4) -> str:
    """Pure comparison of four action sequences -> P / S / R."""
    def eq(a, b):
        return len(a) == len(b) and all(
            _same_action(x, y) for x, y in zip(a, b))

    s1_eq_s2 = eq(s1, s2)
    s1_eq_s3 = eq(s1, s3)
    s1_eq_s4 = eq(s1, s4)
    if s1_eq_s2 and s1_eq_s3 and s1_eq_s4:
        return "P"
    if s1_eq_s2:
        return "S"
    return "R"


def _same_action(a, b) -> bool:
    """Exact equality of two action dicts (order-sensitive on lists)."""
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(
            _same_action(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return len(a) == len(b) and all(
            _same_action(x, y) for x, y in zip(a, b))
    return a == b
