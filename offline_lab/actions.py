"""Action-shape validation — one implementation of the truth.

Promoted from `fast_sim._validate_action` (issue #20 §2.3): the
pool guard in `offline_lab/pool/guard.py` must shape-check vendored agents
against the SAME contract our own agent gets, and the brief forbids
copying a private helper. fast_sim keeps calling this; the import moves,
the behaviour does not.

Deliberately STRICTER than the harness: the harness only requires the
action to be an object (its schema declares no typed properties and the
interpreter no-ops unknown ops), so a wrong inner type passes silently
on the real environment while dev mode raises here. A dev failure
therefore means "caller bug", not necessarily "would fail on Kaggle" —
see R004.
"""

from __future__ import annotations

from typing import Any


def validate_action(index: int, action: Any) -> None:
    """Dev-mode action shape check (see module docstring)."""
    if not isinstance(action, dict):
        raise TypeError(f"agent {index}: action must be a dict")
    for key in ("farmer", "hands", "market"):
        if key not in action:
            raise KeyError(f"agent {index}: action missing '{key}'")
    if not isinstance(action["farmer"], list):
        raise TypeError(f"agent {index}: 'farmer' must be a list")
    if not isinstance(action["hands"], list):
        raise TypeError(f"agent {index}: 'hands' must be a list")
    if not isinstance(action["market"], list):
        raise TypeError(f"agent {index}: 'market' must be a list")
