"""The build's own fingerprint: did the RECIPE change since the artifact was made?

`agent/tile_dp/contract.py` already stamps what the agent can check at runtime — the chain
registry, the engine, the day length, the key layout — and it says plainly why it
stops there:

    Nothing here reads the build's own source: the submission ships `agent/` alone,
    so a fingerprint that needs `offline_lab/` on disk cannot be computed where it
    matters.

That is right for a load. It leaves one question open for a REPO, where
`offline_lab/` does exist: somebody edits the builder and does not rebuild, and the
artifact silently becomes a picture of a recipe nobody runs any more. The contract
cannot see it, because the registry and the engine did not move — only the code that
walks them did.

This module answers exactly that, and nothing else. It is a development-time check;
it is never imported by `agent/`.

## What is hashed, and why only this

`SOURCES` is the code that decides what the graph CONTAINS, given a fixed vocabulary:

    build/graph.py     the BFS, the no-op sweep, the dominance rule, the pruning
    build/chains.py    which chains are offered to each state
    build/ledger.py    what a chain costs, in hours and in goods
    fast_sim.py        the simulator the BFS drives to find each next state

The vocabulary itself — the chain registry and the engine — is deliberately NOT here:
`info["registry"]` and `info["engine"]` already carry it, and hashing it twice would
mean two fingerprints that can disagree about one fact.

## What it cannot tell you

Whether a change was MEANINGFUL. A new docstring in `graph.py` moves this hash and
changes no edge. That is the accepted cost of a check that runs in milliseconds: it
reports "the recipe moved, and the artifact did not", which is a question, not a
verdict. The answer is either a rebuild or a restamp, and the guard's message says so.

It also cannot prove the artifact matches a rebuild — only that nothing anyone can
see has changed since it was stamped. Proving the rest means running the build, which
costs minutes and belongs to a slower cadence.
"""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path

#: Repo root: this file is `offline_lab/build/fingerprint.py`.
ROOT = Path(__file__).resolve().parents[2]

#: The build recipe, in a fixed order — the hash is over the list, so the order is
#: part of it and a rename is a change like any other.
SOURCES: tuple[str, ...] = (
    "offline_lab/build/graph.py",
    "offline_lab/build/chains.py",
    "offline_lab/build/ledger.py",
    "offline_lab/fast_sim.py",
)


def builder_fingerprint(root: Path | None = None) -> str:
    """A 16-hex digest of the build recipe, or a message naming what is missing.

    Each file contributes its path AND its bytes, so moving code between two hashed
    files still moves the digest. A missing file raises rather than hashing to
    something that looks like an answer.
    """
    base = ROOT if root is None else Path(root)
    h = sha256()
    for rel in SOURCES:
        path = base / rel
        if not path.is_file():
            raise FileNotFoundError(
                f"the build recipe is incomplete: {rel} is not a file under {base}. "
                f"Either the file moved — update SOURCES in this module — or the "
                f"checkout is partial, and no fingerprint computed from it is worth "
                f"comparing against.")
        h.update(rel.encode())
        h.update(b"\0")
        h.update(path.read_bytes())
        h.update(b"\0")
    return h.hexdigest()[:16]


def stamp() -> dict[str, object]:
    """What the builder records about its own source, for `write_info`'s `stats`.

    It lives under `stats` and not beside `contract` on purpose: `INFO_KEYS` is a
    closed tuple shared by all four artifacts, so adding a key there would invalidate
    every info file in the repo to record a fact about one of them.
    """
    return {"fingerprint": builder_fingerprint(), "sources": list(SOURCES)}
