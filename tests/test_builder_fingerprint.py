"""Did the build recipe move without the artifact being rebuilt?

`agent/tile_dp/contract.py` stamps what a LOAD can check — the registry, the engine,
the day length, the key layout — and stops there on purpose, because the submission
ships `agent/` alone and a fingerprint over `offline_lab/` cannot be computed where a
load happens.

That leaves one failure this repo can see and the contract cannot: somebody edits the
builder, does not rebuild, and `agent/artifact/tile_graph.npz` quietly becomes a
picture of a recipe nobody runs. The registry did not move and the engine did not
move, so `contract` is still satisfied — only the code that walks them changed.

These guards cost milliseconds. The expensive question — "does a rebuild reproduce
this artifact byte for byte?" — is a different test at a different cadence; it takes
about three minutes and is not here.

Run:  .venv/bin/python -m pytest tests/test_builder_fingerprint.py
"""

from __future__ import annotations

import pytest

from agent.artifact import read_info
from offline_lab.build.fingerprint import SOURCES, builder_fingerprint, stamp


def test_the_artifact_was_built_by_the_recipe_that_is_on_disk() -> None:
    """The one that matters: stored fingerprint == the recipe's fingerprint now.

    A mismatch is a QUESTION, not a verdict. The recipe moved and the artifact did
    not, and only two things make that right: rebuild, or — when the edit provably
    cannot move an edge, a docstring or a rename — restamp. The message says both,
    because a guard that reports a problem without naming the exits gets silenced.
    """
    stored = read_info("tile_graph")["stats"].get("builder")
    assert stored is not None, (
        "agent/artifact/tile_graph.json carries no builder fingerprint. Rebuild "
        "with `python -m offline_lab.build.graph`, which stamps one.")

    now = builder_fingerprint()
    assert stored["fingerprint"] == now, (
        f"the build recipe has moved since the artifact was stamped:\n"
        f"    stamped in the artifact : {stored['fingerprint']}\n"
        f"    the recipe on disk now  : {now}\n"
        f"    files hashed            : {', '.join(SOURCES)}\n"
        f"This does not prove the artifact is wrong — a docstring moves this hash "
        f"and no edge with it. It proves nobody has checked. Either rebuild "
        f"(`python -m offline_lab.build.graph`, ~3 min) or, if the change cannot "
        f"affect the graph, re-stamp the info file and say so in the commit.")


def test_the_stamp_records_which_files_it_hashed() -> None:
    """A bare hash ages badly: a year later nobody knows what it covered.

    The stamp carries the file list, so a fingerprint that fails can be read against
    the recipe it was taken from rather than against today's guess at it.
    """
    stored = read_info("tile_graph")["stats"]["builder"]
    assert stored["sources"] == list(SOURCES), (
        f"the artifact was stamped over a different file list than the one this "
        f"module hashes now:\n  stamped: {stored['sources']}\n  now:     {list(SOURCES)}\n"
        f"The fingerprints are not comparable until the artifact is rebuilt.")


def test_editing_a_hashed_file_moves_the_fingerprint(tmp_path) -> None:
    """The guard's whole premise, exercised: a changed recipe is a changed hash.

    Built against a copy of the tree rather than the real one, so the check that the
    fingerprint responds to an edit never edits the repository to prove it.
    """
    for rel in SOURCES:
        dst = tmp_path / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text("# recipe\n")
    before = builder_fingerprint(tmp_path)

    (tmp_path / SOURCES[0]).write_text("# recipe\n# one more line\n")
    after = builder_fingerprint(tmp_path)
    assert before != after, (
        f"{SOURCES[0]} changed and the fingerprint did not: {before}")


def test_moving_code_between_two_hashed_files_still_moves_it(tmp_path) -> None:
    """Paths are hashed with the bytes, so the same text in a different file is a
    different recipe. Without the path in the digest, cutting a function out of
    `graph.py` and pasting it into `ledger.py` would read as no change at all."""
    for rel in SOURCES:
        dst = tmp_path / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text("")
    (tmp_path / SOURCES[0]).write_text("def rule(): ...\n")
    before = builder_fingerprint(tmp_path)

    (tmp_path / SOURCES[0]).write_text("")
    (tmp_path / SOURCES[1]).write_text("def rule(): ...\n")
    assert builder_fingerprint(tmp_path) != before, (
        "the same code moved between two hashed files and the fingerprint held")


def test_a_missing_recipe_file_raises_instead_of_hashing_a_hole(tmp_path) -> None:
    """A partial checkout must not produce a digest that merely looks like one.

    Hashing what happens to be present would compare cleanly against nothing and
    report a recipe that was never read.
    """
    (tmp_path / SOURCES[0]).parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / SOURCES[0]).write_text("# only this one\n")
    with pytest.raises(FileNotFoundError, match=SOURCES[1]):
        builder_fingerprint(tmp_path)


def test_the_stamp_is_what_the_builder_writes() -> None:
    """`stamp()` is the builder's own call, so the guard reads the same shape the
    build writes rather than a second opinion about it."""
    s = stamp()
    assert set(s) == {"fingerprint", "sources"}
    assert s["fingerprint"] == builder_fingerprint()
    assert s["sources"] == list(SOURCES)
