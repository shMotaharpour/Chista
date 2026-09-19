"""Guards on the vendored competitor agents.

Run:  .venv/bin/python -m tests.test_opponents

`opponents/<slug>/agent.py` is someone else's published code, kept byte-for-byte
so it stays evidence rather than a paraphrase. `opponents/NOTICE.md` satisfies
Apache 2.0 section 4(b) by claiming there are no significant changes, and that
claim is only as good as a check: it has been wrong once already, when a
`.strip()` in the extractor silently cost a trailing byte. These are that check.

The audit tests guard the other half. Six of the agents carry their real logic
base85- or zlib-packed and `exec` it at import, so a scan of the outer file
describes a loader, not an agent -- and "audited, nothing found" is
indistinguishable from "never decoded" unless coverage is asserted separately.
The arena runs this code in our own interpreter, which is why any of it matters.
"""
from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OPPONENTS = REPO / "opponents"

sys.path.insert(0, str(OPPONENTS))
import extract  # noqa: E402 - vendored directory is not a package on the path

# Repo-relative path prefixes worth checking when they appear in the docs.
DOC_PATH = re.compile(r"`((?:opponents|tests|world|docs|tile_dp|bench|lab|agent)/[\w./-]+)`")


def slugs() -> list[Path]:
    return sorted(p for p in OPPONENTS.iterdir() if p.is_dir() and (p / "agent.py").exists())


def test_every_slug_is_complete() -> None:
    incomplete = [p.name for p in OPPONENTS.iterdir()
                  if p.is_dir() and p.name != "__pycache__"
                  and not ((p / "agent.py").exists() and (p / "SOURCE.md").exists())]
    assert not incomplete, f"slug missing agent.py or SOURCE.md: {incomplete}"


def test_agent_count_matches_the_declared_one() -> None:
    from opponents import AGENT_COUNT  # noqa: PLC0415 - read late, after the path guard
    found = len(slugs())
    assert found == AGENT_COUNT, (
        f"{found} slugs on disk, opponents/__init__.py declares {AGENT_COUNT}: "
        "an agent was added or dropped without the docs following"
    )


def test_payload_hashes_match_what_source_md_records() -> None:
    bad = []
    for slug in slugs():
        recorded = re.search(r"\b([0-9a-f]{64})\b", (slug / "SOURCE.md").read_text())
        if recorded is None:
            bad.append(f"{slug.name}: SOURCE.md records no SHA-256")
            continue
        actual = hashlib.sha256((slug / "agent.py").read_bytes()).hexdigest()
        if actual != recorded.group(1):
            bad.append(f"{slug.name}: recorded {recorded.group(1)[:16]}… actual {actual[:16]}…")
    assert not bad, (
        "vendored payload does not match its recorded hash — Apache 2.0 4(b) "
        "says we changed nothing, so either restore the bytes or stop claiming it:\n  "
        + "\n  ".join(bad)
    )


def test_no_agent_reaches_outside_the_process() -> None:
    reaching = []
    for slug in slugs():
        source = (slug / "agent.py").read_text(errors="replace")
        reaching += [f"{slug.name}: {f}" for f in extract.audit(source)
                     if "REACHES OUTSIDE" in f]
    assert not reaching, (
        "an agent imports something that reaches the network, filesystem or a "
        "subprocess, and the arena runs it in our interpreter:\n  "
        + "\n  ".join(reaching)
    )


def test_packed_agents_actually_get_decoded() -> None:
    """Coverage, not findings: an undecoded payload audits as perfectly clean."""
    missed = []
    for slug in slugs():
        source = (slug / "agent.py").read_text(errors="replace")
        packs = "b85decode" in source or "b64decode" in source
        if packs and not extract.payload_digests(source):
            missed.append(slug.name)
    assert not missed, (
        "these agents pack a payload the audit never decoded, so nothing inside "
        "them was ever scanned: " + ", ".join(missed)
    )


def test_every_duplicate_payload_is_declared() -> None:
    """Two slugs, one payload is one agent counted twice: it inflates every
    count and silently shrinks the dev set. The vendored directories stay
    (provenance is information), but the duplication must be DECLARED in
    offline/pool/registry.py's DUPLICATE_OF - an undeclared duplicate
    fails here."""
    by_hash: dict[str, list[str]] = {}
    for slug in slugs():
        digest = hashlib.sha256((slug / "agent.py").read_bytes()).hexdigest()
        by_hash.setdefault(digest, []).append(slug.name)
    dupes = {h: names for h, names in by_hash.items() if len(names) > 1}
    from offline_lab.pool.registry import DUPLICATE_OF
    undeclared = {}
    for digest, names in dupes.items():
        # every slug with a twin must be a KEY in DUPLICATE_OF naming its
        # canonical sibling - a group-level edge would let an undeclared
        # member hide behind another pair's declaration
        # each member must be DECLARED: a key mapping to a sibling, or
        # the canonical target of a sibling's declaration
        missing = [n for n in names
                   if DUPLICATE_OF.get(n) not in names
                   and n not in DUPLICATE_OF.values()]
        if missing:
            undeclared[digest] = missing
    assert not undeclared, (
        "duplicate payloads exist but are not declared in "
        "registry.DUPLICATE_OF - one agent counted twice inflates every "
        "pool count:\n  "
        + "\n  ".join(f"{h[:12]}… -> {names}" for h, names
                       in sorted(undeclared.items()))
    )


def test_docs_do_not_point_at_paths_that_do_not_exist() -> None:
    """The docs arrived describing AgriOracle's layout; keep them describing this one."""
    dangling = []
    for doc in sorted(OPPONENTS.rglob("*.md")):
        for match in DOC_PATH.finditer(doc.read_text(errors="replace")):
            target = match.group(1)
            if not (REPO / target).exists():
                dangling.append(f"{doc.relative_to(REPO)} -> {target}")
    assert not dangling, (
        "documentation references paths this repository does not have:\n  "
        + "\n  ".join(dangling)
    )


def main() -> int:
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
            except Exception as exc:  # noqa: BLE001 - test runner
                failures += 1
                print(f"FAIL {name}: {type(exc).__name__}: {exc}")
            else:
                print(f"PASS {name}")
    if failures:
        print(f"{failures} test(s) failed")
        return 1
    print(f"all opponent guards passed ({len(slugs())} agents)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
