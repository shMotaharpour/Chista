"""Competitor agents, vendored from published Kaggle notebooks.

Each `<slug>/` holds `agent.py` (their code, byte-for-byte) and `SOURCE.md`
(where it came from, what its licence is, and what the audit found). Extraction
is `extract.py`, which parses notebooks with `ast` and never executes them.

These are sparring partners for evaluation, and they are the reason the arena
means anything: `random` and `starter` are trivial, so beating them establishes
that our executor works, not that our strategy does.

Nothing here ships. `tests/test_layering.py` keeps it out of the submission and
`tests/test_opponents.py` keeps the payloads honest.
"""

# How many agents this directory holds. One number, in one place, checked by
# `tests/test_opponents.py` against the directories actually present -- the
# prose used to carry its own count, and disagreed.
AGENT_COUNT = 19
