"""Competitor agents, vendored from published Kaggle notebooks.

Each `<slug>/` holds `agent.py` (their code, byte-for-byte) and `SOURCE.md`
(where it came from, what its licence is, and what the audit found). Extraction
is `extract.py`, which parses notebooks with `ast` and never executes them.

These are sparring partners for `lab/eval/arena.py`. roadmap 2.3 exists because
`random` and `starter` are trivial: beating them establishes that our executor
works, not that our strategy does.
"""
