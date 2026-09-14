# Opponent agents

Third-party agents — competitor code published on Kaggle — kept as **sparring
partners** for `lab/eval/arena.py`. They never ship; they exist to give our
strategies something real to lose to (roadmap 2.3).

Where they came from and under what licence is stated once, for all of them, in
[`NOTICE.md`](NOTICE.md). Read that before committing anything here.

## Layout

```
lab/opponents/
├── NOTICE.md                 where they came from, and under what licence
├── LICENSE-Apache-2.0.txt    a verbatim copy of that licence
├── extract.py                pulls an agent out of a notebook, without running it
└── <slug>/
    ├── agent.py              their code, byte-for-byte as published
    └── SOURCE.md             which notebook, how it was packed, its SHA-256,
                              what the audit found, how to play it
```

The split is deliberate. A per-agent `SOURCE.md` says only what differs between
agents; the licence and the source page are the same for all twenty, so they are
stated once in `NOTICE.md` rather than twenty times where they could drift.

## Adding one

```bash
./run opponents "/path/to/notebooks"           # extract everything it can
./run opponents "/path/to/notebooks" --unpack  # also decode their route tables
```

`extract.py` parses notebooks with `ast` and **never executes them** — they are
arbitrary code published by strangers, and running one to find out what it does
is the wrong order. It reports any import or call that reaches outside the
process; read those before playing the agent, because the arena runs it in our
interpreter.

Re-running the extractor never overwrites a `SOURCE.md` that already exists — if
a re-extraction changes an `agent.py`, the new block lands beside it as
`SOURCE.md.regenerated` and the stale hash fails the tests until they agree.

## Rules

1. **Do not edit their logic.** `agent.py` is the published payload byte-for-byte
   and its SHA-256 is recorded in its `SOURCE.md`; `tests/test_opponents.py`
   checks them and `pyproject.toml` keeps ruff away. If one needs a shim to run, put the
   shim in a separate file and say so.
2. **Never import `lab/opponents/` from `agent/`.** These are inputs to
   evaluation, not to the submission (`tests/test_layering.py` enforces it).
3. **Keep them small.** Decoded route tables go in `<slug>/unpacked/`, which is
   gitignored — one agent unpacks to 11 MB, well past the 5 MiB commit guard.

## Using them

```bash
./run arena --opponent opp:<slug> --seeds 12   # win rate, both seatings
./run profile                                  # what every one of them does
./run bench                                    # the scoreboard, including the field
```

Built-in baselines need no directory: `"random"`, `"pass"` and `"starter"` come
from the environment itself.
