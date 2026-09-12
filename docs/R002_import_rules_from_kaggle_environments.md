# R002 — Import the game rules from kaggle_environments, do not transcribe them

**Summary (≤50 words):** Import game constants and formulas directly from
`kaggle_environments` (`CROPS`, `market_price`, …) instead of transcribing
them. The submission runs inside that environment, so re-import is free,
keeps exactly one source of truth, and eliminates a constants-parity harness.

## Decision

Re-export constants and formulas directly from the installed environment:

```py
from kaggle_environments.envs.kaggriculture.kaggriculture import CROPS, market_price, ...
```

We do not copy the crop tables, the price parameters, or the hire curve into
our own source.

## Why

The submission runs *inside* kaggle_environments — the harness imports the
environment to run the episode, then execs our agent in the same process. So
the module is guaranteed present and already in `sys.modules`; re-importing
it measured ~0.03 s, effectively free against a 1 s turn.

Given that, transcription buys nothing and costs correctness. A copied table
can disagree with the game — silently, and in a way tests written against the
copy cannot detect. Importing gives exactly one source of truth, and it is
the one the interpreter itself uses:

```py
assert rules.CROPS is K.CROPS   # the same object, not a copy
```

It also removes a whole phase of work: there is no constants-extraction task
and no constants-parity harness to maintain, because parity is true by
construction.

## Risk accepted

A kaggle-environments version bump could rename or restructure the private
helpers. The mitigation is the name test plus pinning the local version to
match the competition image. The alternative — a hand-maintained copy of the
rules — carries a larger risk that is much harder to detect.
