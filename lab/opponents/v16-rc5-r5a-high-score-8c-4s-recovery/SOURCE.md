# v16-rc5-r5a-high-score-8c-4s-recovery

Sparring partner for `./run arena`. **Not our code** — see
[../NOTICE.md](../NOTICE.md) for where it came from and under what licence.

## Extraction

| | |
|---|---|
| Source notebook | `v16-rc5-r5a-high-score-8c-4s-recovery.ipynb` |
| Packing style | writefile — `%%writefile` cell body |
| Extracted | 2026-08-27 |
| Size | 412 lines, 23,912 bytes |
| SHA-256 | `7f87c941af3050d0f21376f2843b324d7a06a1a8c050fa554cf07a769e5c937c` |

`agent.py` is the payload byte-for-byte; `tests/test_opponents.py` checks it
against that hash, and re-running the extractor must reproduce it. Extraction is
`lab/opponents/extract.py`, which parses the notebook with `ast` and **never
executes it**.

## Audit

The arena runs this inside our own interpreter, so its imports and calls were
checked before it was ever called:

Clean — every import and call is within the allowlist.

## Play it

```bash
./run arena --opponent opp:v16-rc5-r5a-high-score-8c-4s-recovery --seeds 12
./run profile                                   # what it actually does
```
