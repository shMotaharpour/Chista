# 25-27-strict-future-v27-midgame-meta-reset

Sparring partner for `./run arena`. **Not our code** — see
[../NOTICE.md](../NOTICE.md) for where it came from and under what licence.

## Extraction

| | |
|---|---|
| Source notebook | `25-27-strict-future-v27-midgame-meta-reset.ipynb` |
| Packing style | blob — base85/base64 payload, decoded and decompressed |
| Extracted | 2026-08-27 |
| Size | 416 lines, 20,813 bytes |
| SHA-256 | `f48c21166eac68d1b05a401f04f94a2eb6154e65415af64893672365ff33c7b8` |

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
./run arena --opponent opp:25-27-strict-future-v27-midgame-meta-reset --seeds 12
./run profile                                   # what it actually does
```
