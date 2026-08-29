# 44-46-strict-future-top-30-v22-price-impact

Sparring partner for `./run arena`. **Not our code** — see
[../NOTICE.md](../NOTICE.md) for where it came from and under what licence.

## Extraction

| | |
|---|---|
| Source notebook | `44-46-strict-future-top-30-v22-price-impact.ipynb` |
| Packing style | blob — base85/base64 payload, decoded and decompressed |
| Extracted | 2026-08-27 |
| Size | 346 lines, 18,609 bytes |
| SHA-256 | `62fb5a5f66f0011092a2b51e3192879ba583d5d815d761f176091c615657147a` |

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
./run arena --opponent opp:44-46-strict-future-top-30-v22-price-impact --seeds 12
./run profile                                   # what it actually does
```
