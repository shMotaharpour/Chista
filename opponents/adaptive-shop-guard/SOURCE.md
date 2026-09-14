# adaptive-shop-guard

Sparring partner for `./run arena`. **Not our code** — see
[../NOTICE.md](../NOTICE.md) for where it came from and under what licence.

## Extraction

| | |
|---|---|
| Source notebook | `kaggriculture-adaptive-shop-guard.ipynb` |
| Packing style | blob — base85/base64 payload, decoded and decompressed |
| Extracted | 2026-08-27 |
| Size | 1,091 lines, 93,579 bytes |
| SHA-256 | `2fe711896465626350efafc87960a6d2f05b510ec7af62bc4679a80cdf3f9fc7` |

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
./run arena --opponent opp:adaptive-shop-guard --seeds 12
./run profile                                   # what it actually does
```
