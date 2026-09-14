# adaptive-public-state-multi-route

Sparring partner for `./run arena`. **Not our code** — see
[../NOTICE.md](../NOTICE.md) for where it came from and under what licence.

## Extraction

| | |
|---|---|
| Source notebook | `kaggriculture-adaptive-public-state-multi-route.ipynb` |
| Packing style | writefile — `%%writefile` cell body |
| Extracted | 2026-08-27 |
| Size | 387 lines, 174,502 bytes |
| SHA-256 | `9b5e1157af3d4a643fcd818f5c290003d2672230d5ae8c0024fae0836cc894df` |

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
./run arena --opponent opp:adaptive-public-state-multi-route --seeds 12
./run profile                                   # what it actually does
```
