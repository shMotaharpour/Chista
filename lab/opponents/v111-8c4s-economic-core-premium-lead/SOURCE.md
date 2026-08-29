# v111-8c4s-economic-core-premium-lead

Sparring partner for `./run arena`. **Not our code** — see
[../NOTICE.md](../NOTICE.md) for where it came from and under what licence.

## Extraction

| | |
|---|---|
| Source notebook | `v111-8c4s-economic-core-premium-lead.ipynb` |
| Packing style | writefile — `%%writefile` cell body |
| Extracted | 2026-08-27 |
| Size | 247 lines, 18,946 bytes |
| SHA-256 | `f029fa0cb66a9eb509afbe44e3f59b800332d0419db91607183410e4089c4d19` |

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
./run arena --opponent opp:v111-8c4s-economic-core-premium-lead --seeds 12
./run profile                                   # what it actually does
```
