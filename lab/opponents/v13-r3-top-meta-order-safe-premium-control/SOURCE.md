# v13-r3-top-meta-order-safe-premium-control

Sparring partner for `./run arena`. **Not our code** — see
[../NOTICE.md](../NOTICE.md) for where it came from and under what licence.

## Extraction

| | |
|---|---|
| Source notebook | `v13-r3-top-meta-order-safe-premium-control.ipynb` |
| Packing style | blob — base85/base64 payload, decoded and decompressed |
| Extracted | 2026-08-27 |
| Size | 541 lines, 28,715 bytes |
| SHA-256 | `6f52902081fed08bb5da08d575b796437e645c5320662b448b81eb079f185cfb` |

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
./run arena --opponent opp:v13-r3-top-meta-order-safe-premium-control --seeds 12
./run profile                                   # what it actually does
```
