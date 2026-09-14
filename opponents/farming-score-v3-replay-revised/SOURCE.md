# farming-score-v3-replay-revised

Sparring partner for `./run arena`. **Not our code** — see
[../NOTICE.md](../NOTICE.md) for where it came from and under what licence.

## Extraction

| | |
|---|---|
| Source notebook | `farming-score-v3-replay-revised.ipynb` |
| Packing style | blob — base85/base64 payload, decoded and decompressed |
| Extracted | 2026-08-27 |
| Size | 1,201 lines, 97,665 bytes |
| SHA-256 | `12eb55e1e2455a2bce61c86e7dd11467fd731b31b4154680bf33f83a3eadd729` |

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
./run arena --opponent opp:farming-score-v3-replay-revised --seeds 12
./run profile                                   # what it actually does
```
