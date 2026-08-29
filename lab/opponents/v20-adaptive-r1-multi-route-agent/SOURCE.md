# v20-adaptive-r1-multi-route-agent

Sparring partner for `./run arena`. **Not our code** — see
[../NOTICE.md](../NOTICE.md) for where it came from and under what licence.

## Extraction

| | |
|---|---|
| Source notebook | `v20-adaptive-r1-multi-route-agent[silver-rank-agent].ipynb` |
| Packing style | blob — base85/base64 payload, decoded and decompressed |
| Extracted | 2026-08-27 |
| Size | 1,013 lines, 148,200 bytes |
| SHA-256 | `8ac34abce129cf5c9456776c90edf7d2233b3a280bbdcf7622628825ef3669a0` |

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
./run arena --opponent opp:v20-adaptive-r1-multi-route-agent --seeds 12
./run profile                                   # what it actually does
```
