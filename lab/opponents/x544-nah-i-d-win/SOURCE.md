# x544-nah-i-d-win

Sparring partner for `./run arena`. **Not our code** — see
[../NOTICE.md](../NOTICE.md) for where it came from and under what licence.

## Extraction

| | |
|---|---|
| Source notebook | `kaggriculture-x544-nah-i-d-win.ipynb` |
| Packing style | blob — base85/base64 payload, decoded and decompressed |
| Extracted | 2026-08-27 |
| Size | 102 lines, 144,768 bytes |
| SHA-256 | `e38006a10ce29518c44e79f46422959c818531185b66f5f51a2b6ff3847684a7` |

`agent.py` is the payload byte-for-byte; `tests/test_opponents.py` checks it
against that hash, and re-running the extractor must reproduce it. Extraction is
`lab/opponents/extract.py`, which parses the notebook with `ast` and **never
executes it**.

## Audit

The arena runs this inside our own interpreter, so its imports and calls were
checked before it was ever called:

- `line 14: calls compile()`
- `line 24: calls compile()`

## Play it

```bash
./run arena --opponent opp:x544-nah-i-d-win --seeds 12
./run profile                                   # what it actually does
```
