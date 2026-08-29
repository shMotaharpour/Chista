# economics-driven-rule-agent-ecobot-v6

Sparring partner for `./run arena`. **Not our code** — see
[../NOTICE.md](../NOTICE.md) for where it came from and under what licence.

## Extraction

| | |
|---|---|
| Source notebook | `economics-driven-rule-agent-ecobot-v6.ipynb` |
| Packing style | writefile — `%%writefile` cell body |
| Extracted | 2026-08-27 |
| Size | 1,796 lines, 69,074 bytes |
| SHA-256 | `aca6dcddafd6fd6bc55bf271112fd0041d92b6eb45f6e1babc36b40c81f32cad` |

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
./run arena --opponent opp:economics-driven-rule-agent-ecobot-v6 --seeds 12
./run profile                                   # what it actually does
```
