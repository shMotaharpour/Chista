# findings-from-zero-to-top-meta

Sparring partner for the evaluation arena. **Not our code** — see
[../NOTICE.md](../NOTICE.md) for where it came from and under what licence.

## Extraction

| | |
|---|---|
| Source notebook | `kaggriculture-findings-from-zero-to-top-meta.ipynb` |
| Packing style | blob — base85/base64 payload, decoded and decompressed |
| Extracted | 2026-08-27 |
| Size | 5,455 lines, 446,094 bytes |
| SHA-256 | `3ebb067123cb0c780abd3038a13ecf0599628946a902f00b79ffc58f9dd51067` |

`agent.py` is the payload byte-for-byte; `tests/test_opponents.py` checks it
against that hash, and re-running the extractor must reproduce it. Extraction is
`opponents/extract.py`, which parses the notebook with `ast` and **never
executes it**.

## Audit

The arena runs this inside our own interpreter, so its imports and calls were
checked before it was ever called:

Clean — every import and call is within the allowlist.

This agent packs its payload: **6 decoded** — 0 modules audited above, 6 data tables (route or schedule, no code to read).

## Play it

```python
import importlib.util
spec = importlib.util.spec_from_file_location(
    "opp_agent", "opponents/findings-from-zero-to-top-meta/agent.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)   # import-time: this agent unpacks itself here
agent = mod.agent              # obs -> action
```

Hand it a **copy** of the observation, never the live view (R004).
