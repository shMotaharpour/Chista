# adaptive-replay-agent

Sparring partner for the evaluation arena. **Not our code** — see
[../NOTICE.md](../NOTICE.md) for where it came from and under what licence.

## Extraction

| | |
|---|---|
| Source notebook | `kaggriculture-adaptive-replay-agent.ipynb` |
| Packing style | blob — base85/base64 payload, decoded and decompressed |
| Extracted | 2026-08-27 |
| Size | 551 lines, 28,709 bytes |
| SHA-256 | `3c86ae6e2afaa0910fd62431b62460b6c13a67cc5d7249eff3c4269be68409f8` |

`agent.py` is the payload byte-for-byte; `tests/test_opponents.py` checks it
against that hash, and re-running the extractor must reproduce it. Extraction is
`opponents/extract.py`, which parses the notebook with `ast` and **never
executes it**.

## Audit

The arena runs this inside our own interpreter, so its imports and calls were
checked before it was ever called:

Clean — every import and call is within the allowlist.

This agent packs its payload: **4 decoded** — 0 modules audited above, 4 data tables (route or schedule, no code to read).

## Play it

```python
import importlib.util
spec = importlib.util.spec_from_file_location(
    "opp_agent", "opponents/adaptive-replay-agent/agent.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)   # import-time: this agent unpacks itself here
agent = mod.agent              # obs -> action
```

Hand it a **copy** of the observation, never the live view (R004).
