# v16-rc5-high-score-8c-4s-premium-market-lead

Sparring partner for the evaluation arena. **Not our code** — see
[../NOTICE.md](../NOTICE.md) for where it came from and under what licence.

## Extraction

| | |
|---|---|
| Source notebook | `v16-rc5-high-score-8c-4s-premium-market-lead.ipynb` |
| Packing style | writefile — `%%writefile` cell body |
| Extracted | 2026-08-27 |
| Size | 247 lines, 18,946 bytes |
| SHA-256 | `f029fa0cb66a9eb509afbe44e3f59b800332d0419db91607183410e4089c4d19` |

`agent.py` is the payload byte-for-byte; `tests/test_opponents.py` checks it
against that hash, and re-running the extractor must reproduce it. Extraction is
`opponents/extract.py`, which parses the notebook with `ast` and **never
executes it**.

## Audit

The arena runs this inside our own interpreter, so its imports and calls were
checked before it was ever called:

Clean — every import and call is within the allowlist.

This agent packs its payload: **2 decoded** — 0 modules audited above, 2 data tables (route or schedule, no code to read).

## Play it

```python
import importlib.util
spec = importlib.util.spec_from_file_location(
    "opp_agent", "opponents/v16-rc5-high-score-8c-4s-premium-market-lead/agent.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)   # import-time: this agent unpacks itself here
agent = mod.agent              # obs -> action
```

Hand it a **copy** of the observation, never the live view (R004).
