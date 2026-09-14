# v21-r1-public-state-route-portfolio

Sparring partner for the evaluation arena. **Not our code** — see
[../NOTICE.md](../NOTICE.md) for where it came from and under what licence.

## Extraction

| | |
|---|---|
| Source notebook | `v21-r1-public-state-route-portfolio.ipynb` |
| Packing style | blob — base85/base64 payload, decoded and decompressed |
| Extracted | 2026-08-27 |
| Size | 239 lines, 158,309 bytes |
| SHA-256 | `c6f96a8521dc9aa369b6f27e5b36b9d481e5c1688f50c8ca215c3bb53f1f9eb8` |

`agent.py` is the payload byte-for-byte; `tests/test_opponents.py` checks it
against that hash, and re-running the extractor must reproduce it. Extraction is
`opponents/extract.py`, which parses the notebook with `ast` and **never
executes it**.

## Audit

The arena runs this inside our own interpreter, so its imports and calls were
checked before it was ever called:

```
line 20: compile() -- self-bundled module loader, payload audited separately
```

This agent packs its payload: **23 decoded** — 3 modules audited above, 20 data tables (route or schedule, no code to read).

## Play it

```python
import importlib.util
spec = importlib.util.spec_from_file_location(
    "opp_agent", "opponents/v21-r1-public-state-route-portfolio/agent.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)   # import-time: this agent unpacks itself here
agent = mod.agent              # obs -> action
```

Hand it a **copy** of the observation, never the live view (R004).
