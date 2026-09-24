# Opponent agents

Third-party agents — competitor code published on Kaggle — kept as **sparring
partners** for evaluation. They never ship; they exist to give our strategies
something real to lose to.

Where they came from and under what licence is stated once, for all of them, in
[`NOTICE.md`](NOTICE.md). Read that before committing anything here.

## Layout

```
opponents/
├── NOTICE.md                 where they came from, and under what licence
├── LICENSE-Apache-2.0.txt    a verbatim copy of that licence
├── extract.py                pulls an agent out of a notebook, without running it
└── <slug>/
    ├── agent.py              their code, byte-for-byte as published
    └── SOURCE.md             which notebook, how it was packed, its SHA-256,
                              what the audit found, how to play it
```

The split is deliberate. A per-agent `SOURCE.md` says only what differs between
agents; the licence and the source page are the same for all of them, so they
are stated once in `NOTICE.md` rather than nineteen times where they could
drift. The count itself lives in exactly one place — `AGENT_COUNT` in
`__init__.py` — and `tests/test_opponents.py` checks it against the directories
actually present.

## Rules

1. **Do not edit their logic.** `agent.py` is the published payload
   byte-for-byte and its SHA-256 is recorded in its `SOURCE.md`;
   `tests/test_opponents.py` checks them. No linter or formatter runs over this
   directory. If one needs a shim to run, put the shim in a separate file and
   say so.
2. **Never import `opponents/` from the submission.** These are inputs to
   evaluation, not to the agent — on Kaggle they would not be there to import.
   `tests/test_layering.py` enforces it.
3. **Keep them small.** Decoded route tables belong in `<slug>/unpacked/`, which
   is gitignored — one agent unpacks to 11 MB.

## Reading one

`extract.py` parses notebooks with `ast` and **never executes them** — they are
arbitrary code published by strangers, and running one to find out what it does
is the wrong order.

```python
import sys; sys.path.insert(0, "opponents")
import extract
from pathlib import Path

source = Path("opponents/<slug>/agent.py").read_text()
extract.audit(source)             # what to read before playing it
extract.payload_digests(source)   # the packed modules the audit looked inside
extract.unpack(Path("opponents/<slug>/agent.py"), Path("opponents/<slug>/unpacked"))
```

`audit` reports anything reaching the network, the filesystem or a subprocess.
It **recurses into packed payloads**: six of these agents carry their real logic
base85- or zlib-packed and `exec` it at import, so auditing only the outer file
would describe a loader rather than an agent. `payload_digests` reports that
coverage separately, because "audited and clean" and "never decoded" both look
like an empty finding list otherwise.

Most of what `unpack` produces is data rather than code — precomputed route
tables and action schedules — and that is usually the more interesting half.

## Playing one

Each `agent.py` exposes a top-level `def agent(obs)`; that is the whole
contract, and it is what `extract.py` looks for. They run on either engine path
in `world/`. A slug is not a valid dotted module name, so load one by path:

```python
import importlib.util
spec = importlib.util.spec_from_file_location(
    "opp_agent", "opponents/<slug>/agent.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)   # import-time: a packed agent unpacks itself here
agent = mod.agent              # obs -> action
```

Built-in baselines need no directory: `"random"`, `"pass"` and `"starter"` come
from the environment itself.

> **Before the arena runs them:** hand a third-party agent a *copy* of the
> observation, never the live view. `agent/world/README.md` and R004 record that fast
> mode hands out live views and that a live view can zero the opponent's money
> or grant itself free seeds. The audit found no agent doing anything of the
> sort — the point is that a stray mutation would corrupt an evaluation
> silently, and silence is the failure mode this repository is organised
> against (F047).
