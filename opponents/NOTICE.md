# Third-party agents — provenance and licence

Every `agent.py` under `opponents/<slug>/` is **someone else's work**.

They are extracted from public notebooks on the competition's own code page:

<https://www.kaggle.com/competitions/kaggriculture/code>

and are published there under the **Apache License 2.0**, a verbatim copy of
which sits beside this file in
[`LICENSE-Apache-2.0.txt`](LICENSE-Apache-2.0.txt).

This is stated here once, for all of them, rather than repeated in each agent's
`SOURCE.md` — where it would be nineteen chances to drift out of agreement. Each
agent's own `SOURCE.md` records what is specific to it: which notebook, how it
was packed, its SHA-256, what the safety audit found, and how to play it.

None of this ships. The submission is its own package; this directory exists
only so the evaluation arena has opponents worth losing to.
`tests/test_layering.py` is what keeps the two apart.

## What Apache 2.0 section 4 asks of us

| | how this directory satisfies it |
|---|---|
| (a) give recipients a copy of the License | `LICENSE-Apache-2.0.txt`, beside the code |
| (b) state significant changes | **There are none.** Every `agent.py` is the published payload byte-for-byte. Each `SOURCE.md` records its SHA-256 and `tests/test_opponents.py` checks every one of them against the bytes on disk. No linter or formatter runs over this directory |
| (c) retain notices in the source | Follows from (b) — nothing was stripped, because nothing was edited |
| (d) reproduce any NOTICE file | None of the source notebooks carried one |

That (b) claim is load-bearing, and it has been wrong once: a `.strip()` in the
extractor silently cost one trailing byte. It was caught only because one
notebook publishes its own artifact hash and ours disagreed with it. Which is
the point of `tests/test_opponents.py`: the claim is now checked on every run
rather than believed.

## Attribution a source notebook explicitly asked for

`25-27-strict-future-v27-midgame-meta-reset` credits its 719-action route to
team **Ezzzzzekki** (episode 91493566, seat 0, action SHA-256
`9080682756f5b9fc…`), claiming for itself only the meta audit, the
fixed-versus-router ablation, the chronological freeze protocol, actor-local
WEED repair and the SELL-slot layer. It asks forks to retain both credits, so
both are recorded here.

## If you are one of these authors

Open an issue and it will be removed. These agents are here to be measured
against, not republished as a library, and none is submitted to the competition.
