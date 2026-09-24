# R005 — Every number has a source

**Summary (<=50 words):** Every threshold, budget and constant traces to a
source: an F-finding, the engine, or a recorded measurement. So does every
conclusion drawn from one, because a chain of inference over a measured number
is not itself measured. Missing values are measured or left as named TODOs.

## Decision

A number that enters the code, a document or an issue must be able to answer
"where did this come from?" with one of exactly three answers:

1. **A finding** — an `F<NNN>` file that established it from the engine. Cite
   the id next to the number: `0.965` is F046's, not a rounding choice.
2. **The engine** — read at build time through `kaggle_environments`, never
   transcribed by hand (R002), and re-checked live where the tables do not
   state it (`agent/tile_dp/graph.py::verify_engine_constants` is the pattern).
3. **A measurement you took** — with the procedure and the result written down
   where the number is used, so a later reader can repeat it and disagree.

Anything else is invented, and this rule says invented numbers are defects.

## Why it is a rule and not a preference

Issue #9 shipped with "refuse to start a replan when `remainingOverageTime`
< 45". The 45 was three quarters of the 60-second bank and nothing more. It was
not derived, not measured and not marked as a guess — it read exactly like the
0.965 two paragraphs above it, which *is* F046's measured figure. The owner
caught it; nothing in the repo would have.

That is the shape of the failure. An invented number does not look invented once
it is written down next to real ones. It gets quoted, inherited, tuned around,
and eventually defended — and the cost lands far from where it was introduced.

The same repo already does this right often enough to make the contrast plain:
the runtime budget is F046's, the price pivot is F034's, the day length is
re-derived from `max_lifespan_step` on a live sim rather than trusted, and
`contract_id()` fingerprints the registry and the engine instead of carrying a
hand-bumped version. R005 makes that the standard rather than the habit.

## A measurement must actually measure

A check that compares two hand-written numbers is not a measurement, however it
is framed. `verify_op_steps` (graph.py) counts the `sim.step` calls the probe
itself writes as literals and compares them with the `OP_STEPS` table: both
values are authored by hand, in the same commit, so engine drift — the thing the
probe is named for — cannot fail it.

The test is simple: **could this check fail if the world changed and nobody
edited our code?** If not, it is a consistency check between two copies, and the
number still has no source. `verify_engine_constants` passes that test —
it plants a crop and reads `max_lifespan_step` back out of the engine.

## When you do not have a source

Two options, both acceptable, in this order:

- **Measure it.** Most of these are twenty lines and one episode.
- **Leave a named TODO** with the missing measurement stated: what to measure,
  and what decision depends on it. A visible gap is worth more than a plausible
  number, because a reader can act on the first and cannot see the second.

What is not acceptable is a plausible placeholder that reads as a decision.

## Self-calibrating beats guessed

Where a threshold depends on the machine or the episode, prefer a value the code
derives at runtime from what it has observed over a constant chosen in advance.
The bank policy in #9 is the worked example: instead of a fixed cutoff, the agent
reserves against the worst overrun *it has already produced this episode*, and
the two shape parameters come from the measured timing distribution rather than
from the issue that specified them.

## A conclusion drawn from a measurement is not itself measured

The rule above is about numbers. This one is about what you say next, and it is
the harder half: **an inference inherits the authority of the number it came
from without inheriting its evidence.** It does not read like a guess. It reads
like a consequence.

F048 is the worked example, and it went wrong twice in a day, in opposite
directions.

A grader log held 719 records where the rules say 720 turns. First conclusion:
the exporter dropped one. Measured — it had not; the agent really is called 719
times. Good so far: a guess, checked, corrected.

Then the second conclusion, built on the correction: *if the agent is never
called at step 719, then step 718 must be too late, so a final sale is lost.*
That went into a PR review as a design consequence, carrying endgame advice to
schedule liquidation an hour early.

It was false. The step-718 action is processed in full — a wheat seed bought
there costs ten coins in the terminal state. Nothing is lost. The error was
worse than the first precisely because it sounded derived: it had a real
measurement upstream of it, and every link in between was one twenty-line
script away from being checked.

So the test to apply before stating a consequence:

> **What would I observe if this were false — and have I observed it?**

If the answer is "I would see the money not move, and I have not looked", the
sentence is an inference, not a finding. Two honest ways out, the same as for a
missing number: run the check, or label it. An `F<NNN>` file states which of its
claims are measured and which are read off them; a conclusion with no evidence
of its own does not get to sit in the same list as one that has it.

Why this earns a section rather than a line in the one above: nobody defends an
unexplained `45`. People defend a conclusion, because defending it feels like
defending the measurement it grew from.

## How it is enforced

In review, by asking the question. A reviewer seeing a literal — in code, a
document, an issue or a commit message — asks for its source, and "it seemed
about right" is a change request.

Mechanically, where it is cheap: constants derived from the engine get a live
check like `verify_engine_constants`, and artifacts carry fingerprints of what
they were built from (`contract_id()`), so a number that silently stopped being
true fails loudly instead.
