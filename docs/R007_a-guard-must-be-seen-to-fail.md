# R007 — A guard must be seen to fail

**Summary (<=50 words):** A test earns the name "regression guard" only after
someone has re-introduced the bug and watched it go red. Five guards in this
repository passed while guarding nothing, because each exercised a path the
production code never takes. Reading cannot find this; two minutes of breaking
it can.

## Decision

Before a test may be described as covering a bug — in its name, its docstring,
a PR body or a review — the bug must be put back and the test must be seen to
fail. Then the bug comes out again and the test goes green. The PR says which
bug was re-introduced and what the failure looked like.

This is not a suggestion about diligence. It is the only known way to catch the
defect described below, which has occurred five times in this repository and
was invisible by reading every time.

## The failure this rule exists for

**A guard that exercises a path the production code does not take.** It passes.
It looks thorough. It reports success forever, and the thing it names is
completely unprotected.

The five, each caught only by breaking the code on purpose:

| # | the guard | what it actually tested |
|---|---|---|
| 1 | `test_no_module_outside_the_arena_imports_opponents` | membership of a bare root against full dotted names, so `import opponents.extract` walked straight through |
| 2 | `test_only_the_runner_sets_thread_vars` | `ast.Assign` to a Name — but environment variables are set with `os.environ[...]`, so the scan matched nothing, including the file it guarded |
| 3 | `test_guard_hands_out_copies_not_live_views` | `guarded_call(copy=True)` while the runner passed `copy=False` |
| 4 | `test_each_seat_sees_its_own_player_index` | two hand-made dicts, never the runner's `views[0]` / `views[1]` assignment where the bug lived |
| 5 | `test_fallback_fires_without_scipy` | a patched `_solve_lp` raising `RuntimeError` — scipy was installed the whole time, and the module-level `from scipy.optimize import linprog` was never exercised |

Every one passed. Every one was written in good faith by someone who believed
it covered the case. Four of the five were caught by re-introducing the bug;
the fifth by blocking the import the test claimed to simulate.

## What the rule requires

1. **Break it.** Put the bug back — revert the fix, inject the violation, block
   the import, whatever the guard claims to catch.
2. **Watch it fail.** The test must go red, and the failure message must name
   the right thing. A red test for the wrong reason is not evidence.
3. **Put it back.** Restore the code; confirm green.
4. **Say so.** The PR records which bug was re-introduced and what the failure
   output was. A reviewer should not have to redo step 1 to trust the guard.

## Why reading does not substitute

All five guards above survived review by reading — including review by the
author, who had the bug fresh in mind. The defect is a mismatch between the
path the test drives and the path production drives, and that mismatch lives in
the *call site*, not in the assertion. The assertion is usually correct. It is
simply asked about the wrong thing.

Breaking the code tests the call site, which is the part nobody looks at.

## Where it applies

Any test whose name, docstring or PR description claims to cover a specific
defect, regression or contract. It does not apply to ordinary unit tests of new
behaviour — only to the ones that claim to *guard*.

A guard that cannot be made to fail is not protecting anything. If breaking the
code does not turn it red, the guard is wrong, not the method.

## How it is enforced

Review asks one question: **which bug did you re-introduce, and what did the
failure say?** No answer means the guard is unverified, and the PR says so
rather than claiming coverage it has not demonstrated.
