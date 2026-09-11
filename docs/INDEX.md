# INDEX

One line per file: a link to the file plus its ≤50-word summary.
Order: **Rules** (`R<NNN>_<slug>.md`) first, then **Findings**
(`F<NNN>_<slug>.md`). IDs are permanent — never renumber, never reuse.

## Rules

- [R001_competition_stack_and_commit_rules.md](R001_competition_stack_and_commit_rules.md) —
  Project stack and workflow rules: Kaggle Kaggriculture environment, CPU-only
  PyTorch, OR-Tools and scipy for optimization; every agent commit must use
  `~/.local/bin/agent-commit` so the Co-authored-by trailer separates agent
  work from the user's own commits.
- [R002_import_rules_from_kaggle_environments.md](R002_import_rules_from_kaggle_environments.md) —
  Import game constants and formulas directly from `kaggle_environments`
  instead of transcribing them: one source of truth, zero parity-harness
  maintenance, free at runtime; pinned environment version mitigates private
  API rename risk.

## Findings

(no findings yet — add one line per `F<NNN>_<slug>.md` as files are created)
