# R001 — Commit trailer and repository workflow rules

**Summary (≤50 words):** Every agent commit must run
`~/.local/bin/agent-commit -m "..."`, never plain `git commit`; the wrapper
appends `Co-authored-by: Hermes (GLM via OpenRouter) <hermes@local>` so agent
work stays attributable. Never commit directly to `main` — feature branch, PR,
squash-merge. All repo files are English only; Persian is for chat replies.

## Details

- The user's own `git commit` usage is unchanged.
- The trailer format is fixed, including the `<hermes@local>` address — do not
  "fix" it to a real email.
- Filter agent work with `git log --grep="Co-authored-by: Hermes"`.
- Findings files follow `F<NNN>_<slug>.md`, rules files follow
  `R<NNN>_<slug>.md` (slug ≤ 10 words); each file opens with a ≤ 50-word
  summary, and `INDEX.md` lists rules first, then findings, one line per file
  (link + summary only). IDs are permanent.
