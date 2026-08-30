# ChistaAgent — Agent implementation on the Kaggle env (superseded draft)

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Status:** SUPERSEDED — this was the initial scaffold plan written before the project was identified as the Kaggriculture competition. Kept for history. The live roadmap is `docs/research/004-roadmap.md`.

**Goal (original):** Build the ChistaAgent project skeleton: an AI agent running in a Docker container based on the Kaggle Python image, with an independent "lab" for design/optimization and a docs folder for rules and research findings.

**Architecture (original):** Two separated environments —
1. **Runtime (agent):** Docker container on the Kaggle Python image (only libraries in that image + agent code). Agent logic structured for context engineering (system prompt, context cache, tools, agent loop).
2. **Lab (hosting/optimization):** on the host machine, free dependencies, responsible for evaluation, prompt/context optimization, and docs generation.

> Later decisions: no Docker (user) — plain venv instead; the "agent" is a Kaggle competition submission, not a tool-using LLM agent.

**Tech Stack (original):** Python (Kaggle image), Docker, pytest (lab), OpenRouter/LLM API, markdown docs.

---

## Original structure

```
Chista/ChistaAgent/
├── agent/                  # code running inside the Kaggle container
├── docker/                 # FROM kaggle python image (dropped)
├── lab/                    # independent, free deps
├── docs/
│   ├── rules/              # project rules
│   └── research/           # research findings
```

## Original tasks (historical)

1. Folder skeleton and README
2. Dockerfile based on the Kaggle image (dropped)
3. Context-engineering core (not applicable)
4. Agent loop + tools (not applicable)
5. Entrypoint + API (not applicable)
6. Lab evaluator/runner ✅ (done as v0 harness)
7. Initial docs ✅

## Verification
- venv smoke test passes: env.run completes with rewards
- Atomic commit per task

## Lessons kept
- Bite-sized tasks, exact paths, verification steps, atomic commits — still the working style
