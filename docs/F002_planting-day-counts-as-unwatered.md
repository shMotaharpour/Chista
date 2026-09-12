# F002 — Planting day counts as unwatered

**Summary (<=50 words):** A new plant is born with consecutive_unwatered = 1 — the planting day counts as a dry day. Two consecutive dry nights turn the tile into a WEED, not empty ground, so a plant not watered on its own planting day is gone by the next night.

## Finding

- A new plant starts with `consecutive_unwatered = 1` — the planting day counts as a dry day.
- Two consecutive dry nights turn the tile into a WEED, not into empty ground: a plant must be watered on its planting day or it is gone by the next night.

*Source: "Crops and animals — every rule, numbered" research document, project [AgriOracle](https://github.com/shMotaharpour/AgriOracle).*
