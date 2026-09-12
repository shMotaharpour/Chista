# F044 — Animal age collapses to a residue

**Summary (<=50 words):** An animal's age matters only through a residue (its
position in the production interval), so per-tile animal planning collapses
to a flat DP over the residue instead of the full age.

## Finding

- Age matters only through a residue, so it can be collapsed — the per-tile
  DP over animals is flat.

*Source: "Kaggriculture — the rules, as we have established them" research
document, section 9 (Animals).*
