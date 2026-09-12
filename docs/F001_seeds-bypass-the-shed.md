# F001 — Seeds bypass the shed

**Summary (<=50 words):** Seeds live in private["seeds"], never pass through the shed, and are exempt from its 100-unit capacity. PLANT needs an empty owned tile. The seed check is atomic: over-requesting one crop in a turn drops every PLANT of that crop, silently, not just the excess.

## Finding

- PLANT consumes one seed from `private["seeds"]`; seeds never pass through the shed, so `BUY_SEED` is exempt from its capacity.
- PLANT needs the tile empty and owned.
- Atomic seed check: if the turn's PLANT requests for one crop exceed the seeds held, **every** PLANT of that crop that turn is dropped, not just the excess — silently.

*Source: "Crops and animals — every rule, numbered" research document.*
