# F022 — Production past the cap is lost

**Summary (<=50 words):** Production past max_held is lost — held grows as min(max_held, held + base + bonus). Product already held never spoils.

## Finding

- Production past `max_held` is lost: `min(max_held, held + base + bonus)`. Product already held never spoils.

*Source: "Crops and animals — every rule, numbered" research document, project [AgriOracle](https://github.com/shMotaharpour/AgriOracle).*
