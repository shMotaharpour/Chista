"""The agent's artifact store: one data file per artifact, and its info beside it.

An artifact is built by a builder in `offline_lab/` and read by the agent, so it lives
here, inside `agent/`. Every artifact has exactly two files, named after it:

    <name><suffix>     the data (whatever the reader needs: `.npz`, `.json`, …)
    <name>.json        the info, the same keys for every artifact

The info is not a build log: it is what a reader needs to decide whether it can use the
data at all, plus where it came from. `write_info` refuses a key outside `INFO_KEYS`, so
"the same keys for every artifact" is checked rather than promised.

    name      the artifact's name, equal to the file stem
    kind      what the data is (`tile_graph`, …) - the reader's own word for it
    file      the data file's name, relative to this folder
    contract  what the data IS, computed from its inputs by the builder
    engine    the engine fingerprint it was built against
    registry  the registry fingerprint it was built from, when it has one
    stats     what the reader can sanity-check: counts, sizes, horizons
    source    the builder that wrote it, as `module:function`
    created_utc  when it was written
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

#: This folder: the one place an artifact and its info live.
ARTIFACT_DIR = Path(__file__).resolve().parent

#: The keys every info file has, in the order they are written.
INFO_KEYS: tuple[str, ...] = ("name", "kind", "file", "contract", "engine", "registry",
                              "stats", "source", "created_utc")


def artifact_path(name: str, suffix: str) -> Path:
    """The data file of an artifact."""
    return ARTIFACT_DIR / f"{name}{suffix}"


def info_path(name: str) -> Path:
    """The info file of an artifact."""
    return ARTIFACT_DIR / f"{name}.json"


def write_info(name: str, *, kind: str, file: str, contract: str, engine: str,
               registry: str | None, stats: dict[str, Any], source: str) -> Path:
    """Write the info file beside an artifact. An unknown key raises."""
    info = {"name": name, "kind": kind, "file": file, "contract": contract,
            "engine": engine, "registry": registry, "stats": stats, "source": source,
            "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
    if tuple(info) != INFO_KEYS:
        raise AssertionError(f"the info keys changed: {tuple(info)} != {INFO_KEYS}")
    path = info_path(name)
    path.write_text(json.dumps(info, indent=2, sort_keys=False) + "\n")
    return path


def read_info(name: str) -> dict[str, Any]:
    """Read an artifact's info, refusing a file that is not in the standard shape."""
    info = json.loads(info_path(name).read_text())
    if tuple(info) != INFO_KEYS:
        raise ValueError(f"{info_path(name).name} is not a standard artifact info: "
                         f"keys {tuple(info)} != {INFO_KEYS}")
    return info
