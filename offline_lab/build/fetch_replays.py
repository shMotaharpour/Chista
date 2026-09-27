"""Download a batch of episode replays from Kaggle, one at a time.

Each download gets its own process (the API writes <path>/<file> and the
32 MB parses fine there); a hang or error costs one episode, not the batch.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

KAGGLE_PY = "/home/amirelite_ai/.local/share/uv/tools/kaggle/bin/python"
OUT = Path.home() / "Chista/ClaudResearch/offline_lab/build/kaggle_samples"


def fetch(eid: int) -> str:
    r = subprocess.run(
        [KAGGLE_PY, "-c",
         "import sys; from kaggle.api.kaggle_api_extended import KaggleApi;"
         f"api=KaggleApi(); api.authenticate();"
         f"print(api.competition_episode_replay({eid}, path='{OUT}'))"],
        capture_output=True, text=True, timeout=240)
    return (r.stdout + r.stderr).strip()[-200:]


def main() -> None:
    eids = [int(a) for a in sys.argv[1:]]
    OUT.mkdir(parents=True, exist_ok=True)
    for eid in eids:
        target = OUT / f"episode-{eid}-replay.json"
        if target.exists() and target.stat().st_size > 1000:
            print(f"{eid}: cached")
            continue
        try:
            msg = fetch(eid)
            ok = target.exists() and target.stat().st_size > 1000
            print(f"{eid}: {'OK ' + str(target.stat().st_size) if ok else 'MISS'} {msg}")
        except subprocess.TimeoutExpired:
            print(f"{eid}: TIMEOUT")


if __name__ == "__main__":
    main()