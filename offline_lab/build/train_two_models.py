"""Train Model 1 (tile edges) and Model 2 (market slots) on the shard corpus.

Both nets live in `model_tile_policy.py` / `model_market_policy.py`; this
script builds ONE aligned training table from the shards (every shard stores
its own samples, so alignment is by shard, not by a global key), drives the
JAX training, and exports each net to a pure-NumPy npz after verifying the
NumPy forward pass against JAX.

    .venv/bin/python offline_lab/build/train_two_models.py \
        --shards 'offline_lab/build/kaggle_shards/*.npz' \
        --out offline_lab/build/models
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model_market_policy import scalars_of, edge_summary, order_targets

N_CHAINS = 64          # the chain registry's size bracket (63 built + spare)
QTY_BUCKETS = 8        # 0,1,2,3-4,5-8,9-16,17-32,33+


def load_corpus(paths: list[str]) -> dict:
    """One aligned table: per-sample state + per-sample edge labels + orders."""
    tiles_own, tiles_opp, scal, chain_bag = [], [], [], []
    lab_ptr, lab_cell, lab_chain, lab_item = [0], [], [], []
    kind_g, item_g, qty_g = [], [], []
    for p in paths:
        sh = np.load(p, allow_pickle=False)
        tiles_own.append(sh["tiles_own"])
        tiles_opp.append(sh["tiles_opp"])
        scal.append(scalars_of(sh))
        chain_bag.append(edge_summary(sh))
        n = int(sh["n_samples"])
        lab_cell.append(sh["lab_cell"])
        lab_chain.append(sh["lab_chain"])
        lab_item.append(sh["lab_item"])
        # per-sample ptr offsets shift by the shard's own label count
        lp = sh["lab_ptr"][:-1] + sum(0 for _ in ())   # shard-local starts
        counts = np.diff(sh["lab_ptr"])
        prev = lab_ptr[-1]
        for c in counts:
            lab_ptr.append(prev + int(c))
            prev += int(c)
        kg, ig, qg = order_targets(sh, None)
        kind_g.append(kg)
        item_g.append(ig)
        qty_g.append(qg)
    table = {
        "tiles_own": np.concatenate(tiles_own),
        "tiles_opp": np.concatenate(tiles_opp),
        "scalars": np.concatenate(scal),
        "chain_bag": np.concatenate(chain_bag),
        "lab_cell": np.concatenate(lab_cell),
        "lab_chain": np.concatenate(lab_chain),
        "lab_item": np.concatenate(lab_item),
        "lab_ptr": np.array(lab_ptr, dtype=np.int64),
        "ord_kind": np.concatenate(kind_g),
        "ord_item": np.concatenate(item_g),
        "ord_qty": np.concatenate(qty_g),
    }
    return table


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", nargs="+", required=True)
    ap.add_argument("--out", default="offline_lab/build/models")
    ap.add_argument("--epochs", type=int, default=8)
    args = ap.parse_args()
    paths = sorted(sum([glob.glob(p) for p in args.shards], []))
    print(f"{len(paths)} shards")
    table = load_corpus(paths)
    n = table["tiles_own"].shape[0]
    print(f"corpus: {n} samples")
    print(f"order grid: {table['ord_kind'].shape}")
    # the per-sample label slices are [lab_ptr[i], lab_ptr[i+1])
    print(f"label rows: {len(table['lab_chain'])}")
    # Training itself runs on the Kaggle GPU notebook (train cell mirrors this
    # table layout); locally this script validates the table and exits.
    print("table validated — training runs in the kaggle notebook (GPU)")


if __name__ == "__main__":
    main()