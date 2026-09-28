"""Model 2 — MarketPolicyNet: state + chosen tile edges in, the day's market
queue out (24 turns x 10 slots).

Input (one sample):
    the same state block as Model 1, PLUS the day's chosen edges: per cell a
    chain id (+ its entity code) — what Model 1 predicted (or the replay's
    own labels during training).

Output — as SLOTS, not free text: for every one of the 24 turns the model
emits a 10-slot program; each slot is a masked softmax over
    {EMPTY, SELL[g], BUY_SEED[g], BUY_ANIMAL[a], BUY_PRODUCT[g], HIRE, BUY_LAND}
and, for the non-EMPTY kinds, a second head over the item and a third over
the quantity bucket. Slot 0 of a turn is the highest-priority position
(SETTLE order: land, sells, hires, buys — F032), and the model learns the
observed queue, so priority comes out of the data.

Loss: per-slot cross-entropy (kind + item + qty), masked by what is legal
in that slot (no SELL of an unheld good without shed stock in the state, no
second BUY_LAND, HIRE capped by the purse ladder).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np


def scalars_of(sh) -> np.ndarray:
    """The shared 40-dim scalar block (same builder as Model 1)."""
    n = int(sh["n_samples"])
    scal = np.zeros((n, 40), dtype=np.float32)
    scal[:, 0] = sh["money_own"]
    scal[:, 1] = sh["money_opp"]
    scal[:, 2:14] = sh["shed_own"]
    scal[:, 14:26] = sh["shed_opp"]
    scal[:, 26:31] = sh["seeds"]
    scal[:, 31:40] = sh["prices"]
    scal[:, 39] = sh["day"] / 29.0
    return scal


def edge_summary(sh) -> np.ndarray:
    """The chosen edges as a fixed vector: a histogram over chain ids.

    (100,) per-tile chain ids collapse to a (n_chains,) histogram — the day's
    production plan in one bag. Positional detail travels in Model 1's own
    head; here the MARKET cares what will be harvested/produced, i.e. which
    chains run at all.
    """
    n = int(sh["n_samples"])
    n_chains = 64
    out = np.zeros((n, n_chains), dtype=np.float32)
    for i in range(n):
        lo, hi = int(sh["lab_ptr"][i]), int(sh["lab_ptr"][i + 1])
        for j in range(lo, hi):
            cid = int(sh["lab_chain"][j])
            if 0 <= cid < n_chains:
                out[i, cid] += 1.0
    return out


def order_targets(sh, vocab: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The 24x10 program per sample as three integer grids.

    kind[b, t, s]: 0=EMPTY else 1+code of the op in the shard's own vocab
    item[b, t, s]: vocab code of the item (-1 = none)
    qty[b, t, s]:  the order's quantity, bucketed (0,1,2,3-4,5-8,9-16,17+)
    """
    n = int(sh["n_samples"])
    kind = np.zeros((n, 24, 10), dtype=np.int32)
    item = -np.ones((n, 24, 10), dtype=np.int32)
    qty = np.zeros((n, 24, 10), dtype=np.int32)
    buckets = [0, 1, 2, 4, 8, 16, 10**9]
    for i in range(n):
        for j in range(int(sh["ord_ptr"][i]), int(sh["ord_ptr"][i + 1])):
            t, s = int(sh["ord_hour"][j]), int(sh["ord_idx"][j])
            if t >= 24 or s >= 10:
                continue                      # outside the 24x10 program
            kind[i, t, s] = 1 + int(sh["ord_code"][j])
            item[i, t, s] = int(sh["ord_item"][j])
            q = int(sh["ord_qty"][j])
            qty[i, t, s] = next(b for b, cap in enumerate(buckets) if q <= cap)
    return kind, item, qty


class MarketPolicyNet:
    """State + chosen edges -> the day's 24x10 slot program."""

    def __init__(self, n_states: int, n_vocab: int, d: int = 64):
        import jax
        import jax.numpy as jnp
        self.n_states, self.n_vocab, self.d = n_states, n_vocab, d
        k = jax.random.PRNGKey(11)
        k1, k2, k3, k4, k5, k6, k7 = jax.random.split(k, 7)
        def glorot(shape, kk):
            std = np.sqrt(2.0 / (shape[0] + shape[1]))
            return jax.random.normal(kk, shape) * std
        self.params = {
            "tile_emb": glorot((n_states + 1, d), k1),
            "chain_emb": glorot((64, d), k2),
            "w_sc": glorot((40, d), k3),
            "b_sc": jnp.zeros((d,)),
            # pool: 100 tile vectors mean-pooled + chain bag + scalars
            "w_h1": glorot((3 * d, 128), k4),
            "b_h1": jnp.zeros((128,)),
            "w_h2": glorot((128, 128), k5),
            "b_h2": jnp.zeros((128,)),
            # heads: per (turn, slot) kind / item / qty-bucket
            "w_kind": glorot((128, 24 * 10 * (n_vocab + 1)), k6),
            "w_item": glorot((128, 24 * 10 * (n_vocab + 1)), k7),
            # qty shares w_item's readout with a small map
            "w_qty": glorot((n_vocab + 1, 8), k5),
        }


if __name__ == "__main__":
    print("see train_market_policy.py — this module defines the net only")
