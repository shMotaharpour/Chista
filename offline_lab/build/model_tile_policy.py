"""Model 1 — TilePolicyNet: state in, per-tile edge predictions out.

The architecture is NumPy-native from the ground up (matmul + ReLU + softmax,
no recurrence, no dynamic shapes): what trains in JAX exports verbatim.

Inputs (one sample):
    tiles_own (100,) int32   graph state keys; LOCKED keeps its sentinel
    tiles_opp (100,) int32   rival farm, same encoding
    scalars   (40,)  float32 money_own/opp (log1p), shed_own (12), shed_opp
                     (12), seeds (5), prices (9), mkt_inv (9) — see FORMAT.md

Head:
    score(cell, edge) = dot(tile_vec[cell], edge_emb[edge]) where tile_vec =
    tile embedding + global context. The mask is the GRAPH's own edge list of
    the tile's state — legality is the graph's, the model only ranks.
    Loss: cross-entropy over legal edges; the idle edge (the graph's empty
    chain) is one of them, so "untouched today" is a class, not a mask gap.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np


def build_shard_arrays(paths: list[str]):
    """Concatenate shard files into one fixed-shape training table."""
    import numpy as np
    tiles_own, tiles_opp, scalars = [], [], []
    lab_ptr, lab_cell, lab_chain = [0], [], []
    for p in paths:
        sh = np.load(p, allow_pickle=False)
        tiles_own.append(sh["tiles_own"])
        tiles_opp.append(sh["tiles_opp"])
        n = int(sh["n_samples"])
        scal = np.zeros((n, 40), dtype=np.float32)
        scal[:, 0] = sh["money_own"]
        scal[:, 1] = sh["money_opp"]
        scal[:, 2:14] = sh["shed_own"]
        scal[:, 14:26] = sh["shed_opp"]
        scal[:, 26:31] = sh["seeds"]
        scal[:, 31:40] = sh["prices"]
        scal[:, 39] = sh["day"] / 29.0
        scalars.append(scal)
        for i in range(n):
            lab_cell.append(sh["lab_cell"])
            lab_chain.append(sh["lab_chain"])
            lab_ptr.append(lab_ptr[-1] + len(sh["lab_cell"]))
    return (np.concatenate(tiles_own), np.concatenate(tiles_opp),
            np.concatenate(scalars), np.concatenate(lab_cell),
            np.concatenate(lab_chain), np.concatenate(lab_ptr))


class TilePolicyNet:
    """Per-tile edge ranking over the graph's own legal edges."""

    def __init__(self, n_states: int, n_chains: int, d: int = 32):
        import jax.numpy as jnp
        import jax
        self.n_states, self.n_chains, self.d = n_states, n_chains, d
        k = jax.random.PRNGKey(7)
        k1, k2, k3, k4, k5, k6 = jax.random.split(k, 6)
        def glorot(shape, kk):
            std = np.sqrt(2.0 / (shape[0] + shape[1]))
            return jax.random.normal(kk, shape) * std
        self.params = {
            "tile_emb": glorot((n_states + 1, d), k1),   # +1 = LOCKED/unknown
            "edge_emb": glorot((n_chains, d), k2),
            "w_sc": glorot((40, d), k3),
            "b_sc": jnp.zeros((d,)),
            "w_t": glorot((2 * d, d), k4),
            "b_t": jnp.zeros((d,)),
            "w_o": glorot((d, d), k5),
            "b_o": jnp.zeros((d,)),
        }

    def _forward(self, params, tiles, scal, edge_lo, edge_hi, edge_chain):
        import jax.numpy as jnp
        t = params["tile_emb"][tiles]                     # (B, 100, d)
        ctx = jnp.tanh(scal @ params["w_sc"] + params["b_sc"])
        ctx = jnp.repeat(ctx[:, None, :], 100, axis=1)    # (B, 100, d)
        tv = jnp.tanh(jnp.concatenate([t, ctx], -1) @ params["w_t"] + params["b_t"])
        tv = jnp.tanh(tv @ params["w_o"] + params["b_o"])
        # candidate edges per sample-tile: the graph's own (variable-length);
        # padded to a fixed [B, 100, E, d] window with edge_lo/edge_hi slices
        ee = params["edge_emb"][edge_chain]               # (B, 100, E, d)
        score = jnp.einsum("bkd,bked->bke", tv, ee)       # (B, 100, E)
        return score

    def train(self, paths, epochs: int = 12, lr: float = 3e-3, batch: int = 32):
        import jax
        import jax.numpy as jnp
        import optax
        tiles_own, tiles_opp, scal, lab_cell, lab_chain, lab_ptr = \
            build_shard_arrays(paths)
        # per (sample, cell) the graph's legal edges come pre-sliced in the
        # shard build; here the masked head is assembled at batch time from
        # the graph's CSR — the loop is the builder's, not the model's.
        raise NotImplementedError(
            "the masked-batch assembly needs the graph CSR; "
            "see train_tile_policy.py which drives this class")


if __name__ == "__main__":
    print("see train_tile_policy.py — this module defines the net only")
