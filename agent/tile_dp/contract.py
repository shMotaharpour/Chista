"""The artifact's identity: the fingerprints a load is checked against.

Ids in an artifact are POSITIONS in the table that produced it, so a load has to compare the
table it read with the one the artifact was built from. The engine and the day length are
stamped the same way: an artifact that disagrees is refused loudly instead of decoding into
other chains.
"""

from __future__ import annotations

from pathlib import Path

from agent.world.rules import TURNS_PER_DAY

from agent.tile_dp.chains import TileChain, chain_name, loaded_chains
from agent.tile_dp.tile_state import KEY_BITS

def _fingerprint(chains: tuple[TileChain, ...]) -> str:
    """The registry fingerprint of a chain list: ids are positions, so this is what an
    artifact's info carries and what a build compares."""
    from hashlib import sha256
    return sha256("\n".join(chain_name(c) for c in chains).encode()).hexdigest()[:16]


def fingerprint_chains(chains) -> str:
    """The ONE fingerprint of a chain table, used by the builder and by the loader.

    Ids are positions, so an artifact that stores ids is only readable together with the
    exact table that produced it. Two different hashes for the same table would make the
    agent refuse its own build.
    """
    return _fingerprint(tuple(chains))


def registry_fingerprint() -> str:
    """Fingerprint of the loaded registry: ids are positions, so an artifact that stores
    ids is only readable together with the exact registry that produced it."""
    from hashlib import sha256
    return _fingerprint(loaded_chains())


def engine_fingerprint() -> str:
    """Fingerprint of the engine source the graph decodes against."""
    from hashlib import sha1

    from kaggle_environments.envs.kaggriculture import kaggriculture as K
    return sha1(Path(K.__file__).read_bytes()).hexdigest()[:8]


def contract_id() -> str:
    """What an artifact IS, from what the agent can check at runtime: the registry it indexes
    into, the engine, the day length and the key layout.

    Nothing here reads the build's own source: the submission ships `agent/` alone, so a
    fingerprint that needs `offline_lab/` on disk cannot be computed where it matters.
    """
    from agent.tile_dp.tile_state import KEY_BITS
    return (f"tile-dp/reg={registry_fingerprint()}"
            f"+eng={engine_fingerprint()}+tpd={TURNS_PER_DAY}+pb={KEY_BITS}")
