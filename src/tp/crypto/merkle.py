"""Merkle tree helpers shared across deterministic tooling layers.

This module provides a simple Merkle root computation using duplicate-last
odd-leaf handling. For RFC 9162 Certificate Transparency-style trees with
verifiable inclusion proofs, use the ct_merkle module instead.
"""

from __future__ import annotations

import hashlib
from typing import Sequence

__all__ = ["merkle_root_sha256"]

# Pre-compute empty tree hash at module load
_EMPTY_HASH_HEX = hashlib.sha256(b"").hexdigest()


def merkle_root_sha256(leaf_hashes: Sequence[bytes]) -> str:
    """Compute Merkle root using duplicate-last odd-leaf handling.

    This implementation pairs adjacent nodes and duplicates the last node
    when the count is odd, which is simpler but produces different results
    than the RFC 9162 split-tree construction.

    Args:
        leaf_hashes: Sequence of pre-hashed leaf values (raw bytes).

    Returns:
        Lowercase hexadecimal string of the 32-byte Merkle root.

    Example:
        >>> import hashlib
        >>> leaves = [hashlib.sha256(b"a").digest(), hashlib.sha256(b"b").digest()]
        >>> root = merkle_root_sha256(leaves)
        >>> len(root)
        64
    """
    if not leaf_hashes:
        return _EMPTY_HASH_HEX

    # Work with a mutable list to avoid repeated allocations
    layer = list(leaf_hashes)

    while len(layer) > 1:
        # Duplicate last element if odd count
        if len(layer) % 2 == 1:
            layer.append(layer[-1])

        # Build next layer by pairing adjacent nodes
        next_layer: list[bytes] = []
        for index in range(0, len(layer), 2):
            combined = layer[index] + layer[index + 1]
            next_layer.append(hashlib.sha256(combined).digest())
        layer = next_layer

    return layer[0].hex()
