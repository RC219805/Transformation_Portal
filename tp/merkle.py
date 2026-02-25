"""Shared Merkle helpers for deterministic hash trees."""

from __future__ import annotations

import hashlib
from typing import Sequence


def merkle_root_sha256(leaf_hashes: Sequence[bytes]) -> str:
    """Compute Merkle root using duplicate-last odd-leaf handling."""
    if not leaf_hashes:
        return hashlib.sha256(b"").hexdigest()

    layer = list(leaf_hashes)
    while len(layer) > 1:
        if len(layer) % 2 == 1:
            layer.append(layer[-1])

        next_layer: list[bytes] = []
        for index in range(0, len(layer), 2):
            next_layer.append(hashlib.sha256(layer[index] + layer[index + 1]).digest())
        layer = next_layer

    return layer[0].hex()
