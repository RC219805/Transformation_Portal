"""Certificate-Transparency-style Merkle helpers.

Implements the RFC 9162 Merkle Tree Hash construction:

- leaf hash: SHA256(0x00 || leaf_bytes)
- node hash: SHA256(0x01 || left || right)

The tree root uses the split-tree construction rather than duplicate-last
pairing so inclusion proofs are transparency-grade and path-independent.
"""

from __future__ import annotations

import hashlib
from typing import Sequence


def _sha256(payload: bytes) -> bytes:
    return hashlib.sha256(payload).digest()


def ct_leaf_hash(leaf_bytes: bytes) -> bytes:
    """Hash a Merkle leaf under the CT leaf-domain separator."""
    return _sha256(b"\x00" + leaf_bytes)


def ct_node_hash(left: bytes, right: bytes) -> bytes:
    """Hash an interior Merkle node under the CT node-domain separator."""
    return _sha256(b"\x01" + left + right)


def _largest_power_of_two_less_than(n: int) -> int:
    if n <= 1:
        raise ValueError("n must be greater than 1")
    return 1 << ((n - 1).bit_length() - 1)


def _mth(leaf_hashes: Sequence[bytes]) -> bytes:
    count = len(leaf_hashes)
    if count == 0:
        return _sha256(b"")
    if count == 1:
        return leaf_hashes[0]
    split = _largest_power_of_two_less_than(count)
    return ct_node_hash(
        _mth(leaf_hashes[:split]),
        _mth(leaf_hashes[split:]),
    )


def ct_merkle_root(leaf_hashes: Sequence[bytes]) -> bytes:
    """Return the CT Merkle root bytes for a sequence of leaf hashes."""
    return _mth(tuple(leaf_hashes))


def ct_merkle_root_sha256(leaf_hashes: Sequence[bytes]) -> str:
    """Return the CT Merkle root as lowercase hex."""
    return ct_merkle_root(leaf_hashes).hex()


def _inclusion_proof_hashes(leaf_hashes: Sequence[bytes], leaf_index: int) -> list[bytes]:
    count = len(leaf_hashes)
    if count == 0:
        raise ValueError("cannot build inclusion proof for an empty tree")
    if leaf_index < 0 or leaf_index >= count:
        raise IndexError("leaf_index out of range")
    if count == 1:
        return []

    split = _largest_power_of_two_less_than(count)
    if leaf_index < split:
        return _inclusion_proof_hashes(leaf_hashes[:split], leaf_index) + [_mth(leaf_hashes[split:])]
    return _inclusion_proof_hashes(leaf_hashes[split:], leaf_index - split) + [_mth(leaf_hashes[:split])]


def ct_inclusion_proof(leaf_hashes: Sequence[bytes], leaf_index: int) -> list[bytes]:
    """Return the RFC 9162 audit path for the selected leaf."""
    return _inclusion_proof_hashes(tuple(leaf_hashes), leaf_index)


def ct_inclusion_proof_sha256(leaf_hashes: Sequence[bytes], leaf_index: int) -> list[str]:
    """Return inclusion proof sibling digests as lowercase hex."""
    return [digest.hex() for digest in ct_inclusion_proof(leaf_hashes, leaf_index)]


def verify_ct_inclusion_proof(
    *,
    leaf_hash: bytes,
    leaf_index: int,
    tree_size: int,
    proof: Sequence[bytes],
    expected_root: bytes,
) -> bool:
    """Verify an RFC 9162 inclusion proof.

    The implementation follows the audit-path reconstruction algorithm using
    the caller-provided leaf index and tree size.
    """
    if tree_size <= 0:
        return False
    if leaf_index < 0 or leaf_index >= tree_size:
        return False

    fn = leaf_index
    sn = tree_size - 1
    digest = leaf_hash
    proof_index = 0

    while sn > 0:
        if fn % 2 == 1:
            if proof_index >= len(proof):
                return False
            digest = ct_node_hash(proof[proof_index], digest)
            proof_index += 1
        elif fn < sn:
            if proof_index >= len(proof):
                return False
            digest = ct_node_hash(digest, proof[proof_index])
            proof_index += 1
        fn //= 2
        sn //= 2

    return proof_index == len(proof) and digest == expected_root
