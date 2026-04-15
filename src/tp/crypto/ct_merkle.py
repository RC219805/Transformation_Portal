"""Certificate-Transparency-style Merkle helpers.

Implements the RFC 9162 Merkle Tree Hash construction:

- leaf hash: SHA256(0x00 || leaf_bytes)
- node hash: SHA256(0x01 || left || right)

The tree root uses the split-tree construction rather than duplicate-last
pairing so inclusion proofs are transparency-grade and path-independent.
"""

from __future__ import annotations

import hashlib
import hmac
from typing import Sequence

# ---------------------------------------------------------------------------
# Type aliases for API clarity
# ---------------------------------------------------------------------------
Sha256Digest = bytes
"""A 32-byte SHA-256 digest."""

MerkleRoot = bytes
"""A 32-byte Merkle tree root (SHA-256)."""

AuditPath = list[bytes]
"""Ordered list of sibling digests forming an RFC 9162 inclusion proof."""

__all__ = [
    # Type aliases
    "Sha256Digest",
    "MerkleRoot",
    "AuditPath",
    # Validation
    "validate_sha256_digest",
    # CT-style Merkle functions
    "ct_leaf_hash",
    "ct_node_hash",
    "ct_merkle_root",
    "ct_merkle_root_sha256",
    "ct_inclusion_proof",
    "ct_inclusion_proof_sha256",
    "verify_ct_inclusion_proof",
]

# Pre-allocated domain separators for CT hashing (module-level constants)
_LEAF_PREFIX = b"\x00"
_NODE_PREFIX = b"\x01"
_EMPTY_HASH = hashlib.sha256(b"").digest()

# Expected size of a SHA-256 digest in bytes
_SHA256_DIGEST_SIZE = 32


def validate_sha256_digest(value: bytes, name: str = "digest") -> Sha256Digest:
    """Validate that value is exactly 32 bytes (SHA-256 digest size).

    Args:
        value: The bytes object to validate.
        name: Human-readable name for error messages (e.g., "leaf_hash").

    Returns:
        The validated value (unchanged) for fluent chaining.

    Raises:
        TypeError: If value is not bytes.
        ValueError: If value is not exactly 32 bytes.

    Example:
        >>> digest = validate_sha256_digest(hashlib.sha256(b"test").digest())
        >>> len(digest)
        32
    """
    if not isinstance(value, bytes):
        raise TypeError(f"{name} must be bytes, got {type(value).__name__}")
    if len(value) != _SHA256_DIGEST_SIZE:
        raise ValueError(
            f"{name} must be exactly {_SHA256_DIGEST_SIZE} bytes (SHA-256), "
            f"got {len(value)} bytes"
        )
    return value


def _sha256(payload: bytes) -> bytes:
    """Compute SHA-256 digest of the given payload."""
    return hashlib.sha256(payload).digest()


def ct_leaf_hash(leaf_bytes: bytes) -> bytes:
    """Hash a Merkle leaf under the CT leaf-domain separator (0x00 prefix)."""
    return _sha256(_LEAF_PREFIX + leaf_bytes)


def ct_node_hash(left: bytes, right: bytes) -> bytes:
    """Hash an interior Merkle node under the CT node-domain separator (0x01 prefix)."""
    return _sha256(_NODE_PREFIX + left + right)


def _largest_power_of_two_less_than(n: int) -> int:
    """Return the largest power of two less than n.

    Raises:
        ValueError: If n <= 1 (no valid power of two exists).
    """
    if n <= 1:
        raise ValueError("n must be greater than 1")
    return 1 << ((n - 1).bit_length() - 1)


def _mth_recursive(leaves: list[bytes], start: int, end: int) -> bytes:
    """Recursive MTH computation using index bounds to avoid slice copies.

    Args:
        leaves: The full list of leaf hashes (not copied on recursive calls).
        start: Start index (inclusive) of the current subtree.
        end: End index (exclusive) of the current subtree.

    Returns:
        The Merkle root hash for the specified subtree.
    """
    size = end - start
    if size == 1:
        return leaves[start]
    split = _largest_power_of_two_less_than(size)
    return ct_node_hash(
        _mth_recursive(leaves, start, start + split),
        _mth_recursive(leaves, start + split, end),
    )


def ct_merkle_root(leaf_hashes: Sequence[bytes]) -> bytes:
    """Return the CT Merkle root bytes for a sequence of leaf hashes.

    Implements the RFC 9162 Merkle Tree Hash (MTH) construction using
    index-based recursion to minimize memory allocations.

    Args:
        leaf_hashes: Sequence of pre-hashed leaf values (typically from ct_leaf_hash).

    Returns:
        The 32-byte SHA-256 Merkle root digest.
    """
    count = len(leaf_hashes)
    if count == 0:
        return _EMPTY_HASH
    if count == 1:
        return leaf_hashes[0]

    # Convert to list once for efficient indexing
    leaves = list(leaf_hashes) if not isinstance(leaf_hashes, list) else leaf_hashes
    return _mth_recursive(leaves, 0, count)


def ct_merkle_root_sha256(leaf_hashes: Sequence[bytes]) -> str:
    """Return the CT Merkle root as lowercase hex.

    Args:
        leaf_hashes: Sequence of pre-hashed leaf values.

    Returns:
        Lowercase hexadecimal string of the Merkle root.
    """
    return ct_merkle_root(leaf_hashes).hex()


def _inclusion_proof_recursive(
    leaves: list[bytes], start: int, end: int, leaf_index: int
) -> list[bytes]:
    """Build inclusion proof recursively using index bounds.

    Args:
        leaves: The full list of leaf hashes.
        start: Start index (inclusive) of the current subtree.
        end: End index (exclusive) of the current subtree.
        leaf_index: Absolute index of the target leaf within the full tree.

    Returns:
        List of sibling hashes forming the audit path.
    """
    size = end - start
    if size == 1:
        return []

    split = _largest_power_of_two_less_than(size)
    mid = start + split

    if leaf_index < mid:
        # Target is in left subtree; sibling is right subtree root
        return _inclusion_proof_recursive(leaves, start, mid, leaf_index) + [
            _mth_recursive(leaves, mid, end)
        ]
    else:
        # Target is in right subtree; sibling is left subtree root
        return _inclusion_proof_recursive(leaves, mid, end, leaf_index) + [
            _mth_recursive(leaves, start, mid)
        ]


def ct_inclusion_proof(leaf_hashes: Sequence[bytes], leaf_index: int) -> list[bytes]:
    """Return the RFC 9162 audit path for the selected leaf.

    The audit path consists of sibling hashes needed to recompute the
    Merkle root from the specified leaf.

    Args:
        leaf_hashes: Sequence of pre-hashed leaf values.
        leaf_index: Zero-based index of the leaf to prove inclusion for.

    Returns:
        List of sibling digests forming the audit path.

    Raises:
        ValueError: If the tree is empty.
        IndexError: If leaf_index is out of range.
    """
    count = len(leaf_hashes)
    if count == 0:
        raise ValueError("cannot build inclusion proof for an empty tree")
    if leaf_index < 0 or leaf_index >= count:
        raise IndexError("leaf_index out of range")
    if count == 1:
        return []

    leaves = list(leaf_hashes) if not isinstance(leaf_hashes, list) else leaf_hashes
    return _inclusion_proof_recursive(leaves, 0, count, leaf_index)


def ct_inclusion_proof_sha256(leaf_hashes: Sequence[bytes], leaf_index: int) -> list[str]:
    """Return inclusion proof sibling digests as lowercase hex.

    Args:
        leaf_hashes: Sequence of pre-hashed leaf values.
        leaf_index: Zero-based index of the leaf to prove inclusion for.

    Returns:
        List of lowercase hex strings representing the audit path.
    """
    return [digest.hex() for digest in ct_inclusion_proof(leaf_hashes, leaf_index)]


def verify_ct_inclusion_proof(
    *,
    leaf_hash: Sha256Digest,
    leaf_index: int,
    tree_size: int,
    proof: Sequence[Sha256Digest],
    expected_root: MerkleRoot,
) -> bool:
    """Verify an RFC 9162 inclusion proof using constant-time comparison.

    The implementation follows the audit-path reconstruction algorithm using
    the caller-provided leaf index and tree size. The final root comparison
    uses ``hmac.compare_digest`` to prevent timing side-channel attacks.

    Args:
        leaf_hash: The 32-byte hash of the leaf being verified.
        leaf_index: Zero-based index of the leaf in the original tree.
        tree_size: Total number of leaves in the tree.
        proof: Sequence of sibling hashes (the audit path).
        expected_root: The expected 32-byte Merkle root to verify against.

    Returns:
        True if the proof is valid and reconstructs the expected root.

    Security:
        Uses constant-time comparison for the final root verification to
        prevent timing attacks that could leak information about valid proofs.
    """
    if tree_size <= 0:
        return False
    if leaf_index < 0 or leaf_index >= tree_size:
        return False

    fn = leaf_index
    sn = tree_size - 1
    digest = leaf_hash
    proof_index = 0
    proof_len = len(proof)

    while sn > 0:
        if fn % 2 == 1:
            if proof_index >= proof_len:
                return False
            digest = ct_node_hash(proof[proof_index], digest)
            proof_index += 1
        elif fn < sn:
            if proof_index >= proof_len:
                return False
            digest = ct_node_hash(digest, proof[proof_index])
            proof_index += 1
        fn //= 2
        sn //= 2

    # Use constant-time comparison to prevent timing side-channel attacks
    return proof_index == proof_len and hmac.compare_digest(digest, expected_root)
