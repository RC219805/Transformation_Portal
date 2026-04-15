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

ConsistencyPath = list[bytes]
"""Ordered list of sibling digests forming an RFC 9162 consistency proof."""

__all__ = [
    # Type aliases
    "Sha256Digest",
    "MerkleRoot",
    "AuditPath",
    "ConsistencyPath",
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
    "ct_consistency_proof",
    "ct_consistency_proof_sha256",
    "verify_ct_consistency_proof",
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
        raise ValueError(f"{name} must be exactly {_SHA256_DIGEST_SIZE} bytes (SHA-256), " f"got {len(value)} bytes")
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


def _inclusion_proof_recursive(leaves: list[bytes], start: int, end: int, leaf_index: int) -> list[bytes]:
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
        return _inclusion_proof_recursive(leaves, start, mid, leaf_index) + [_mth_recursive(leaves, mid, end)]
    else:
        # Target is in right subtree; sibling is left subtree root
        return _inclusion_proof_recursive(leaves, mid, end, leaf_index) + [_mth_recursive(leaves, start, mid)]


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


# ---------------------------------------------------------------------------
# Consistency Proofs (RFC 9162 Section 2.1.4)
# ---------------------------------------------------------------------------


def _subproof(leaves: list[bytes], m: int, start: int, end: int, complete_subtree: bool) -> list[bytes]:
    """Build consistency subproof recursively.

    This implements the RFC 9162 SUBPROOF algorithm.

    Args:
        leaves: The full list of leaf hashes.
        m: Number of leaves in the older tree (relative to the subtree root).
        start: Start index (inclusive) of the current subtree.
        end: End index (exclusive) of the current subtree.
        complete_subtree: True if this subtree is a complete subtree of the older tree.

    Returns:
        List of sibling hashes forming part of the consistency proof.
    """
    n = end - start

    if m == n:
        if complete_subtree:
            return []
        else:
            return [_mth_recursive(leaves, start, end)]

    if m == 0:
        return [_mth_recursive(leaves, start, end)]

    k = _largest_power_of_two_less_than(n)

    if m <= k:
        return _subproof(leaves, m, start, start + k, complete_subtree) + [_mth_recursive(leaves, start + k, end)]
    else:
        return _subproof(leaves, m - k, start + k, end, False) + [_mth_recursive(leaves, start, start + k)]


def ct_consistency_proof(leaf_hashes: Sequence[bytes], first_tree_size: int) -> ConsistencyPath:
    """Return the RFC 9162 consistency proof between two tree sizes.

    The consistency proof allows a verifier to confirm that a smaller tree
    is a prefix of a larger tree, ensuring append-only semantics.

    Args:
        leaf_hashes: Sequence of pre-hashed leaf values for the full (larger) tree.
        first_tree_size: Number of leaves in the older (smaller) tree.

    Returns:
        List of sibling digests forming the consistency proof.

    Raises:
        ValueError: If first_tree_size is invalid (< 1 or > len(leaf_hashes)).

    Example:
        >>> leaves = [ct_leaf_hash(b"a"), ct_leaf_hash(b"b"), ct_leaf_hash(b"c")]
        >>> proof = ct_consistency_proof(leaves, 2)
        >>> len(proof) >= 0
        True
    """
    second_tree_size = len(leaf_hashes)

    if first_tree_size < 1:
        raise ValueError("first_tree_size must be >= 1")
    if first_tree_size > second_tree_size:
        raise ValueError(f"first_tree_size ({first_tree_size}) cannot exceed " f"current tree size ({second_tree_size})")

    if first_tree_size == second_tree_size:
        return []

    leaves = list(leaf_hashes) if not isinstance(leaf_hashes, list) else leaf_hashes
    return _subproof(leaves, first_tree_size, 0, second_tree_size, True)


def ct_consistency_proof_sha256(leaf_hashes: Sequence[bytes], first_tree_size: int) -> list[str]:
    """Return consistency proof sibling digests as lowercase hex.

    Args:
        leaf_hashes: Sequence of pre-hashed leaf values for the full tree.
        first_tree_size: Number of leaves in the older (smaller) tree.

    Returns:
        List of lowercase hex strings representing the consistency proof.
    """
    return [digest.hex() for digest in ct_consistency_proof(leaf_hashes, first_tree_size)]


def verify_ct_consistency_proof(
    *,
    first_tree_size: int,
    second_tree_size: int,
    first_root: MerkleRoot,
    second_root: MerkleRoot,
    proof: Sequence[Sha256Digest],
) -> bool:
    """Verify an RFC 9162 consistency proof using constant-time comparison.

    This verifies that a tree with first_tree_size leaves is a prefix of
    a tree with second_tree_size leaves, given their respective roots.

    Args:
        first_tree_size: Number of leaves in the older (smaller) tree.
        second_tree_size: Number of leaves in the newer (larger) tree.
        first_root: The 32-byte Merkle root of the smaller tree.
        second_root: The 32-byte Merkle root of the larger tree.
        proof: Sequence of sibling hashes (the consistency proof).

    Returns:
        True if the proof is valid (the smaller tree is a prefix of the larger).

    Security:
        Uses constant-time comparison for final root verification to
        prevent timing attacks that could leak information about valid proofs.
    """
    if first_tree_size < 1 or second_tree_size < first_tree_size:
        return False

    if first_tree_size == second_tree_size:
        if len(proof) > 0:
            return False
        return hmac.compare_digest(first_root, second_root)

    proof_list = list(proof)
    proof_len = len(proof_list)

    if proof_len == 0:
        return False

    # Check if first_tree_size is a power of two
    is_power_of_two = (first_tree_size & (first_tree_size - 1)) == 0

    if is_power_of_two:
        # For power-of-two first_tree_size: simple iterative extension
        # The proof elements extend sr to the right at each tree level
        sr = first_root
        for elem in proof_list:
            sr = ct_node_hash(sr, elem)
        return hmac.compare_digest(sr, second_root)

    # Non-power-of-two case
    # proof[0] is the complete subtree hash, subsequent elements build paths
    fn = first_tree_size - 1
    sn = second_tree_size - 1
    fr = proof_list[0]
    sr = proof_list[0]
    proof_index = 1

    # Process while fn is odd (at right-child position in the tree)
    while fn & 1 == 1:
        if fn == sn:
            # Both trees have same structure at this level, combine from left
            if proof_index >= proof_len:
                return False
            fr = ct_node_hash(proof_list[proof_index], fr)
            sr = ct_node_hash(proof_list[proof_index], sr)
            proof_index += 1
        else:
            # Trees diverge: sr extends to the right first, then both combine
            if proof_index >= proof_len:
                return False
            sr = ct_node_hash(sr, proof_list[proof_index])
            proof_index += 1
            if proof_index >= proof_len:
                return False
            fr = ct_node_hash(proof_list[proof_index], fr)
            sr = ct_node_hash(proof_list[proof_index], sr)
            proof_index += 1
        fn >>= 1
        sn >>= 1

    # Continue building toward roots
    while sn > 0:
        if proof_index >= proof_len:
            break
        if fn & 1 == 1 or fn == sn:
            # Both combine from left
            fr = ct_node_hash(proof_list[proof_index], fr)
            sr = ct_node_hash(proof_list[proof_index], sr)
            proof_index += 1
        elif fn < sn:
            # Only sr extends to the right
            sr = ct_node_hash(sr, proof_list[proof_index])
            proof_index += 1
        fn >>= 1
        sn >>= 1

    if proof_index != proof_len:
        return False

    # Use constant-time comparison for both root checks
    return hmac.compare_digest(fr, first_root) and hmac.compare_digest(sr, second_root)
