"""Tests for CT-style Merkle helpers."""

from __future__ import annotations

import hashlib

import pytest

from tp.crypto.ct_merkle import (
    AuditPath,
    ConsistencyPath,
    MerkleRoot,
    Sha256Digest,
    ct_consistency_proof,
    ct_inclusion_proof,
    ct_leaf_hash,
    ct_merkle_root,
    ct_node_hash,
    validate_sha256_digest,
    verify_ct_consistency_proof,
    verify_ct_inclusion_proof,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# RFC 9162 Golden Test Vectors
# ---------------------------------------------------------------------------
# These test vectors verify compliance with RFC 9162 Merkle Tree Hash
# construction. The leaf data is d0, d1, d2, ... dn as per the RFC examples.


class TestRfc9162GoldenVectors:
    """RFC 9162 compliance verification using golden test vectors.

    Test vectors derived from RFC 9162 Section 2.1.3 examples.
    Leaf data: d0=b"", d1=b"\\x00", d2=b"\\x01", etc.
    """

    @pytest.fixture
    def rfc_leaves(self) -> list[bytes]:
        """Generate leaf hashes as per RFC 9162 examples."""
        # d0 = empty bytes, d1 = 0x00, d2 = 0x01, ..., d6 = 0x05
        leaf_data = [b"", b"\x00", b"\x01", b"\x02", b"\x03", b"\x04", b"\x05"]
        return [ct_leaf_hash(data) for data in leaf_data]

    def test_leaf_hash_domain_separation(self) -> None:
        """Verify leaf hash uses 0x00 prefix (domain separation)."""
        leaf = ct_leaf_hash(b"test")
        expected = hashlib.sha256(b"\x00test").digest()
        assert leaf == expected

    def test_node_hash_domain_separation(self) -> None:
        """Verify node hash uses 0x01 prefix (domain separation)."""
        left = b"L" * 32
        right = b"R" * 32
        node = ct_node_hash(left, right)
        expected = hashlib.sha256(b"\x01" + left + right).digest()
        assert node == expected

    def test_empty_tree_root(self) -> None:
        """MTH({}) = SHA-256("") per RFC 9162."""
        root = ct_merkle_root([])
        expected = hashlib.sha256(b"").digest()
        assert root == expected
        assert root.hex() == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"

    def test_single_leaf_root(self, rfc_leaves: list[bytes]) -> None:
        """MTH({d0}) = SHA-256(0x00 || d0) per RFC 9162."""
        root = ct_merkle_root(rfc_leaves[:1])
        assert root == rfc_leaves[0]

    def test_two_leaf_root(self, rfc_leaves: list[bytes]) -> None:
        """MTH({d0, d1}) = SHA-256(0x01 || MTH({d0}) || MTH({d1}))."""
        root = ct_merkle_root(rfc_leaves[:2])
        expected = ct_node_hash(rfc_leaves[0], rfc_leaves[1])
        assert root == expected

    def test_three_leaf_root_split_tree(self, rfc_leaves: list[bytes]) -> None:
        """MTH with 3 leaves uses split-tree construction, not duplicate-last.

        RFC 9162: k = 2 (largest power of 2 < 3)
        MTH({d0,d1,d2}) = SHA-256(0x01 || MTH({d0,d1}) || MTH({d2}))
        """
        root = ct_merkle_root(rfc_leaves[:3])

        left = ct_node_hash(rfc_leaves[0], rfc_leaves[1])
        right = rfc_leaves[2]
        expected = ct_node_hash(left, right)
        assert root == expected

    def test_seven_leaf_root_full_structure(self, rfc_leaves: list[bytes]) -> None:
        """Verify complete tree structure for 7 leaves per RFC 9162.

        k = 4 (largest power of 2 < 7)
        MTH({d0..d6}) = SHA-256(0x01 || MTH({d0..d3}) || MTH({d4..d6}))
        """
        root = ct_merkle_root(rfc_leaves[:7])

        # Left subtree: MTH({d0,d1,d2,d3})
        left_left = ct_node_hash(rfc_leaves[0], rfc_leaves[1])
        left_right = ct_node_hash(rfc_leaves[2], rfc_leaves[3])
        left = ct_node_hash(left_left, left_right)

        # Right subtree: MTH({d4,d5,d6}) with k=2
        right_left = ct_node_hash(rfc_leaves[4], rfc_leaves[5])
        right_right = rfc_leaves[6]
        right = ct_node_hash(right_left, right_right)

        expected = ct_node_hash(left, right)
        assert root == expected

    def test_inclusion_proof_for_seven_leaf_tree(self, rfc_leaves: list[bytes]) -> None:
        """Verify inclusion proof structure matches RFC 9162 examples."""
        root = ct_merkle_root(rfc_leaves[:7])

        # Verify all leaves can be proven
        for i in range(7):
            proof = ct_inclusion_proof(rfc_leaves[:7], i)
            assert verify_ct_inclusion_proof(
                leaf_hash=rfc_leaves[i],
                leaf_index=i,
                tree_size=7,
                proof=proof,
                expected_root=root,
            )

    def test_inclusion_proof_length_bounds(self, rfc_leaves: list[bytes]) -> None:
        """Proof length should be ceil(log2(n)) for n leaves."""
        for n in range(1, 8):
            leaves = rfc_leaves[:n]
            for i in range(n):
                proof = ct_inclusion_proof(leaves, i)
                # Proof length is at most ceil(log2(n))
                assert len(proof) <= (n - 1).bit_length()


class TestCtMerkleRoot:
    """Tests for ct_merkle_root function."""

    def test_empty_tree_uses_sha256_empty(self) -> None:
        assert ct_merkle_root([]).hex() == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"

    def test_single_leaf_equals_leaf_hash(self) -> None:
        leaf = ct_leaf_hash(b"artifact-a")
        assert ct_merkle_root([leaf]) == leaf


class TestCtInclusionProof:
    """Tests for inclusion proof generation and verification."""

    def test_round_trip_for_odd_tree(self) -> None:
        leaf_hashes = [ct_leaf_hash(value) for value in (b"a", b"b", b"c", b"d", b"e")]
        root = ct_merkle_root(leaf_hashes)

        for index, leaf_hash in enumerate(leaf_hashes):
            proof = ct_inclusion_proof(leaf_hashes, index)
            assert verify_ct_inclusion_proof(
                leaf_hash=leaf_hash,
                leaf_index=index,
                tree_size=len(leaf_hashes),
                proof=proof,
                expected_root=root,
            )

    def test_rejects_wrong_leaf(self) -> None:
        leaf_hashes = [ct_leaf_hash(value) for value in (b"a", b"b", b"c")]
        root = ct_merkle_root(leaf_hashes)
        proof = ct_inclusion_proof(leaf_hashes, 1)

        assert not verify_ct_inclusion_proof(
            leaf_hash=ct_leaf_hash(b"tampered"),
            leaf_index=1,
            tree_size=len(leaf_hashes),
            proof=proof,
            expected_root=root,
        )

    def test_rejects_wrong_root(self) -> None:
        """Verify that tampered root is rejected (constant-time comparison)."""
        leaf_hashes = [ct_leaf_hash(value) for value in (b"a", b"b", b"c")]
        root = ct_merkle_root(leaf_hashes)
        proof = ct_inclusion_proof(leaf_hashes, 0)

        # Tamper with one byte of the root
        tampered_root = bytes([root[0] ^ 0x01]) + root[1:]

        assert not verify_ct_inclusion_proof(
            leaf_hash=leaf_hashes[0],
            leaf_index=0,
            tree_size=len(leaf_hashes),
            proof=proof,
            expected_root=tampered_root,
        )


class TestCtConsistencyProof:
    """Tests for consistency proof generation and verification."""

    def test_same_tree_size_returns_empty_proof(self) -> None:
        """Consistency proof for same tree size should be empty."""
        leaf_hashes = [ct_leaf_hash(value) for value in (b"a", b"b", b"c")]
        proof = ct_consistency_proof(leaf_hashes, 3)
        assert proof == []

    def test_consistency_proof_round_trip(self) -> None:
        """Verify consistency proofs for various tree growth scenarios."""
        # Build progressively larger trees
        leaves = [ct_leaf_hash(bytes([i])) for i in range(8)]

        for first_size in range(1, 8):
            first_root = ct_merkle_root(leaves[:first_size])

            for second_size in range(first_size, 9):
                second_root = ct_merkle_root(leaves[:second_size])
                proof = ct_consistency_proof(leaves[:second_size], first_size)

                assert verify_ct_consistency_proof(
                    first_tree_size=first_size,
                    second_tree_size=second_size,
                    first_root=first_root,
                    second_root=second_root,
                    proof=proof,
                ), f"Failed for first_size={first_size}, second_size={second_size}"

    def test_rejects_inconsistent_roots(self) -> None:
        """Consistency verification rejects mismatched roots."""
        leaves = [ct_leaf_hash(bytes([i])) for i in range(4)]
        first_root = ct_merkle_root(leaves[:2])
        second_root = ct_merkle_root(leaves[:4])

        # Get valid proof
        proof = ct_consistency_proof(leaves[:4], 2)

        # Tamper with first root
        tampered_first = bytes([first_root[0] ^ 0xFF]) + first_root[1:]
        assert not verify_ct_consistency_proof(
            first_tree_size=2,
            second_tree_size=4,
            first_root=tampered_first,
            second_root=second_root,
            proof=proof,
        )

        # Tamper with second root
        tampered_second = bytes([second_root[0] ^ 0xFF]) + second_root[1:]
        assert not verify_ct_consistency_proof(
            first_tree_size=2,
            second_tree_size=4,
            first_root=first_root,
            second_root=tampered_second,
            proof=proof,
        )

    def test_rejects_invalid_tree_sizes(self) -> None:
        """Consistency proof generation rejects invalid parameters."""
        leaves = [ct_leaf_hash(b"a"), ct_leaf_hash(b"b")]

        with pytest.raises(ValueError, match="first_tree_size must be >= 1"):
            ct_consistency_proof(leaves, 0)

        with pytest.raises(ValueError, match="cannot exceed"):
            ct_consistency_proof(leaves, 3)

    def test_verifier_rejects_invalid_sizes(self) -> None:
        """Consistency verification rejects invalid tree sizes."""
        root = ct_merkle_root([ct_leaf_hash(b"a")])

        # first_tree_size < 1
        assert not verify_ct_consistency_proof(
            first_tree_size=0,
            second_tree_size=1,
            first_root=root,
            second_root=root,
            proof=[],
        )

        # first > second
        assert not verify_ct_consistency_proof(
            first_tree_size=2,
            second_tree_size=1,
            first_root=root,
            second_root=root,
            proof=[],
        )

    def test_power_of_two_tree_sizes(self) -> None:
        """Verify consistency proofs work for power-of-two tree sizes."""
        leaves = [ct_leaf_hash(bytes([i])) for i in range(8)]

        # Power of two to power of two
        for first_size in [1, 2, 4]:
            first_root = ct_merkle_root(leaves[:first_size])
            second_root = ct_merkle_root(leaves[:8])
            proof = ct_consistency_proof(leaves[:8], first_size)

            assert verify_ct_consistency_proof(
                first_tree_size=first_size,
                second_tree_size=8,
                first_root=first_root,
                second_root=second_root,
                proof=proof,
            )


class TestValidateSha256Digest:
    """Tests for validate_sha256_digest input validation helper."""

    def test_accepts_valid_32_byte_digest(self) -> None:
        digest = hashlib.sha256(b"test data").digest()
        result = validate_sha256_digest(digest)
        assert result == digest
        assert len(result) == 32

    def test_accepts_ct_leaf_hash_output(self) -> None:
        leaf = ct_leaf_hash(b"artifact")
        result = validate_sha256_digest(leaf, name="leaf_hash")
        assert result == leaf

    def test_rejects_non_bytes_type(self) -> None:
        with pytest.raises(TypeError, match="must be bytes"):
            validate_sha256_digest("not bytes")  # type: ignore[arg-type]

    def test_rejects_too_short(self) -> None:
        with pytest.raises(ValueError, match="must be exactly 32 bytes"):
            validate_sha256_digest(b"short")

    def test_rejects_too_long(self) -> None:
        too_long = b"x" * 64
        with pytest.raises(ValueError, match="must be exactly 32 bytes"):
            validate_sha256_digest(too_long)

    def test_includes_name_in_error_message(self) -> None:
        with pytest.raises(TypeError, match="my_digest"):
            validate_sha256_digest(123, name="my_digest")  # type: ignore[arg-type]

        with pytest.raises(ValueError, match="leaf_hash"):
            validate_sha256_digest(b"short", name="leaf_hash")


class TestTypeAliases:
    """Tests for type alias exports and documentation."""

    def test_sha256_digest_alias_is_bytes(self) -> None:
        """Sha256Digest should be an alias for bytes."""
        assert Sha256Digest is bytes

    def test_merkle_root_alias_is_bytes(self) -> None:
        """MerkleRoot should be an alias for bytes."""
        assert MerkleRoot is bytes

    def test_audit_path_alias_is_list_bytes(self) -> None:
        """AuditPath should be list[bytes]."""
        # AuditPath is a type alias, check it's the expected form
        assert AuditPath == list[bytes]

    def test_consistency_path_alias_is_list_bytes(self) -> None:
        """ConsistencyPath should be list[bytes]."""
        assert ConsistencyPath == list[bytes]

    def test_aliases_importable_from_package(self) -> None:
        """Type aliases should be importable from tp.crypto."""
        from tp.crypto import AuditPath, ConsistencyPath, MerkleRoot, Sha256Digest

        assert Sha256Digest is bytes
        assert MerkleRoot is bytes
        assert AuditPath == list[bytes]
        assert ConsistencyPath == list[bytes]
