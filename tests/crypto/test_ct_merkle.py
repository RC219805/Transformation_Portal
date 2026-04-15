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
#
# PINNED EXPECTED VALUES: These are pre-computed reference values that would
# catch any implementation that is consistently wrong. The computation is:
#   leaf_hash(d) = SHA256(0x00 || d)
#   node_hash(l, r) = SHA256(0x01 || l || r)
#
# Leaf data from RFC 9162 Section 2.1.3:
#   d0 = "" (empty), d1 = 0x00, d2 = 0x01, d3 = 0x02, d4 = 0x03, d5 = 0x04, d6 = 0x05

# Pre-computed leaf hashes (SHA256(0x00 || leaf_data))
RFC_LEAF_HASH_D0 = "6e340b9cffb37a989ca544e6bb780a2c78901d3fb33738768511a30617afa01d"  # SHA256(0x00 || "")
RFC_LEAF_HASH_D1 = "96a296d224f285c67bee93c30f8a309157f0daa35dc5b87e410b78630a09cfc7"  # SHA256(0x00 || 0x00)
RFC_LEAF_HASH_D2 = "b413f47d13ee2fe6c845b2ee141af81de858df4ec549a58b7970bb96645bc8d2"  # SHA256(0x00 || 0x01)
RFC_LEAF_HASH_D3 = "fcf0a6c700dd13e274b6fba8deea8dd9b26e4eedde3495717cac8408c9c5177f"  # SHA256(0x00 || 0x02)
RFC_LEAF_HASH_D4 = "583c7dfb7b3055d99465544032a571e10a134b1b6f769422bbb71fd7fa167a5d"  # SHA256(0x00 || 0x03)
RFC_LEAF_HASH_D5 = "4f35212d12f9ad2036492c95f1fe79baf4ec7bd9bef3dffa7579f2293ff546a4"  # SHA256(0x00 || 0x04)
RFC_LEAF_HASH_D6 = "9f1afa4dc124cba73134e82ff50f17c8f7164257c79fed9a13f5943a6acb8e3d"  # SHA256(0x00 || 0x05)

# Pre-computed Merkle roots for various tree sizes
RFC_ROOT_1_LEAF = RFC_LEAF_HASH_D0  # MTH({d0}) = leaf_hash(d0)
RFC_ROOT_2_LEAVES = "fac54203e7cc696cf0dfcb42c92a1d9dbaf70ad9e621f4bd8d98662f00e3c125"  # MTH({d0, d1})
RFC_ROOT_3_LEAVES = "68cb24df6ba89442113931dd829cbcae8ae19a76996ce0c4b7d9d65d168d35d2"  # MTH({d0, d1, d2})
RFC_ROOT_7_LEAVES = "f50c51ec2b7e07916f1a744c80e39f0b5d2932c2a0f411a75c9ff869013a75f9"  # MTH({d0..d6})


class TestRfc9162GoldenVectors:
    """RFC 9162 compliance verification using golden test vectors.

    Test vectors derived from RFC 9162 Section 2.1.3 examples.
    Leaf data: d0=b"", d1=b"\\x00", d2=b"\\x01", etc.

    These tests use PINNED expected hex values to ensure the implementation
    matches the RFC specification independently (not self-referentially).
    """

    @pytest.fixture
    def rfc_leaves(self) -> list[bytes]:
        """Generate leaf hashes as per RFC 9162 examples."""
        # d0 = empty bytes, d1 = 0x00, d2 = 0x01, ..., d6 = 0x05
        leaf_data = [b"", b"\x00", b"\x01", b"\x02", b"\x03", b"\x04", b"\x05"]
        return [ct_leaf_hash(data) for data in leaf_data]

    def test_leaf_hash_pinned_vectors(self) -> None:
        """Verify leaf hashes against pinned RFC 9162 expected values."""
        assert ct_leaf_hash(b"").hex() == RFC_LEAF_HASH_D0
        assert ct_leaf_hash(b"\x00").hex() == RFC_LEAF_HASH_D1
        assert ct_leaf_hash(b"\x01").hex() == RFC_LEAF_HASH_D2
        assert ct_leaf_hash(b"\x02").hex() == RFC_LEAF_HASH_D3
        assert ct_leaf_hash(b"\x03").hex() == RFC_LEAF_HASH_D4
        assert ct_leaf_hash(b"\x04").hex() == RFC_LEAF_HASH_D5
        assert ct_leaf_hash(b"\x05").hex() == RFC_LEAF_HASH_D6

    def test_merkle_root_pinned_vectors(self, rfc_leaves: list[bytes]) -> None:
        """Verify Merkle roots against pinned RFC 9162 expected values."""
        # Single leaf: MTH({d0})
        assert ct_merkle_root(rfc_leaves[:1]).hex() == RFC_ROOT_1_LEAF

        # Two leaves: MTH({d0, d1})
        assert ct_merkle_root(rfc_leaves[:2]).hex() == RFC_ROOT_2_LEAVES

        # Three leaves: MTH({d0, d1, d2})
        assert ct_merkle_root(rfc_leaves[:3]).hex() == RFC_ROOT_3_LEAVES

        # Seven leaves: MTH({d0..d6})
        assert ct_merkle_root(rfc_leaves[:7]).hex() == RFC_ROOT_7_LEAVES

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


class TestVerifierInputValidation:
    """Tests for verifier input validation (fail-closed behavior).

    Verifiers should return False (not raise) for malformed inputs like:
    - Wrong types (str, int, bytearray)
    - Wrong lengths (not 32 bytes)
    - Invalid proof elements
    """

    @pytest.fixture
    def valid_setup(self) -> dict:
        """Create a valid inclusion proof setup for testing."""
        leaf_hashes = [ct_leaf_hash(value) for value in (b"a", b"b", b"c")]
        root = ct_merkle_root(leaf_hashes)
        proof = ct_inclusion_proof(leaf_hashes, 0)
        return {
            "leaf_hash": leaf_hashes[0],
            "leaf_index": 0,
            "tree_size": 3,
            "proof": proof,
            "expected_root": root,
        }

    @pytest.fixture
    def valid_consistency_setup(self) -> dict:
        """Create a valid consistency proof setup for testing."""
        leaves = [ct_leaf_hash(bytes([i])) for i in range(4)]
        first_root = ct_merkle_root(leaves[:2])
        second_root = ct_merkle_root(leaves[:4])
        proof = ct_consistency_proof(leaves[:4], 2)
        return {
            "first_tree_size": 2,
            "second_tree_size": 4,
            "first_root": first_root,
            "second_root": second_root,
            "proof": proof,
        }

    def test_inclusion_verifier_accepts_bytearray_inputs(self, valid_setup: dict) -> None:
        """Verifier should accept bytearray and coerce to bytes."""
        # bytearray is coerced to bytes
        result = verify_ct_inclusion_proof(
            leaf_hash=bytearray(valid_setup["leaf_hash"]),
            leaf_index=valid_setup["leaf_index"],
            tree_size=valid_setup["tree_size"],
            proof=valid_setup["proof"],
            expected_root=bytearray(valid_setup["expected_root"]),
        )
        assert result is True

    def test_inclusion_verifier_rejects_wrong_type_leaf_hash(self, valid_setup: dict) -> None:
        """Verifier returns False for non-bytes leaf_hash."""
        result = verify_ct_inclusion_proof(
            leaf_hash="not bytes",  # type: ignore[arg-type]
            leaf_index=valid_setup["leaf_index"],
            tree_size=valid_setup["tree_size"],
            proof=valid_setup["proof"],
            expected_root=valid_setup["expected_root"],
        )
        assert result is False

    def test_inclusion_verifier_rejects_wrong_length_leaf_hash(self, valid_setup: dict) -> None:
        """Verifier returns False for wrong-length leaf_hash."""
        result = verify_ct_inclusion_proof(
            leaf_hash=b"too short",
            leaf_index=valid_setup["leaf_index"],
            tree_size=valid_setup["tree_size"],
            proof=valid_setup["proof"],
            expected_root=valid_setup["expected_root"],
        )
        assert result is False

    def test_inclusion_verifier_rejects_wrong_type_root(self, valid_setup: dict) -> None:
        """Verifier returns False for non-bytes expected_root."""
        result = verify_ct_inclusion_proof(
            leaf_hash=valid_setup["leaf_hash"],
            leaf_index=valid_setup["leaf_index"],
            tree_size=valid_setup["tree_size"],
            proof=valid_setup["proof"],
            expected_root="not bytes",  # type: ignore[arg-type]
        )
        assert result is False

    def test_inclusion_verifier_rejects_wrong_length_root(self, valid_setup: dict) -> None:
        """Verifier returns False for wrong-length expected_root."""
        result = verify_ct_inclusion_proof(
            leaf_hash=valid_setup["leaf_hash"],
            leaf_index=valid_setup["leaf_index"],
            tree_size=valid_setup["tree_size"],
            proof=valid_setup["proof"],
            expected_root=b"x" * 64,
        )
        assert result is False

    def test_inclusion_verifier_rejects_malformed_proof_element(self, valid_setup: dict) -> None:
        """Verifier returns False for malformed proof elements."""
        bad_proof = [b"short"] if valid_setup["proof"] else [b"short"]
        result = verify_ct_inclusion_proof(
            leaf_hash=valid_setup["leaf_hash"],
            leaf_index=valid_setup["leaf_index"],
            tree_size=valid_setup["tree_size"],
            proof=bad_proof,
            expected_root=valid_setup["expected_root"],
        )
        assert result is False

    def test_consistency_verifier_accepts_bytearray_inputs(self, valid_consistency_setup: dict) -> None:
        """Consistency verifier should accept bytearray and coerce to bytes."""
        result = verify_ct_consistency_proof(
            first_tree_size=valid_consistency_setup["first_tree_size"],
            second_tree_size=valid_consistency_setup["second_tree_size"],
            first_root=bytearray(valid_consistency_setup["first_root"]),
            second_root=bytearray(valid_consistency_setup["second_root"]),
            proof=valid_consistency_setup["proof"],
        )
        assert result is True

    def test_consistency_verifier_rejects_wrong_type_first_root(self, valid_consistency_setup: dict) -> None:
        """Consistency verifier returns False for non-bytes first_root."""
        result = verify_ct_consistency_proof(
            first_tree_size=valid_consistency_setup["first_tree_size"],
            second_tree_size=valid_consistency_setup["second_tree_size"],
            first_root="not bytes",  # type: ignore[arg-type]
            second_root=valid_consistency_setup["second_root"],
            proof=valid_consistency_setup["proof"],
        )
        assert result is False

    def test_consistency_verifier_rejects_wrong_length_root(self, valid_consistency_setup: dict) -> None:
        """Consistency verifier returns False for wrong-length roots."""
        result = verify_ct_consistency_proof(
            first_tree_size=valid_consistency_setup["first_tree_size"],
            second_tree_size=valid_consistency_setup["second_tree_size"],
            first_root=b"short",
            second_root=valid_consistency_setup["second_root"],
            proof=valid_consistency_setup["proof"],
        )
        assert result is False

    def test_consistency_verifier_rejects_malformed_proof_element(self, valid_consistency_setup: dict) -> None:
        """Consistency verifier returns False for malformed proof elements."""
        bad_proof = [b"short"]
        result = verify_ct_consistency_proof(
            first_tree_size=valid_consistency_setup["first_tree_size"],
            second_tree_size=valid_consistency_setup["second_tree_size"],
            first_root=valid_consistency_setup["first_root"],
            second_root=valid_consistency_setup["second_root"],
            proof=bad_proof,
        )
        assert result is False
