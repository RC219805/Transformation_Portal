"""Tests for CT-style Merkle helpers."""

from __future__ import annotations

import hashlib

import pytest

from tp.crypto.ct_merkle import (
    AuditPath,
    MerkleRoot,
    Sha256Digest,
    ct_inclusion_proof,
    ct_leaf_hash,
    ct_merkle_root,
    validate_sha256_digest,
    verify_ct_inclusion_proof,
)

pytestmark = pytest.mark.unit


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

    def test_aliases_importable_from_package(self) -> None:
        """Type aliases should be importable from tp.crypto."""
        from tp.crypto import AuditPath, MerkleRoot, Sha256Digest

        assert Sha256Digest is bytes
        assert MerkleRoot is bytes
