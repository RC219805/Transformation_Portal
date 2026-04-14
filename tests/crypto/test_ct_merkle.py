"""Tests for CT-style Merkle helpers."""

from __future__ import annotations

import pytest

from tp.crypto.ct_merkle import (
    ct_inclusion_proof,
    ct_leaf_hash,
    ct_merkle_root,
    verify_ct_inclusion_proof,
)

pytestmark = pytest.mark.unit


def test_ct_merkle_root_empty_tree_uses_sha256_empty() -> None:
    assert ct_merkle_root([]).hex() == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


def test_ct_merkle_root_single_leaf_equals_leaf_hash() -> None:
    leaf = ct_leaf_hash(b"artifact-a")
    assert ct_merkle_root([leaf]) == leaf


def test_ct_inclusion_proof_round_trip_for_odd_tree() -> None:
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


def test_ct_inclusion_proof_rejects_wrong_leaf() -> None:
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
