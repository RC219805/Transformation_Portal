"""Tests for Lux run-card v2 artifact trees."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from transformation_portal.lux_depth_v3.artifact_tree import (
    MAX_ARTIFACT_TREE_LEAVES,
    MAX_ARTIFACT_TREE_PROOF_DEPTH,
    MAX_ARTIFACT_TREE_VALIDATION_ERRORS,
    RUN_CARD_ARTIFACT_LEAF_FORMAT,
    RUN_CARD_ARTIFACT_TREE_ALGORITHM,
    artifact_leaf_sha256,
    build_artifact_tree,
    verify_artifact_tree_payload,
)

pytestmark = pytest.mark.unit


def _artifact(relative_path: str, *, digest: str) -> dict[str, object]:
    return {
        "artifact_type": "depth_u16_png",
        "path": relative_path,
        "relative_path": relative_path,
        "size_bytes": 123,
        "sha256": digest,
    }


def test_build_artifact_tree_sorts_and_hashes_artifacts() -> None:
    artifact_index = [
        _artifact("manifests/b.json", digest="b" * 64),
        _artifact("depth/a.png", digest="a" * 64),
    ]

    tree = build_artifact_tree(artifact_index, include_proofs=True)

    assert tree["algorithm"] == RUN_CARD_ARTIFACT_TREE_ALGORITHM
    assert tree["leaf_format"] == RUN_CARD_ARTIFACT_LEAF_FORMAT
    assert tree["leaf_count"] == 2
    assert [entry["relative_path"] for entry in tree["artifacts"]] == [
        "depth/a.png",
        "manifests/b.json",
    ]
    assert tree["artifacts"][0]["leaf_sha256"] == artifact_leaf_sha256(_artifact("depth/a.png", digest="a" * 64))


def test_verify_artifact_tree_payload_accepts_built_tree() -> None:
    artifact_index = [
        _artifact("depth/a.png", digest="a" * 64),
        _artifact("depth/b.png", digest="b" * 64),
        _artifact("manifests/c.json", digest="c" * 64),
    ]

    tree = build_artifact_tree(artifact_index, include_proofs=True)
    assert verify_artifact_tree_payload(tree, artifact_index=artifact_index) == []


def test_build_artifact_tree_can_omit_proofs() -> None:
    artifact_index = [
        _artifact("depth/a.png", digest="a" * 64),
        _artifact("depth/b.png", digest="b" * 64),
    ]

    tree = build_artifact_tree(artifact_index, include_proofs=False)

    assert "proofs" not in tree
    assert verify_artifact_tree_payload(tree, artifact_index=artifact_index) == []


def test_verify_artifact_tree_payload_rejects_tampered_root() -> None:
    artifact_index = [
        _artifact("depth/a.png", digest="a" * 64),
        _artifact("depth/b.png", digest="b" * 64),
    ]
    tree = build_artifact_tree(artifact_index, include_proofs=True)
    tree["root_sha256"] = "f" * 64

    errors = verify_artifact_tree_payload(tree, artifact_index=artifact_index)
    assert any("artifact_tree.root_sha256 mismatch" in error for error in errors)


def test_verify_artifact_tree_payload_rejects_wrong_leaf_index() -> None:
    artifact_index = [
        _artifact("depth/a.png", digest="a" * 64),
        _artifact("depth/b.png", digest="b" * 64),
    ]
    tree = build_artifact_tree(artifact_index, include_proofs=True)
    tree["proofs"][0]["leaf_index"] = 1

    errors = verify_artifact_tree_payload(tree, artifact_index=artifact_index)
    assert any("artifact_tree proof leaf_index mismatch" in error for error in errors)


def test_verify_artifact_tree_payload_rejects_wrong_position() -> None:
    artifact_index = [
        _artifact("depth/a.png", digest="a" * 64),
        _artifact("depth/b.png", digest="b" * 64),
    ]
    tree = build_artifact_tree(artifact_index, include_proofs=True)
    for proof in tree["proofs"]:
        for step in proof["path"]:
            if step["position"] == "left":
                step["position"] = "right"
            elif step["position"] == "right":
                step["position"] = "left"

    errors = verify_artifact_tree_payload(tree, artifact_index=artifact_index)
    assert any("artifact_tree proof path position mismatch" in error for error in errors)


def test_build_artifact_tree_rejects_oversized_index_before_hashing() -> None:
    artifact_index = [_artifact("depth/a.png", digest="a" * 64)] * (MAX_ARTIFACT_TREE_LEAVES + 1)

    with pytest.raises(ValueError, match="artifact_index exceeds the bounded limit"):
        build_artifact_tree(artifact_index, include_proofs=False)


def test_verify_artifact_tree_rejects_oversized_payload_before_rebuild() -> None:
    artifact_index = [_artifact("depth/a.png", digest="a" * 64)]
    tree = build_artifact_tree(artifact_index, include_proofs=False)
    tree["artifacts"] = tree["artifacts"] * (MAX_ARTIFACT_TREE_LEAVES + 1)

    errors = verify_artifact_tree_payload(tree, artifact_index=artifact_index)

    assert errors == [f"artifact_tree.artifacts exceeds the bounded limit of {MAX_ARTIFACT_TREE_LEAVES}"]


def test_verify_artifact_tree_rejects_oversized_proof_path() -> None:
    artifact_index = [_artifact("depth/a.png", digest="a" * 64)]
    tree = build_artifact_tree(artifact_index, include_proofs=True)
    tree["proofs"][0]["path"] = [{"position": "left", "hash": "a" * 64}] * (MAX_ARTIFACT_TREE_PROOF_DEPTH + 1)

    errors = verify_artifact_tree_payload(tree, artifact_index=artifact_index)

    assert any("proof path exceeds the bounded limit" in error for error in errors)


def test_verify_artifact_tree_caps_missing_proof_diagnostics() -> None:
    artifact_index = [
        _artifact(f"depth/{index:04d}.png", digest=f"{index:064x}")
        for index in range(MAX_ARTIFACT_TREE_VALIDATION_ERRORS + 10)
    ]
    tree = build_artifact_tree(artifact_index, include_proofs=True)
    tree["proofs"] = []

    errors = verify_artifact_tree_payload(tree, artifact_index=artifact_index)

    assert len(errors) == MAX_ARTIFACT_TREE_VALIDATION_ERRORS + 1
    assert errors[-1] == (
        "Artifact tree validation stopped after the bounded limit of " f"{MAX_ARTIFACT_TREE_VALIDATION_ERRORS} errors"
    )


def test_build_all_artifact_proofs_shares_subtree_work() -> None:
    import transformation_portal.lux_depth_v3.artifact_tree as artifact_tree_module

    artifact_index = [_artifact(f"depth/{index:04d}.png", digest=f"{index:064x}") for index in range(257)]
    node_hash_calls = 0
    original_node_hash = artifact_tree_module.ct_node_hash

    def counting_node_hash(left: bytes, right: bytes) -> bytes:
        nonlocal node_hash_calls
        node_hash_calls += 1
        return original_node_hash(left, right)

    with patch.object(artifact_tree_module, "ct_node_hash", side_effect=counting_node_hash):
        tree = build_artifact_tree(artifact_index, include_proofs=True)

    assert len(tree["proofs"]) == len(artifact_index)
    assert node_hash_calls <= len(artifact_index) * 2
