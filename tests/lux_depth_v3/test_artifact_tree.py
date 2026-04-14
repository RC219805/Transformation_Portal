"""Tests for Lux run-card v2 artifact trees."""

from __future__ import annotations

import pytest

from transformation_portal.lux_depth_v3.artifact_tree import (
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
