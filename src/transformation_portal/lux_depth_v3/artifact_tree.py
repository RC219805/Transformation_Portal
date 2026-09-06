"""Transparency-grade artifact tree helpers for run-card v2."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from functools import lru_cache
from typing import Any

from tp.crypto.ct_merkle import (
    ct_leaf_hash,
    ct_merkle_root,
    ct_node_hash,
    verify_ct_inclusion_proof,
)
from transformation_portal.ingest.canonical_json import canonicalize_json

RUN_CARD_ARTIFACT_TREE_ALGORITHM = "ct-sha256-v1"
RUN_CARD_ARTIFACT_LEAF_FORMAT = "tp.run_card.artifact_leaf.v1"
MAX_ARTIFACT_TREE_LEAVES = 524_288
MAX_ARTIFACT_TREE_PROOF_DEPTH = (MAX_ARTIFACT_TREE_LEAVES - 1).bit_length()
MAX_ARTIFACT_TREE_VALIDATION_ERRORS = 64
_ARTIFACT_TREE_TRUNCATION_MESSAGE = (
    "Artifact tree validation stopped after the bounded limit of " f"{MAX_ARTIFACT_TREE_VALIDATION_ERRORS} errors"
)


class _BoundedArtifactTreeErrors(list[str]):
    def __init__(self) -> None:
        super().__init__()
        self.truncated = False

    def mark_truncated(self) -> None:
        if not self.truncated:
            super().append(_ARTIFACT_TREE_TRUNCATION_MESSAGE)
            self.truncated = True

    def append(self, message: str) -> None:
        if self.truncated:
            return
        if len(self) >= MAX_ARTIFACT_TREE_VALIDATION_ERRORS:
            self.mark_truncated()
            return
        super().append(message)

    def stop_if_full(self) -> bool:
        if self.truncated:
            return True
        if len(self) >= MAX_ARTIFACT_TREE_VALIDATION_ERRORS:
            self.mark_truncated()
            return True
        return False


def _require_bounded_leaf_count(count: int, *, field: str) -> None:
    if count > MAX_ARTIFACT_TREE_LEAVES:
        raise ValueError(f"{field} exceeds the bounded limit of {MAX_ARTIFACT_TREE_LEAVES}")


def _all_inclusion_proofs(leaf_hashes: Sequence[bytes]) -> list[list[bytes]]:
    """Build all CT proofs in O(n log n), sharing cached subtree roots."""

    @lru_cache(maxsize=None)
    def subtree_root(start: int, end: int) -> bytes:
        size = end - start
        if size == 1:
            return leaf_hashes[start]
        split = 1 << ((size - 1).bit_length() - 1)
        middle = start + split
        return ct_node_hash(subtree_root(start, middle), subtree_root(middle, end))

    def inclusion_proof(start: int, end: int, leaf_index: int) -> list[bytes]:
        size = end - start
        if size == 1:
            return []
        split = 1 << ((size - 1).bit_length() - 1)
        middle = start + split
        if leaf_index < middle:
            return inclusion_proof(start, middle, leaf_index) + [subtree_root(middle, end)]
        return inclusion_proof(middle, end, leaf_index) + [subtree_root(start, middle)]

    return [inclusion_proof(0, len(leaf_hashes), index) for index in range(len(leaf_hashes))]


def _require_string(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _require_sha256(value: Any, *, field: str) -> str:
    digest = _require_string(value, field=field).lower()
    if len(digest) != 64:
        raise ValueError(f"{field} must be a 64-character sha256 digest")
    try:
        bytes.fromhex(digest)
    except ValueError as exc:
        raise ValueError(f"{field} must be valid hex") from exc
    return digest


def _require_size_bytes(value: Any, *, field: str) -> int:
    if not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def normalize_artifact_leaf_payload(artifact: Mapping[str, Any]) -> dict[str, Any]:
    """Project a run-card artifact entry into the committed CT leaf payload."""
    return {
        "relative_path": _require_string(artifact.get("relative_path"), field="relative_path"),
        "artifact_type": _require_string(artifact.get("artifact_type"), field="artifact_type"),
        "size_bytes": _require_size_bytes(artifact.get("size_bytes"), field="size_bytes"),
        "sha256": _require_sha256(artifact.get("sha256"), field="sha256"),
    }


def canonical_artifact_leaf_bytes(artifact: Mapping[str, Any]) -> bytes:
    """Serialize the committed artifact leaf record under tp.canonical.json.v1."""
    return canonicalize_json(normalize_artifact_leaf_payload(artifact))


def artifact_leaf_sha256(artifact: Mapping[str, Any]) -> str:
    """Return the CT leaf digest hex for the artifact leaf payload."""
    return ct_leaf_hash(canonical_artifact_leaf_bytes(artifact)).hex()


def build_artifact_tree(
    artifact_index: Sequence[Mapping[str, Any]],
    *,
    include_proofs: bool = True,
) -> dict[str, Any]:
    """Build the transparency-grade artifact tree payload for run-card v2."""
    _require_bounded_leaf_count(len(artifact_index), field="artifact_index")
    normalized_artifacts = [normalize_artifact_leaf_payload(artifact) for artifact in artifact_index]
    normalized_artifacts.sort(key=lambda item: item["relative_path"])

    leaf_hashes = [ct_leaf_hash(canonicalize_json(artifact)) for artifact in normalized_artifacts]
    artifacts_with_leaf_hash = [
        {
            **artifact,
            "leaf_sha256": leaf_hash.hex(),
        }
        for artifact, leaf_hash in zip(normalized_artifacts, leaf_hashes)
    ]

    artifact_tree: dict[str, Any] = {
        "algorithm": RUN_CARD_ARTIFACT_TREE_ALGORITHM,
        "leaf_format": RUN_CARD_ARTIFACT_LEAF_FORMAT,
        "leaf_count": len(artifacts_with_leaf_hash),
        "root_sha256": ct_merkle_root(leaf_hashes).hex(),
        "artifacts": artifacts_with_leaf_hash,
    }
    if include_proofs:
        proofs: list[dict[str, Any]] = []
        all_sibling_hashes = _all_inclusion_proofs(leaf_hashes) if leaf_hashes else []
        for index, (artifact, sibling_hashes) in enumerate(zip(artifacts_with_leaf_hash, all_sibling_hashes)):
            proof_steps: list[dict[str, str]] = []
            fn = index
            sn = len(leaf_hashes) - 1
            sibling_index = 0
            while sn > 0 and sibling_index < len(sibling_hashes):
                if fn % 2 == 1:
                    position = "left"
                    proof_steps.append({"position": position, "hash": sibling_hashes[sibling_index].hex()})
                    sibling_index += 1
                elif fn < sn:
                    position = "right"
                    proof_steps.append({"position": position, "hash": sibling_hashes[sibling_index].hex()})
                    sibling_index += 1
                fn //= 2
                sn //= 2
            proofs.append(
                {
                    "relative_path": artifact["relative_path"],
                    "leaf_index": index,
                    "leaf_sha256": artifact["leaf_sha256"],
                    "path": proof_steps,
                }
            )
        artifact_tree["proofs"] = proofs
    return artifact_tree


def verify_artifact_tree_payload(
    artifact_tree: Mapping[str, Any],
    *,
    artifact_index: Sequence[Mapping[str, Any]],
) -> list[str]:
    """Validate a run-card v2 artifact_tree block against the artifact index."""
    errors = _BoundedArtifactTreeErrors()

    if len(artifact_index) > MAX_ARTIFACT_TREE_LEAVES:
        return [f"artifact_index exceeds the bounded limit of {MAX_ARTIFACT_TREE_LEAVES}"]

    if artifact_tree.get("algorithm") != RUN_CARD_ARTIFACT_TREE_ALGORITHM:
        errors.append(f"artifact_tree.algorithm must be {RUN_CARD_ARTIFACT_TREE_ALGORITHM!r}")
    if artifact_tree.get("leaf_format") != RUN_CARD_ARTIFACT_LEAF_FORMAT:
        errors.append(f"artifact_tree.leaf_format must be {RUN_CARD_ARTIFACT_LEAF_FORMAT!r}")

    artifacts = artifact_tree.get("artifacts")
    if not isinstance(artifacts, list):
        return errors + ["artifact_tree.artifacts must be a list"]
    if len(artifacts) > MAX_ARTIFACT_TREE_LEAVES:
        return errors + [f"artifact_tree.artifacts exceeds the bounded limit of {MAX_ARTIFACT_TREE_LEAVES}"]

    expected_tree = build_artifact_tree(artifact_index, include_proofs=False)
    if artifact_tree.get("leaf_count") != len(artifacts):
        errors.append("artifact_tree.leaf_count must equal len(artifact_tree.artifacts)")
    if artifact_tree.get("root_sha256") != expected_tree["root_sha256"]:
        errors.append(
            "artifact_tree.root_sha256 mismatch: "
            f"provided={artifact_tree.get('root_sha256')}, expected={expected_tree['root_sha256']}"
        )

    expected_artifacts = expected_tree["artifacts"]
    if artifacts != expected_artifacts:
        errors.append("artifact_tree.artifacts do not match committed artifact_index leaf payloads")

    proofs = artifact_tree.get("proofs")
    if proofs is None:
        return errors
    if not isinstance(proofs, list):
        return errors + ["artifact_tree.proofs must be a list when present"]
    if len(proofs) > MAX_ARTIFACT_TREE_LEAVES:
        return errors + [f"artifact_tree.proofs exceeds the bounded limit of {MAX_ARTIFACT_TREE_LEAVES}"]
    if len(proofs) != len(expected_artifacts):
        errors.append("artifact_tree.proofs must contain exactly one entry per artifact")

    proof_by_path: dict[str, Mapping[str, Any]] = {}
    for proof_entry in proofs:
        if errors.stop_if_full():
            return list(errors)
        if not isinstance(proof_entry, Mapping):
            errors.append("artifact_tree.proofs entries must be objects")
            continue
        relative_path = proof_entry.get("relative_path")
        if isinstance(relative_path, str) and relative_path:
            proof_by_path[relative_path] = proof_entry

    expected_root_bytes = bytes.fromhex(expected_tree["root_sha256"])
    for expected_index, expected_artifact in enumerate(expected_artifacts):
        if errors.stop_if_full():
            return list(errors)
        relative_path = expected_artifact["relative_path"]
        proof_entry = proof_by_path.get(relative_path)
        if proof_entry is None:
            errors.append(f"artifact_tree.proofs missing entry for {relative_path}")
            continue
        if proof_entry.get("leaf_sha256") != expected_artifact["leaf_sha256"]:
            errors.append(f"artifact_tree proof leaf_sha256 mismatch for {relative_path}")
            continue
        leaf_index = proof_entry.get("leaf_index")
        if not isinstance(leaf_index, int):
            errors.append(f"artifact_tree proof leaf_index must be integer for {relative_path}")
            continue
        if leaf_index != expected_index:
            errors.append(
                f"artifact_tree proof leaf_index mismatch for {relative_path}: "
                f"provided={leaf_index}, expected={expected_index}"
            )
            continue
        proof_path = proof_entry.get("path")
        if not isinstance(proof_path, list):
            errors.append(f"artifact_tree proof path must be a list for {relative_path}")
            continue
        if len(proof_path) > MAX_ARTIFACT_TREE_PROOF_DEPTH:
            errors.append(
                f"artifact_tree proof path exceeds the bounded limit of {MAX_ARTIFACT_TREE_PROOF_DEPTH} "
                f"for {relative_path}"
            )
            continue
        sibling_hashes: list[bytes] = []
        position_error = False
        # CT audit-path position validation state.
        # fn = first node index (current leaf/node position in the level)
        # sn = subtree node count - 1 (last valid index in the level)
        # These are used to walk the tree structure and determine expected sibling positions.
        fn = leaf_index
        sn = len(expected_artifacts) - 1
        step_index = 0
        for step in proof_path:
            if errors.stop_if_full():
                return list(errors)
            if not isinstance(step, Mapping):
                errors.append(f"artifact_tree proof path steps must be objects for {relative_path}")
                sibling_hashes = []
                break
            try:
                sibling_hashes.append(bytes.fromhex(_require_sha256(step.get("hash"), field="proof.hash")))
            except ValueError as exc:
                errors.append(f"artifact_tree proof hash invalid for {relative_path}: {exc}")
                sibling_hashes = []
                break
            step_position = step.get("position")
            if step_position is not None:
                # Walk the tree state to find where the next sibling is expected.
                # If fn is odd, sibling is on the left; if fn < sn and even, sibling is on right.
                # Skip levels where no sibling exists (fn == sn and both even).
                while sn > 0:
                    if fn % 2 == 1:
                        expected_position = "left"
                        break
                    elif fn < sn:
                        expected_position = "right"
                        break
                    fn //= 2
                    sn //= 2
                else:
                    expected_position = None
                if expected_position is not None and step_position != expected_position:
                    errors.append(
                        f"artifact_tree proof path position mismatch at step {step_index} for {relative_path}: "
                        f"provided={step_position}, expected={expected_position}"
                    )
                    position_error = True
                    break
                fn //= 2
                sn //= 2
            step_index += 1
        if position_error:
            continue
        if not sibling_hashes and proof_path:
            continue
        if not verify_ct_inclusion_proof(
            leaf_hash=bytes.fromhex(expected_artifact["leaf_sha256"]),
            leaf_index=leaf_index,
            tree_size=len(expected_artifacts),
            proof=sibling_hashes,
            expected_root=expected_root_bytes,
        ):
            errors.append(f"artifact_tree inclusion proof verification failed for {relative_path}")
    return list(errors)
