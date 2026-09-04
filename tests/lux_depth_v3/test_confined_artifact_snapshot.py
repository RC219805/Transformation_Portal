from __future__ import annotations

import hashlib
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from transformation_portal.lux_depth_v3.execution_evidence import (
    ArtifactEvidenceError,
    ConfinedArtifactSnapshot,
    read_confined_artifact_snapshot,
)

pytestmark = pytest.mark.unit


def test_confined_snapshot_captures_exact_bytes_and_matches_record(tmp_path: Path) -> None:
    output_root = tmp_path / "output"
    artifact_path = output_root / "nested" / "artifact.bin"
    artifact_path.parent.mkdir(parents=True)
    artifact_bytes = b"prepared-artifact\x00bytes"
    artifact_path.write_bytes(artifact_bytes)

    snapshot = read_confined_artifact_snapshot(
        output_root,
        artifact_path,
        context="prepared artifact",
        max_bytes=len(artifact_bytes),
    )

    assert snapshot.data == artifact_bytes
    assert snapshot.relative_path == "nested/artifact.bin"
    assert snapshot.sha256 == hashlib.sha256(artifact_bytes).hexdigest()
    assert snapshot.size_bytes == len(artifact_bytes)
    assert snapshot.matches(
        {
            "path": "nested/artifact.bin",
            "sha256": hashlib.sha256(artifact_bytes).hexdigest(),
            "size_bytes": len(artifact_bytes),
            "media_type": "application/octet-stream",
        }
    )

    for changed_record in (
        {
            "path": "other/artifact.bin",
            "sha256": snapshot.sha256,
            "size_bytes": snapshot.size_bytes,
        },
        {
            "path": snapshot.relative_path,
            "sha256": "0" * 64,
            "size_bytes": snapshot.size_bytes,
        },
        {
            "path": snapshot.relative_path,
            "sha256": snapshot.sha256,
            "size_bytes": snapshot.size_bytes + 1,
        },
        {
            "path": snapshot.relative_path,
            "sha256": snapshot.sha256,
            "size_bytes": True,
        },
    ):
        assert not snapshot.matches(changed_record)

    with pytest.raises(FrozenInstanceError):
        snapshot.size_bytes = 0  # type: ignore[misc]


def test_confined_snapshot_enforces_caller_byte_limit(tmp_path: Path) -> None:
    output_root = tmp_path / "output"
    output_root.mkdir()
    artifact_path = output_root / "artifact.bin"
    artifact_path.write_bytes(b"four")

    with pytest.raises(ArtifactEvidenceError) as exc_info:
        read_confined_artifact_snapshot(
            output_root,
            artifact_path,
            context="prepared artifact",
            max_bytes=3,
        )

    assert exc_info.value.code == "artifact_too_large"


def test_confined_snapshot_rejects_symlink_artifact(tmp_path: Path) -> None:
    output_root = tmp_path / "output"
    output_root.mkdir()
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"outside")
    artifact_path = output_root / "artifact.bin"
    artifact_path.symlink_to(outside)

    with pytest.raises(ArtifactEvidenceError) as exc_info:
        read_confined_artifact_snapshot(
            output_root,
            artifact_path,
            context="prepared artifact",
            max_bytes=64,
        )

    assert exc_info.value.code == "symlink_forbidden"


@pytest.mark.parametrize("max_bytes", [-1, True, 1.5])
def test_confined_snapshot_rejects_invalid_byte_limit(tmp_path: Path, max_bytes: object) -> None:
    output_root = tmp_path / "output"
    output_root.mkdir()
    artifact_path = output_root / "artifact.bin"
    artifact_path.write_bytes(b"data")

    with pytest.raises(ValueError, match="max_bytes"):
        read_confined_artifact_snapshot(
            output_root,
            artifact_path,
            context="prepared artifact",
            max_bytes=max_bytes,  # type: ignore[arg-type]
        )


def test_snapshot_type_is_public() -> None:
    snapshot = ConfinedArtifactSnapshot(
        data=b"",
        relative_path="empty.bin",
        sha256=hashlib.sha256(b"").hexdigest(),
        size_bytes=0,
    )

    assert snapshot.matches(
        {
            "path": "empty.bin",
            "sha256": hashlib.sha256(b"").hexdigest(),
            "size_bytes": 0,
        }
    )
