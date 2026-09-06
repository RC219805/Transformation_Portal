from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from transformation_portal.lux_depth_v3 import reconstruction_manifest as reconstruction_manifest_module
from transformation_portal.lux_depth_v3.reconstruction_manifest import (
    build_reconstruction_manifest,
    load_reconstruction_manifest,
    manifest_image_paths,
    reconstruction_manifest_path,
    write_reconstruction_manifest,
)
from transformation_portal.lux_depth_v3.scene_context import CameraProvenance, CameraWithProvenance, SceneContext
from transformation_portal.lux_depth_v3.scene_groups import SceneGroup, compute_scene_id
from transformation_portal.spatial_ai.reconstruction.contracts import CameraParams

pytestmark = pytest.mark.unit


def _camera(tx: float) -> CameraWithProvenance:
    intrinsics = np.array(
        [[1000.0, 0.0, 32.0], [0.0, 1000.0, 32.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    extrinsics = np.eye(4, dtype=np.float32)
    extrinsics[0, 3] = tx
    return CameraWithProvenance(
        params=CameraParams(
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            width=64,
            height=64,
        ),
        provenance=CameraProvenance(
            source="sidecar",
            confidence="high",
            file="/tmp/scene_cameras.json",
        ),
    )


def _context(tmp_path: Path) -> SceneContext:
    dataset_root = tmp_path / "input"
    first = dataset_root / "scene_a" / "view_1.jpg"
    second = dataset_root / "scene_a" / "view_2.jpg"
    second.parent.mkdir(parents=True, exist_ok=True)
    first.write_bytes(b"a")
    second.write_bytes(b"b")
    images = (first, second)
    scene = SceneGroup(scene_id=compute_scene_id(images, dataset_root), images=images)
    return SceneContext.build(
        scene=scene,
        dataset_root=dataset_root,
        cameras=(_camera(0.0), _camera(0.2)),
        metadata={"grouping_mode": "parent_dir"},
    )


def test_reconstruction_manifest_roundtrip_and_hash_verification(tmp_path: Path):
    context = _context(tmp_path)
    manifest = build_reconstruction_manifest(
        context=context,
        iterations=321,
        tier="apex_research",
    )
    path = write_reconstruction_manifest(manifest=manifest, output_dir=tmp_path / "out")
    loaded = load_reconstruction_manifest(manifest_path=path)

    assert path == reconstruction_manifest_path(scene_id=context.scene_id, output_dir=tmp_path / "out")
    assert loaded.scene_id == context.scene_id
    assert loaded.reconstruction_parameters["iterations"] == 321
    assert loaded.reconstruction_parameters["tier"] == "apex_research"
    assert len(loaded.images) == 2
    assert len(loaded.image_hashes) == 2
    assert manifest_image_paths(loaded)[0].exists()


def test_reconstruction_manifest_rejects_image_hash_drift(tmp_path: Path):
    context = _context(tmp_path)
    manifest = build_reconstruction_manifest(
        context=context,
        iterations=100,
        tier="apex_research",
    )
    path = write_reconstruction_manifest(manifest=manifest, output_dir=tmp_path / "out")

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["image_hashes"][0] = "sha256:deadbeef"
    path.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="image hash mismatch"):
        load_reconstruction_manifest(manifest_path=path)


def test_reconstruction_manifest_uses_captured_digest_overrides_without_source_reads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context(tmp_path)
    digests = ("1" * 64, "2" * 64)

    def reject_source_hashing(_path: Path) -> str:
        raise AssertionError("captured digest overrides must avoid source reads")

    monkeypatch.setattr(reconstruction_manifest_module, "compute_file_sha256", reject_source_hashing)
    manifest = build_reconstruction_manifest(
        context=context,
        iterations=100,
        tier="apex_research",
        image_sha256_overrides=digests,
    )

    assert manifest.image_hashes == tuple(f"sha256:{digest}" for digest in digests)


def test_reconstruction_manifest_preserves_case_sensitive_image_paths(tmp_path: Path) -> None:
    dataset_root = tmp_path / "InputRoot"
    first = dataset_root / "Scene_A" / "View_1.JPG"
    second = dataset_root / "Scene_A" / "View_2.JPG"
    second.parent.mkdir(parents=True)
    first.write_bytes(b"a")
    second.write_bytes(b"b")
    images = (first, second)
    context = SceneContext.build(
        scene=SceneGroup(scene_id=compute_scene_id(images, dataset_root), images=images),
        dataset_root=dataset_root,
        cameras=(_camera(0.0), _camera(0.2)),
        metadata={"grouping_mode": "parent_dir"},
    )

    manifest = build_reconstruction_manifest(
        context=context,
        iterations=100,
        tier="apex_research",
    )
    path = write_reconstruction_manifest(manifest=manifest, output_dir=tmp_path / "out")
    loaded = load_reconstruction_manifest(manifest_path=path)

    assert loaded.images == ("Scene_A/View_1.JPG", "Scene_A/View_2.JPG")
    assert manifest_image_paths(loaded) == images


def test_reconstruction_manifest_canonical_paths_do_not_follow_replaced_root(tmp_path: Path) -> None:
    context = _context(tmp_path)
    dataset_root = context.dataset_root
    preserved_root = tmp_path / "preserved-input"
    replacement_root = tmp_path / "replacement-input"
    dataset_root.rename(preserved_root)
    replacement_image_dir = replacement_root / "scene_a"
    replacement_image_dir.mkdir(parents=True)
    (replacement_image_dir / "view_1.jpg").write_bytes(b"replacement-a")
    (replacement_image_dir / "view_2.jpg").write_bytes(b"replacement-b")
    dataset_root.symlink_to(replacement_root, target_is_directory=True)

    manifest = build_reconstruction_manifest(
        context=context,
        iterations=100,
        tier="apex_research",
        image_sha256_overrides=("1" * 64, "2" * 64),
        paths_are_canonical=True,
    )

    assert manifest.dataset_root == str(dataset_root)
    assert manifest.images == ("scene_a/view_1.jpg", "scene_a/view_2.jpg")
    assert str(replacement_root) not in json.dumps(manifest.to_payload(), sort_keys=True)


@pytest.mark.parametrize("invalid_kind", ["relative", "outside"])
def test_reconstruction_manifest_rejects_invalid_canonical_paths(tmp_path: Path, invalid_kind: str) -> None:
    context = _context(tmp_path)
    if invalid_kind == "relative":
        dataset_root = Path("relative-input")
        images = (dataset_root / "scene_a/view_1.jpg", dataset_root / "scene_a/view_2.jpg")
    else:
        dataset_root = context.dataset_root
        images = (tmp_path / "outside.jpg", context.images[1])
    invalid_context = SceneContext(
        scene_id=context.scene_id,
        dataset_root=dataset_root,
        images=images,
        cameras=context.cameras,
        metadata=context.metadata,
    )

    with pytest.raises(ValueError, match="Canonical scene"):
        build_reconstruction_manifest(
            context=invalid_context,
            iterations=100,
            tier="apex_research",
            image_sha256_overrides=("1" * 64, "2" * 64),
            paths_are_canonical=True,
        )


def test_reconstruction_manifest_canonical_paths_require_captured_digests(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="paths_are_canonical requires image_sha256_overrides"):
        build_reconstruction_manifest(
            context=_context(tmp_path),
            iterations=100,
            tier="apex_research",
            paths_are_canonical=True,
        )
