from pathlib import Path

import numpy as np
import pytest

from transformation_portal.lux_depth_v3.scene_context import SceneContext
from transformation_portal.lux_depth_v3.scene_groups import SceneGroup, compute_scene_id
from transformation_portal.spatial_ai.reconstruction.contracts import CameraParams


def _camera(tx: float = 0.0) -> CameraParams:
    intrinsics = np.array(
        [[1000.0, 0.0, 32.0], [0.0, 1000.0, 32.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    extrinsics = np.eye(4, dtype=np.float32)
    extrinsics[0, 3] = tx
    return CameraParams(
        intrinsics=intrinsics,
        extrinsics=extrinsics,
        width=64,
        height=64,
    )


def test_scene_context_build_validates_and_builds(tmp_path: Path):
    dataset_root = tmp_path / "input"
    first = dataset_root / "scene_a" / "view_1.jpg"
    second = dataset_root / "scene_a" / "view_2.jpg"
    second.parent.mkdir(parents=True, exist_ok=True)
    first.write_bytes(b"a")
    second.write_bytes(b"b")

    images = (first, second)
    scene = SceneGroup(scene_id=compute_scene_id(images, dataset_root), images=images)

    context = SceneContext.build(
        scene=scene,
        dataset_root=dataset_root,
        cameras=(_camera(0.0), _camera(0.1)),
    )

    assert context.scene_id == scene.scene_id
    assert context.images == images
    assert len(context.cameras) == 2


def test_scene_context_build_raises_on_camera_count_mismatch(tmp_path: Path):
    dataset_root = tmp_path / "input"
    first = dataset_root / "scene_a" / "view_1.jpg"
    second = dataset_root / "scene_a" / "view_2.jpg"
    second.parent.mkdir(parents=True, exist_ok=True)
    first.write_bytes(b"a")
    second.write_bytes(b"b")

    images = (first, second)
    scene = SceneGroup(scene_id=compute_scene_id(images, dataset_root), images=images)

    with pytest.raises(ValueError, match="Camera/image alignment mismatch"):
        SceneContext.build(
            scene=scene,
            dataset_root=dataset_root,
            cameras=(_camera(0.0),),
        )


def test_scene_context_build_raises_on_scene_id_mismatch(tmp_path: Path):
    dataset_root = tmp_path / "input"
    first = dataset_root / "scene_a" / "view_1.jpg"
    second = dataset_root / "scene_a" / "view_2.jpg"
    second.parent.mkdir(parents=True, exist_ok=True)
    first.write_bytes(b"a")
    second.write_bytes(b"b")

    scene = SceneGroup(scene_id="not_the_expected_hash", images=(first, second))

    with pytest.raises(ValueError, match="Scene ID mismatch"):
        SceneContext.build(
            scene=scene,
            dataset_root=dataset_root,
            cameras=(_camera(0.0), _camera(0.1)),
        )
